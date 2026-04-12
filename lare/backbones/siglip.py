import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseEncoder, EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

_MODELS = {
    "siglip":        ("ViT-B-16-SigLIP",           "webli"),
    "siglip-so400m": ("ViT-SO400M-14-SigLIP-384",  "webli"),
    "siglip2":       ("ViT-SO400M-16-SigLIP2-384", "webli"),
}


class SigLIPEncoder(BaseEncoder):

    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__(model_name, device)
        self._load()

    def _load(self) -> None:
        import open_clip

        key = self.model_name.lower()
        if key not in _MODELS:
            raise ValueError(f"Unknown SigLIP variant '{key}'. Supported: {list(_MODELS)}")
        oc_name, pretrained = _MODELS[key]

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            oc_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(oc_name)

        trunk = self.model.visual.trunk
        n_blocks = len(trunk.blocks)
        self._target_layer = max(0, n_blocks - 7)

        pe = trunk.patch_embed
        _img_size = pe.img_size[0] if hasattr(pe, "img_size") else 384
        _patch_size = pe.patch_size[0] if hasattr(pe, "patch_size") else 14
        self.config = EncoderConfig(
            image_size=_img_size,
            embed_dim=trunk.embed_dim,
            patch_size=_patch_size,
            grid_size=pe.grid_size[0] if hasattr(pe, "grid_size") else _img_size // _patch_size,
        )
        logger.info(f"SigLIP: {oc_name} | {self.config.image_size}px | layer {self._target_layer}/{n_blocks}")

    def encode_image(self, image: Image.Image) -> EncoderOutput:
        return self.encode_images_batch([image])[0]

    def encode_images_batch(self, images: List[Image.Image]) -> List[EncoderOutput]:
        if not images:
            return []
        pixels = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeddings, patch_attn, patch_tokens = self._forward(pixels)
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        emb_np = embeddings.cpu().numpy()
        inv_maps = self._build_inverse_attention(patch_attn)
        # Project patches to shared embedding space
        patch_embs = self._project_patches(patch_tokens)
        outputs = []
        for i in range(len(images)):
            outputs.append(EncoderOutput(
                embedding=emb_np[i],
                attention=inv_maps[i],
                patch_embeddings=patch_embs[i],
            ))
        return outputs

    def encode_text(self, text) -> np.ndarray:
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        result = emb.cpu().numpy()
        return result[0] if is_single else result

    def encode_text_tokens(self, text) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Encode text and return both [CLS] embedding and per-token embeddings.

        Returns:
            cls_embs: [N, D] pooled text embeddings
            token_embs_list: list of [n_tokens, D] per text (content tokens only)
        """
        texts = [text] if isinstance(text, str) else list(text)
        tm = self.model.text
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            x = tm.token_embedding(tokens)
            x = x + tm.positional_embedding
            x = x.permute(1, 0, 2)
            x = tm.transformer(x)
            x = x.permute(1, 0, 2)
            x = tm.ln_final(x)
            all_token_feats = tm.text_projection(x)  # [B, seq_len, D]
            all_token_feats = all_token_feats / (all_token_feats.norm(dim=-1, keepdim=True) + 1e-8)

            # Pooled (last non-pad token for SigLIP)
            cls_embs = self.model.encode_text(tokens)
            cls_embs = cls_embs / (cls_embs.norm(dim=-1, keepdim=True) + 1e-8)

        cls_np = cls_embs.cpu().numpy()
        token_feats_np = all_token_feats.cpu().numpy()

        # Extract only content tokens (non-padding)
        token_embs_list = []
        tokens_np = tokens.cpu().numpy()
        pad_id = tm.pad_id if hasattr(tm, 'pad_id') else 0
        # SigLIP uses 1 as pad, SigLIP2 uses 0 — also exclude EOS
        for i in range(len(texts)):
            t = tokens_np[i]
            # Find content: non-pad and non-zero (handles both models)
            mask = (t != pad_id) & (t != 0) if pad_id != 0 else (t != 0)
            # For SigLIP (pad=0 reported but actual pad=1): use first contiguous non-1 block
            if pad_id == 0 and 1 in t:
                first_pad = np.where(t == 1)[0]
                if len(first_pad) > 0 and first_pad[0] > 0:
                    mask = np.zeros(len(t), dtype=bool)
                    mask[:first_pad[0]] = True
            token_embs_list.append(token_feats_np[i][mask])

        return cls_np, token_embs_list

    def _forward(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trunk = self.model.visual.trunk
        patch_attn = None

        with torch.no_grad():
            x = trunk.patch_embed(pixels)
            x = trunk.pos_drop(x + trunk.pos_embed)
            if hasattr(trunk, "patch_drop") and trunk.patch_drop is not None:
                x = trunk.patch_drop(x)

            for i, block in enumerate(trunk.blocks):
                if i == self._target_layer:
                    patch_attn, x = self._block_with_attn(block, x)
                else:
                    x = block(x)

            x = trunk.norm(x)
            patch_tokens = x  # [B, N_patches, D] — before pooling

            if hasattr(trunk, "attn_pool") and trunk.attn_pool is not None:
                x = trunk.attn_pool(x)
            else:
                x = x.mean(dim=1)
            if hasattr(trunk, "fc_norm") and trunk.fc_norm is not None:
                x = trunk.fc_norm(x)
            embeddings = self.model.visual.head(x) if hasattr(self.model.visual, "head") else x

        return embeddings, patch_attn, patch_tokens

    def _project_patches(self, patch_tokens: torch.Tensor) -> np.ndarray:
        """Project patch tokens through the visual head to the shared embedding space."""
        with torch.no_grad():
            head = self.model.visual.head if hasattr(self.model.visual, "head") else None
            if head is not None:
                B, N, D = patch_tokens.shape
                flat = patch_tokens.reshape(B * N, D)
                projected = head(flat).reshape(B, N, -1)
            else:
                projected = patch_tokens
            projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        return projected.cpu().numpy()

    def _block_with_attn(self, block, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        m = block.attn
        head_dim = C // m.num_heads

        x_norm = block.norm1(x)
        qkv = m.qkv(x_norm).reshape(B, N, 3, m.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)  # [B, H, N, N] — all tokens are patches (no CLS)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = m.proj(out)
        if hasattr(m, "proj_drop"):
            out = m.proj_drop(out)

        x = x + block.drop_path1(block.ls1(out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return attn, x
