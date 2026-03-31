import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseEncoder, EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

_MODELS = {
    "clip":       ("ViT-B-16",     "openai"),
    "clip-large": ("ViT-L-14",     "openai"),
    "clip-336":   ("ViT-L-14-336", "openai"),
}


class CLIPEncoder(BaseEncoder):

    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__(model_name, device)
        self._load()

    def _load(self) -> None:
        import open_clip

        key = self.model_name.lower()
        if key not in _MODELS:
            raise ValueError(f"Unknown CLIP variant '{key}'. Supported: {list(_MODELS)}")
        oc_name, pretrained = _MODELS[key]

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            oc_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(oc_name)

        self._use_timm = hasattr(self.model.visual, "trunk")

        if self._use_timm:
            trunk = self.model.visual.trunk
            n_blocks = len(trunk.blocks)
            self._target_layer = max(0, n_blocks - 3)
            pe = trunk.patch_embed
            self.config = EncoderConfig(
                image_size=pe.img_size[0] if hasattr(pe, "img_size") else 224,
                embed_dim=self.model.visual.output_dim if hasattr(self.model.visual, "output_dim") else trunk.embed_dim,
                patch_size=pe.patch_size[0] if hasattr(pe, "patch_size") else 14,
                grid_size=pe.grid_size[0],
            )
            logger.info(f"CLIP (timm): {oc_name} | {self.config.image_size}px | layer {self._target_layer}/{n_blocks}")
        else:
            resblocks = self.model.visual.transformer.resblocks
            n_blocks = len(resblocks)
            self._target_layer = max(0, n_blocks - 3)
            vis = self.model.visual
            image_size = 336 if "336" in oc_name else 224
            patch_size = vis.conv1.kernel_size[0]
            self.config = EncoderConfig(
                image_size=image_size,
                embed_dim=self.model.visual.output_dim,
                patch_size=patch_size,
                grid_size=image_size // patch_size,
            )
            logger.info(f"CLIP (openai): {oc_name} | {image_size}px | layer {self._target_layer}/{n_blocks}")

    def encode_image(self, image: Image.Image) -> EncoderOutput:
        return self.encode_images_batch([image])[0]

    def encode_images_batch(self, images: List[Image.Image]) -> List[EncoderOutput]:
        if not images:
            return []
        pixels = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeddings, patch_attn = (
            self._forward_timm(pixels) if self._use_timm else self._forward_openai(pixels)
        )
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        emb_np = embeddings.cpu().numpy()
        inv_maps = self._build_inverse_attention(patch_attn)
        outputs = []
        for i in range(len(images)):
            outputs.append(EncoderOutput(embedding=emb_np[i], attention=inv_maps[i]))
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

    def _forward_timm(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trunk = self.model.visual.trunk
        full_attn = None

        with torch.no_grad():
            x = trunk.patch_embed(pixels)
            x = trunk._pos_embed(x)

            for i, block in enumerate(trunk.blocks):
                if i == self._target_layer:
                    full_attn, x = self._timm_block_with_attn(block, x)
                else:
                    x = block(x)

            x = trunk.norm(x)
            x = x[:, 0]
            if hasattr(trunk, "fc_norm") and trunk.fc_norm is not None:
                x = trunk.fc_norm(x)
            embeddings = self.model.visual.head(x) if hasattr(self.model.visual, "head") else x

        patch_attn = full_attn[:, :, 1:, 1:]  # remove CLS
        return embeddings, patch_attn

    def _timm_block_with_attn(self, block, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        m = block.attn
        head_dim = C // m.num_heads

        x_norm = block.norm1(x)
        qkv = m.qkv(x_norm).reshape(B, N, 3, m.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)  # [B, H, N, N] where N includes CLS

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = m.proj(out)
        if hasattr(m, "proj_drop"):
            out = m.proj_drop(out)

        x = x + block.drop_path1(block.ls1(out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return attn, x

    def _forward_openai(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vis = self.model.visual
        full_attn = None

        with torch.no_grad():
            x = vis.conv1(pixels).reshape(pixels.shape[0], vis.conv1.out_channels, -1).permute(0, 2, 1)
            cls = vis.class_embedding.to(dtype=x.dtype, device=x.device)
            cls = cls[None, None, :].expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + vis.positional_embedding.to(x.dtype)
            x = vis.ln_pre(x)

            for i, block in enumerate(vis.transformer.resblocks):
                if i == self._target_layer:
                    full_attn, x = self._openai_block_with_attn(block, x)
                else:
                    x = block(x)

            x = vis.ln_post(x[:, 0, :])
            embeddings = x @ vis.proj if vis.proj is not None else x

        patch_attn = full_attn[:, :, 1:, 1:]  # remove CLS
        return embeddings, patch_attn

    def _openai_block_with_attn(self, block, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        m = block.attn
        num_heads = m.num_heads
        head_dim = C // num_heads

        x_norm = block.ln_1(x)
        qkv = F.linear(x_norm, m.in_proj_weight, m.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = F.linear(out, m.out_proj.weight, m.out_proj.bias)

        x = x + out
        x = x + block.mlp(block.ln_2(x))
        return attn, x
