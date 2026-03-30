from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch


@dataclass
class EncoderConfig:
    image_size: int = 224
    embed_dim: int = 768
    patch_size: int = 16
    grid_size: int = 14


@dataclass
class EncoderOutput:
    embedding: np.ndarray  # [D] L2-normalized
    attention: np.ndarray  # [H, W] inverse attention map in [0, 1]


class BaseEncoder(ABC):

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = EncoderConfig()

    @abstractmethod
    def encode_image(self, image: Image.Image) -> EncoderOutput: ...

    @abstractmethod
    def encode_images_batch(self, images: List[Image.Image]) -> List[EncoderOutput]: ...

    @abstractmethod
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray: ...

    def _normalize_map(self, attn: np.ndarray) -> np.ndarray:
        lo, hi = attn.min(), attn.max()
        return (attn - lo) / (hi - lo + 1e-8)

    def _build_inverse_attention(
        self,
        patch_attn: torch.Tensor,
        top_k_heads: int = 4,
    ) -> np.ndarray:
        """
        Shared attention processing used by both CLIP and SigLIP.

        Input `patch_attn` must contain only patch-to-patch attention weights
        (CLS row/column already removed for CLIP). Shape: [B, H, N_patch, N_patch].

        Paper Eq. 1: a^(h)_i = sum_j A^(h)_{j,i}
        Column-wise sum gives the total attention each patch receives.
        Patches that receive little attention are "overlooked" background regions.

        Returns [B, image_size, image_size] or [image_size, image_size] for B=1.
        """
        B, H, N, _ = patch_attn.shape
        grid = self.config.grid_size
        size = self.config.image_size
        attn_np = patch_attn.cpu().numpy()

        batch_maps = []
        for b in range(B):
            head_maps, variances = [], []
            for h in range(H):
                received = attn_np[b, h].sum(axis=0).reshape(grid, grid)
                normalized = self._normalize_map(received)
                head_maps.append(normalized)
                variances.append(normalized.var())

            top_k = min(top_k_heads, H)
            best = np.argsort(variances)[-top_k:]
            mean_attn = np.mean([head_maps[i] for i in best], axis=0)

            resized = cv2.resize(mean_attn, (size, size), interpolation=cv2.INTER_LINEAR)
            batch_maps.append((1.0 - resized).astype(np.float32))

        return np.stack(batch_maps) if B > 1 else batch_maps[0]
