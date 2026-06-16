from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class RegionDetector:
    """Sliding-window region detection + IoU NMS + crop extraction."""

    def __init__(
        self,
        kernel_size: int = 64,
        stride: int = 32,
        n_regions: int = 5,
        iou_threshold: float = 0.3,
        min_crop_size: int = 32,
        output_size: int = None,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_regions = n_regions
        self.iou_threshold = iou_threshold
        self.min_crop_size = min_crop_size
        self.output_size = output_size

    def detect(self, inv_attention: np.ndarray, image_size: Tuple[int, int]) -> List[BBox]:
        img_w, img_h = image_size
        mask_h, mask_w = inv_attention.shape
        attn = inv_attention.astype(np.float32)

        candidates: List[BBox] = []
        for y in range(0, mask_h - self.kernel_size + 1, self.stride):
            for x in range(0, mask_w - self.kernel_size + 1, self.stride):
                score = float(attn[y: y + self.kernel_size, x: x + self.kernel_size].mean())
                box = BBox(
                    x1=int(x * img_w / mask_w),
                    y1=int(y * img_h / mask_h),
                    x2=int((x + self.kernel_size) * img_w / mask_w),
                    y2=int((y + self.kernel_size) * img_h / mask_h),
                    score=score,
                )
                if box.area >= self.min_crop_size ** 2:
                    candidates.append(box)

        if not candidates:
            return []

        candidates.sort(key=lambda b: -b.score)
        return self._nms(candidates)[: self.n_regions]

    def extract_crops(self, image: Image.Image, boxes: List[BBox]) -> List[Image.Image]:
        crops = []
        w, h = image.size
        size = self.output_size or 224

        for box in boxes:
            x1, y1 = max(0, box.x1), max(0, box.y1)
            x2, y2 = min(w, box.x2), min(h, box.y2)
            if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                continue
            crops.append(image.crop((x1, y1, x2, y2)).resize((size, size), Image.BICUBIC))

        return crops

    def _nms(self, boxes: List[BBox]) -> List[BBox]:
        kept: List[BBox] = []
        for box in boxes:
            if all(self._iou(box, k) <= self.iou_threshold for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (a.area + b.area - inter)
