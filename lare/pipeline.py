from PIL import Image
from typing import List, Tuple, Union
import numpy as np

from .backbones import create_encoder
from .backbones.base import BaseEncoder
from .detector import RegionDetector, BBox
from .scoring import ConfidenceGatedScoring, RetrievalScore


class LARE:
    """
    Low-Attention Region Encoding pipeline.

    Augments a frozen dual-encoder at inference time by:
      1. Detecting low-attention regions in the encoder's attention map
      2. Re-encoding each region with the same visual encoder
      3. Scoring via confidence-gated fusion (paper Eq. 4)

    Usage::

        from lare import create_lare
        pipeline = create_lare(model="siglip")

        score = pipeline.retrieve(image, "a person carrying a red bag")

        # Batch evaluation
        global_embs, region_embs, _ = pipeline.encode_batch(images)
        text_embs = pipeline.encode_text(queries)
        sim_matrix = pipeline.scorer.score_matrix(text_embs, global_embs, region_embs)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        detector: RegionDetector,
        scorer: ConfidenceGatedScoring,
    ):
        self.encoder = encoder
        self.detector = detector
        self.scorer = scorer
        self.detector.output_size = encoder.config.image_size

    def encode(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, List[np.ndarray], List[BBox]]:
        """Returns (global_emb [D], region_embs, boxes)."""
        enc = self.encoder.encode_image(image)
        boxes = self.detector.detect(enc.attention, image.size)
        crops = self.detector.extract_crops(image, boxes)
        region_embs: List[np.ndarray] = []
        if crops:
            region_embs = [o.embedding for o in self.encoder.encode_images_batch(crops)]
        return enc.embedding, region_embs, boxes

    def retrieve(self, image: Image.Image, query: str) -> RetrievalScore:
        """Confidence-gated similarity between an image and a text query."""
        global_emb, region_embs, _ = self.encode(image)
        text_emb = self.encoder.encode_text(query)
        return self.scorer.score(text_emb, global_emb, region_embs)

    def encode_batch(
        self, images: List[Image.Image]
    ) -> Tuple[np.ndarray, List[List[np.ndarray]], List[List[BBox]]]:
        """
        Encode a batch of images. All crops are batched into a single encoder
        call, making this much faster than calling encode() in a loop.

        Returns: global_embs [N, D], region_embs [N][n_regions], boxes [N].
        """
        enc_outputs = self.encoder.encode_images_batch(images)

        all_crops: List[Image.Image] = []
        crop_to_img: List[int] = []
        all_boxes: List[List[BBox]] = []

        for i, (img, enc) in enumerate(zip(images, enc_outputs)):
            boxes = self.detector.detect(enc.attention, img.size)
            crops = self.detector.extract_crops(img, boxes)
            all_boxes.append(boxes)
            for crop in crops:
                all_crops.append(crop)
                crop_to_img.append(i)

        region_embs: List[List[np.ndarray]] = [[] for _ in images]
        if all_crops:
            crop_outputs = self.encoder.encode_images_batch(all_crops)
            for img_i, out in zip(crop_to_img, crop_outputs):
                region_embs[img_i].append(out.embedding)

        global_embs = np.stack([e.embedding for e in enc_outputs])
        return global_embs, region_embs, all_boxes

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode one or more text queries. Returns [D] or [N, D]."""
        return self.encoder.encode_text(text)


def create_lare(
    model: str = "siglip",
    n_regions: int = 5,
    kernel_size: int = 64,
    stride: int = 32,
    iou_threshold: float = 0.3,
    tau: float = 0.25,
    csls_k: int = 10,
    device: str = None,
) -> LARE:
    """
    Create a LARE pipeline.

    Args:
        model: One of: clip, clip-large, clip-336, siglip, siglip-so400m, siglip2.
        n_regions: Number of low-attention regions to extract (paper: N=5).
        kernel_size: Sliding window size over the attention map.
        stride: Sliding window stride.
        iou_threshold: IoU threshold for NMS.
        tau: Confidence threshold for gated scoring (paper: tau=0.25).
        csls_k: CSLS hub-correction neighbourhood size (0 to disable).
        device: "cuda" or "cpu". Auto-detected if None.
    """
    encoder = create_encoder(model, device=device)
    detector = RegionDetector(
        kernel_size=kernel_size,
        stride=stride,
        n_regions=n_regions,
        iou_threshold=iou_threshold,
        output_size=encoder.config.image_size,
    )
    scorer = ConfidenceGatedScoring(tau=tau, csls_k=csls_k)
    return LARE(encoder, detector, scorer)
