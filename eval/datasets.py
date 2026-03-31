"""
Dataset loaders for LARE evaluation.

Each loader returns:
    images          : List[PIL.Image]
    texts           : List[str]
    text_to_image   : np.ndarray[int]  —  text_to_image[i] = image index for query i
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

ReturnType = Tuple[List[Image.Image], List[str], np.ndarray]


def load_karpathy(images_dir: str, limit: int = None) -> ReturnType:
    """MS-COCO Karpathy test split (captions from HuggingFace yerevann/coco-karpathy)."""
    from datasets import load_dataset

    logger.info("Loading MS-COCO Karpathy test split...")
    dataset = load_dataset("yerevann/coco-karpathy", split="test")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    images, texts = [], []
    text_to_image_map: dict = {}
    img_idx = txt_idx = 0

    for item in tqdm(dataset, desc="Karpathy"):
        filename = item.get("filename") or f"COCO_val2014_{item['cocoid']:012d}.jpg"
        path = _resolve_path(images_dir, filename)
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {path}: {e}")
            continue
        images.append(img)
        for cap in item["sentences"]:
            texts.append(cap)
            text_to_image_map[txt_idx] = img_idx
            txt_idx += 1
        img_idx += 1

    return images, texts, _build_index(text_to_image_map, txt_idx)


def load_flickr30k(
    images_dir: str,
    dataset_json: str,
    split: str = "test",
    limit: int = None,
) -> ReturnType:
    """Flickr30k split from a Karpathy-style dataset.json."""
    logger.info(f"Loading Flickr30k '{split}' split...")
    with open(dataset_json) as f:
        raw = json.load(f)

    entries = raw["images"] if isinstance(raw, dict) and "images" in raw else raw
    entries = [e for e in entries if e.get("split") == split]
    if limit:
        entries = entries[:limit]

    images, texts = [], []
    text_to_image_map: dict = {}
    img_idx = txt_idx = 0

    for item in tqdm(entries, desc=f"Flickr30k ({split})"):
        filename = item.get("filename")
        if not filename:
            continue
        path = _resolve_path(images_dir, filename)
        if path is None:
            logger.warning(f"Image not found: {filename}")
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {path}: {e}")
            continue
        images.append(img)
        for s in item.get("sentences", []):
            texts.append(s.get("raw") or s.get("sentence") or s.get("text") or str(s))
            text_to_image_map[txt_idx] = img_idx
            txt_idx += 1
        img_idx += 1

    return images, texts, _build_index(text_to_image_map, txt_idx)


def load_dense_set(
    images_dir: str,
    captions_json: str,
    limit: int = None,
) -> ReturnType:
    """Dense-Set benchmark in COCO annotation format."""
    logger.info(f"Loading Dense-Set from {captions_json}...")
    with open(captions_json) as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    img_captions: dict = defaultdict(list)
    for ann in data["annotations"]:
        img_captions[ann["image_id"]].append(ann["caption"])

    images, texts = [], []
    text_to_image_map: dict = {}
    img_idx = txt_idx = 0

    for img_id in tqdm(img_captions, desc="Dense-Set"):
        if limit and img_idx >= limit:
            break
        filename = id_to_filename.get(img_id)
        if not filename:
            continue
        path = _resolve_path(images_dir, filename)
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {path}: {e}")
            continue
        images.append(img)
        for cap in img_captions[img_id]:
            texts.append(cap)
            text_to_image_map[txt_idx] = img_idx
            txt_idx += 1
        img_idx += 1

    logger.info(f"Dense-Set: {img_idx} images, {txt_idx} captions")
    return images, texts, _build_index(text_to_image_map, txt_idx)


def _resolve_path(directory: str, filename: str):
    p = Path(directory) / filename
    if p.exists():
        return p
    p2 = Path(directory) / Path(filename).name
    return p2 if p2.exists() else None


def _build_index(mapping: dict, n_texts: int) -> np.ndarray:
    idx = np.zeros(n_texts, dtype=np.int64)
    for t, i in mapping.items():
        idx[t] = i
    return idx
