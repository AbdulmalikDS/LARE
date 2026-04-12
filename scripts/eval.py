#!/usr/bin/env python
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="siglip",
                   choices=["clip", "clip-large", "clip-336", "siglip", "siglip-so400m", "siglip2"])
    p.add_argument("--dataset", default="karpathy",
                   choices=["karpathy", "flickr30k", "dense_set"])
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--captions-json", default=None)
    p.add_argument("--flickr-json", default=None)
    p.add_argument("--n-regions", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.25)
    p.add_argument("--csls-k", type=int, default=10,
                   help="CSLS neighbourhood size for hub correction (0 to disable)")
    p.add_argument("--device", default=None)
    p.add_argument("--img-batch-size", type=int, default=32)
    p.add_argument("--txt-batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from eval.datasets import load_karpathy, load_flickr30k, load_dense_set

    if args.dataset == "karpathy":
        if not args.images_dir:
            logger.error("--images-dir is required for karpathy (path to COCO val2014 images)")
            sys.exit(1)
        images, texts, text_to_image = load_karpathy(args.images_dir, limit=args.limit)
    elif args.dataset == "flickr30k":
        if not args.images_dir or not args.flickr_json:
            logger.error("--images-dir and --flickr-json are required for flickr30k")
            sys.exit(1)
        images, texts, text_to_image = load_flickr30k(
            args.images_dir, args.flickr_json, split="test", limit=args.limit,
        )
    elif args.dataset == "dense_set":
        if not args.images_dir or not args.captions_json:
            logger.error("--images-dir and --captions-json are required for dense_set")
            sys.exit(1)
        images, texts, text_to_image = load_dense_set(
            args.images_dir, args.captions_json, limit=args.limit,
        )

    if not images:
        logger.error("No images loaded.")
        sys.exit(1)

    logger.info(f"Loaded {len(images)} images, {len(texts)} queries")

    from lare import create_lare
    csls_k = args.csls_k if args.pipeline else 0
    pipeline = create_lare(model=args.model, n_regions=args.n_regions, tau=args.tau, csls_k=csls_k, device=args.device)

    # Encode images
    all_global, all_regions = [], []
    for i in tqdm(range(0, len(images), args.img_batch_size), desc="Images"):
        batch = images[i: i + args.img_batch_size]
        if args.pipeline:
            g, r, _ = pipeline.encode_batch(batch)
            all_global.append(g)
            all_regions.extend(r)
        else:
            outs = pipeline.encoder.encode_images_batch(batch)
            all_global.append(np.stack([o.embedding for o in outs]))
            all_regions.extend([[] for _ in batch])
    global_embs = np.concatenate(all_global, axis=0)

    # Encode texts
    all_text = []
    for i in tqdm(range(0, len(texts), args.txt_batch_size), desc="Texts"):
        all_text.append(pipeline.encode_text(texts[i: i + args.txt_batch_size]))
    text_embs = np.concatenate(all_text, axis=0)

    # Similarity matrix
    if args.pipeline:
        sim = pipeline.scorer.score_matrix(text_embs, global_embs, all_regions)
    else:
        sim = text_embs @ global_embs.T

    from eval.metrics import recall_at_k
    results = recall_at_k(sim, text_to_image)

    mode = "LARE" if args.pipeline else "Baseline"
    print(f"\n{'='*52}")
    print(f"  {args.model.upper()} | {args.dataset} | {mode}")
    print(f"{'='*52}")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
