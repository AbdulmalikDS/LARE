#!/usr/bin/env bash
# Reproduce Table 3 numbers from the LARE paper.
#
# Usage:
#   COCO_DIR=/path/to/val2014 \
#   FLICKR_DIR=/path/to/flickr30k/images FLICKR_JSON=/path/to/dataset.json \
#   DENSESET_JSON=/path/to/dense_set_captions.json \
#   bash scripts/reproduce.sh [model]
#
# model defaults to siglip-so400m (the strongest variant reported in the paper).

set -euo pipefail

MODEL=${1:-siglip-so400m}
COCO_DIR=${COCO_DIR:-}
FLICKR_DIR=${FLICKR_DIR:-}
FLICKR_JSON=${FLICKR_JSON:-}
DENSESET_JSON=${DENSESET_JSON:-}

die() { echo "error: $*" >&2; exit 1; }

run() {
    echo
    echo "==================================================================="
    echo "  $*"
    echo "==================================================================="
    python scripts/eval.py "$@"
}

# COCO Karpathy
if [[ -n "$COCO_DIR" ]]; then
    run --model "$MODEL" --dataset karpathy --images-dir "$COCO_DIR"
    run --model "$MODEL" --dataset karpathy --images-dir "$COCO_DIR" --pipeline
else
    echo "skip: COCO (set COCO_DIR to enable)"
fi

# Flickr30K
if [[ -n "$FLICKR_DIR" && -n "$FLICKR_JSON" ]]; then
    run --model "$MODEL" --dataset flickr30k --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON"
    run --model "$MODEL" --dataset flickr30k --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON" --pipeline
else
    echo "skip: Flickr30K (set FLICKR_DIR and FLICKR_JSON to enable)"
fi

# Dense-Set (shares val2014 images with COCO)
if [[ -n "$COCO_DIR" && -n "$DENSESET_JSON" ]]; then
    run --model "$MODEL" --dataset dense_set --images-dir "$COCO_DIR" --captions-json "$DENSESET_JSON"
    run --model "$MODEL" --dataset dense_set --images-dir "$COCO_DIR" --captions-json "$DENSESET_JSON" --pipeline
else
    echo "skip: Dense-Set (set COCO_DIR and DENSESET_JSON to enable)"
fi
