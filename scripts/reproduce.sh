#!/usr/bin/env bash
# Usage:
#   COCO_DIR=/path/to/val2014 FLICKR_DIR=/path/to/flickr30k-images \
#   bash scripts/reproduce.sh [model]

set -euo pipefail

MODEL=${1:-siglip-so400m}
COCO_DIR=${COCO_DIR:-}
FLICKR_DIR=${FLICKR_DIR:-}

FLICKR_JSON=benchmarks/captions/flickr30k_karpathy.json
COCO_DENSE_JSON=benchmarks/captions/coco_dense.json
FLICKR_DENSE_JSON=benchmarks/captions/flickr30k_dense.json

run() {
    echo
    echo "==================================================================="
    echo "  $*"
    echo "==================================================================="
    python scripts/eval.py "$@"
}

if [[ -n "$COCO_DIR" ]]; then
    run --model "$MODEL" --dataset karpathy --images-dir "$COCO_DIR"
    run --model "$MODEL" --dataset karpathy --images-dir "$COCO_DIR" --pipeline
    run --model "$MODEL" --dataset dense_set --images-dir "$COCO_DIR" --captions-json "$COCO_DENSE_JSON"
    run --model "$MODEL" --dataset dense_set --images-dir "$COCO_DIR" --captions-json "$COCO_DENSE_JSON" --pipeline
else
    echo "skip: COCO (set COCO_DIR to enable)"
fi

if [[ -n "$FLICKR_DIR" ]]; then
    run --model "$MODEL" --dataset flickr30k --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON"
    run --model "$MODEL" --dataset flickr30k --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON" --pipeline
    run --model "$MODEL" --dataset dense_set --images-dir "$FLICKR_DIR" --captions-json "$FLICKR_DENSE_JSON"
    run --model "$MODEL" --dataset dense_set --images-dir "$FLICKR_DIR" --captions-json "$FLICKR_DENSE_JSON" --pipeline
else
    echo "skip: Flickr30K (set FLICKR_DIR to enable)"
fi
