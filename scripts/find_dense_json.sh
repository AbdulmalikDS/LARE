#!/usr/bin/env bash
# Test each candidate Dense-Set caption file on COCO val2014
# to find the one matching paper baseline R@1 = 26.61%.
set -u

GPU="${GPU:-cuda:0}"
IMG_DIR="/home/aalquwayfili/Projects/Semantic Search/data/coco/val2014"
OUT=figures/dense_search.log
mkdir -p figures

CANDIDATES=(
  "/home/aalquwayfili/Projects/Semantic Search/data/dense-datasets/val2014/captions_coco_format.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/dense-datasets/val2014/refined_val2014_denseset_captions_gpt_oss.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/dense-datasets/val2014/refined_val2014_denseset_captions_gpt_oss2.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/dense-datasets/val2014/refined_val2014_denseset_captions_gpt_oss_blip.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/dense-datasets/val2014/val2014_mscoco_refined_captions.json"
)

echo "Search started $(date)" > "$OUT"
for j in "${CANDIDATES[@]}"; do
  name="$(basename "$j" .json)"
  echo "=== $name ===" | tee -a "$OUT"
  python scripts/eval.py --model siglip-so400m --device "$GPU" \
    --dataset dense_set --images-dir "$IMG_DIR" --captions-json "$j" 2>&1 \
    | grep -E "R@|Loaded" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "Done $(date)" >> "$OUT"
