#!/usr/bin/env bash
# Overnight benchmark queue — runs all paper-reproduction experiments.
# Safe to run unattended: each experiment is isolated, failures don't abort the rest.
#
# Uses GPU1 by default (set GPU env var to override).

set -u

GPU="${GPU:-cuda:1}"
MODEL="${MODEL:-siglip-so400m}"
OUT_DIR="${OUT_DIR:-figures/overnight_logs}"
mkdir -p "$OUT_DIR"

COCO_DIR="/home/aalquwayfili/Projects/Semantic Search/data/coco/val2014"
FLICKR_DIR="/home/aalquwayfili/Projects/Semantic Search/data/Flicker30K/flickr30k/flickr30k-images"
FLICKR_JSON="/home/aalquwayfili/Projects/Semantic Search/data/Flicker30K/flickr30k/dataset.json"

# Dense-Set candidates (paper reports COCO-Dense baseline R@1 = 26.61%, 3089 imgs)
DENSE_JSONS=(
  "/home/aalquwayfili/Projects/Semantic Search/data/MSCOCO_val2014_denseset_blip.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/MSCOCO_val2014_denseset_gpt_oss_refined_mscoco_captions.json"
  "/home/aalquwayfili/Projects/Semantic Search/data/val2014_filtered_denseset_blip_captions.json"
)

FLICKR_DENSE_JSON="/home/aalquwayfili/Projects/Semantic Search/data/Flicker30K/flickr30k_denseset_blip_gpt_oss.json"

SUMMARY="$OUT_DIR/SUMMARY.txt"
echo "Overnight benchmark queue started $(date)" > "$SUMMARY"
echo "Model: $MODEL, GPU: $GPU" >> "$SUMMARY"
echo "" >> "$SUMMARY"

run_eval() {
  local tag="$1"; shift
  local log="$OUT_DIR/$tag.log"
  echo "--- [$tag] start $(date +%H:%M:%S) ---" | tee -a "$SUMMARY"
  if python scripts/eval.py --model "$MODEL" --device "$GPU" "$@" > "$log" 2>&1; then
    tail -6 "$log" | grep -E "R@|Method|====" | tee -a "$SUMMARY"
    echo "--- [$tag] done  $(date +%H:%M:%S) ---" | tee -a "$SUMMARY"
  else
    echo "  [$tag] FAILED — see $log" | tee -a "$SUMMARY"
  fi
  echo "" | tee -a "$SUMMARY"
}

# 1) COCO Karpathy baseline + LARE
run_eval "coco_baseline"      --dataset karpathy --images-dir "$COCO_DIR"
run_eval "coco_lare"          --dataset karpathy --images-dir "$COCO_DIR" --pipeline

# 2) COCO-Dense: search for the json that matches paper's R@1 = 26.61%
for j in "${DENSE_JSONS[@]}"; do
  name="$(basename "$j" .json)"
  run_eval "densecoco_${name}_baseline" --dataset dense_set \
    --images-dir "$COCO_DIR" --captions-json "$j"
done

# 3) Flickr30K baseline + LARE (if data exists)
if [[ -d "$FLICKR_DIR" ]]; then
  run_eval "flickr_baseline" --dataset flickr30k \
    --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON"
  run_eval "flickr_lare"     --dataset flickr30k \
    --images-dir "$FLICKR_DIR" --flickr-json "$FLICKR_JSON" --pipeline
fi

# 4) Flickr-Dense baseline (Flickr dense uses a different json as captions source)
if [[ -f "$FLICKR_DENSE_JSON" ]]; then
  run_eval "flickr_dense_baseline" --dataset dense_set \
    --images-dir "$FLICKR_DIR" --captions-json "$FLICKR_DENSE_JSON"
fi

echo "" >> "$SUMMARY"
echo "Overnight queue finished $(date)" >> "$SUMMARY"
echo "Done. Summary: $SUMMARY"
