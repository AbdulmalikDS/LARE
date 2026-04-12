#!/usr/bin/env bash
set -e

# Run all paper models sequentially
MODELS=("clip-large" "siglip-so400m" "siglip2")

# Set paths directly to your local data folder for convenience
export COCO_DIR="/home/aalquwayfili/Projects/Semantic Search/data/coco/val2014"
export FLICKR_DIR="/home/aalquwayfili/Projects/Semantic Search/data/Flicker30K/flickr30k/flickr30k-images"

echo "Evaluating all models: ${MODELS[*]}"
echo "Writing results to full_reproduction_log.txt..."

> full_reproduction_log.txt

for MODEL in "${MODELS[@]}"; do
    echo "===================================================" | tee -a full_reproduction_log.txt
    echo "  EVALUATING MODEL: $MODEL                           " | tee -a full_reproduction_log.txt
    echo "===================================================" | tee -a full_reproduction_log.txt
    
    # Run the exact reproduce script from the repo
    bash scripts/reproduce.sh "$MODEL" | tee -a full_reproduction_log.txt
done

echo ""
echo "Done! The output has been safely written to full_reproduction_log.txt."
