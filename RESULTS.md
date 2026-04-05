# Reproduction Results

All numbers are **Text → Image Recall@K (%)** on the full test split.
Reproduced with the current `main` branch on a single Quadro RTX 8000.

## COCO Karpathy (5000 images, 25000 captions)

| Model            | Method     | R@1   | R@5   | R@10  |
|------------------|------------|-------|-------|-------|
| SigLIP So/14     | Baseline   | 54.24 | 76.78 | 84.21 |
| SigLIP So/14     | LARE (N=5, τ=0.25) | 54.23 | 76.78 | 84.22 |

_Paper Table 3 reports: baseline 54.24 / 76.78 / 84.21, LARE 54.26 / 76.80 / 84.24._
_Matches within noise (~0.03%)._ ✅

## Flickr30K (1000 test images)

| Model            | Method     | R@1   | R@5   | R@10  |
|------------------|------------|-------|-------|-------|
| SigLIP So/14     | Baseline   | 82.94 | 96.08 | 98.00 |
| SigLIP So/14     | LARE (N=5, τ=0.25) | 82.96 | 96.08 | 98.04 |

_Paper Table 3 reports: baseline 82.94 / 96.08 / 98.00, LARE 82.94 / 96.12 / 98.00._
_Matches within noise._ ✅

## COCO-Dense (3089 images)

Paper target (SigLIP So/14): baseline 26.61 / 46.31 / 55.22, LARE 29.94 / 50.17 / 59.26.

Tested caption files (all on val2014 images):

| Captions file | N_img | N_cap | R@1 | R@5 | R@10 |
|---------------|------:|------:|----:|----:|-----:|
| `MSCOCO_val2014_denseset_blip.json` | 3089 | — | 13.64 | 27.76 | 35.75 |
| `MSCOCO_val2014_denseset_gpt_oss_refined_mscoco_captions.json` | 3089 | 15265 | 41.76 | 65.97 | 75.49 |
| `val2014_filtered_denseset_blip_captions.json` | 2823 | — | 11.91 | 25.76 | 33.97 |

**None match** the paper's 26.61. The paper's number sits between blip (13.64) and
refined-MSCOCO (41.76) — suggests a third caption file or a different split we don't
have locally. ❌ _Status: the exact Dense-Set setup used in the paper is unresolved._

## Flickr30K-Dense (2477 images)

Paper target: baseline 5.05 / 15.50 / 20.96, LARE 12.33 / 19.87 / 24.10.

Using `flickr30k_denseset_blip_gpt_oss.json`: baseline **9.93 / 21.15 / 27.05**.
Doesn't match paper's 5.05 baseline. ❌ _Same issue — wrong caption file._

## How to reproduce

```bash
conda env create -f environment.yml
conda activate lare

# 1) COCO Karpathy (requires val2014 images)
python scripts/eval.py --model siglip-so400m --dataset karpathy \
    --images-dir /path/to/coco/val2014
python scripts/eval.py --model siglip-so400m --dataset karpathy \
    --images-dir /path/to/coco/val2014 --pipeline

# 2) Flickr30K (requires images + Karpathy-style dataset.json)
python scripts/eval.py --model siglip-so400m --dataset flickr30k \
    --images-dir /path/to/flickr30k/images --flickr-json /path/to/dataset.json
python scripts/eval.py --model siglip-so400m --dataset flickr30k \
    --images-dir /path/to/flickr30k/images --flickr-json /path/to/dataset.json \
    --pipeline

# 3) Dense-Set (download from HuggingFace: AbdulmalekDS/Dense-Set)
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json /path/to/dense_set_captions.json
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json /path/to/dense_set_captions.json --pipeline
```

Or run all three benchmarks with `scripts/reproduce.sh`.
