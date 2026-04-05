# Reproduction Results

All numbers are **Text → Image Recall@K (%)** on the full test split, SigLIP So/14
(`siglip-so400m`), single Quadro RTX 8000.

## COCO (5000 test images, 25000 captions)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline   | 54.24 | 76.78 | 84.21 |
| LARE (N=5, τ=0.25) | 54.23 | 76.78 | 84.22 |

Paper Table 3: baseline 54.24 / 76.78 / 84.21, LARE 54.26 / 76.80 / 84.24.

## Flickr30K (1000 test images)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline   | 82.94 | 96.08 | 98.00 |
| LARE       | 82.96 | 96.08 | 98.04 |

Paper Table 3: baseline 82.94 / 96.08 / 98.00, LARE 82.94 / 96.12 / 98.00.

## COCO-Dense (3089 images)

Using `data/captions/coco_dense.json`:

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 13.64 | 27.76 | 35.75 |

Paper Table 3: baseline 26.61 / 46.31 / 55.22, LARE 29.94 / 50.17 / 59.26.
The committed caption file does not match the paper's baseline — the exact
caption file used for the paper is still being located.

## Flickr30K-Dense (2477 images)

Using `data/captions/flickr30k_dense.json`:

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 9.93  | 21.15 | 27.05 |

Paper Table 3: baseline 5.05 / 15.50 / 20.96, LARE 12.33 / 19.87 / 24.10.

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
