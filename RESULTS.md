# Results

**Text → Image Recall@K (%)**, SigLIP So/14 (`siglip-so400m`), full test split,
single Quadro RTX 8000.

## COCO (5000 test images, 25000 captions)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline (paper) | 54.24 | 76.78 | 84.21 |
| LARE (paper)     | 54.26 | 76.80 | 84.24 |
| Baseline (ours)  | 54.24 | 76.78 | 84.21 |
| LARE (ours)      | 54.23 | 76.78 | 84.22 |

## Flickr30K (1000 test images)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline (paper) | 82.94 | 96.08 | 98.00 |
| LARE (paper)     | 82.94 | 96.12 | 98.00 |
| Baseline (ours)  | 82.94 | 96.08 | 98.00 |
| LARE (ours)      | 82.96 | 96.08 | 98.04 |

## COCO-Dense (3089 images)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline (paper) | 26.61 | 46.31 | 55.22 |
| LARE (paper)     | 29.94 | 50.17 | 59.26 |
| Baseline (ours)  | 13.64 | 27.76 | 35.75 |

## Flickr30K-Dense (2477 images)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline (paper) |  5.05 | 15.50 | 20.96 |
| LARE (paper)     | 12.33 | 19.87 | 24.10 |
| Baseline (ours)  |  9.93 | 21.15 | 27.05 |

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
