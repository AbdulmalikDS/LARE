# Results

**Text → Image Recall@K (%)**, SigLIP So/14 (`siglip-so400m`), full test split,
single Quadro RTX 8000.

## COCO (5000 test images, 25000 captions)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline   | 54.24 | 76.78 | 84.21 |
| LARE (N=5, τ=0.25) | 54.23 | 76.78 | 84.22 |

## Flickr30K (1000 test images)

| Method     | R@1   | R@5   | R@10  |
|------------|-------|-------|-------|
| Baseline   | 82.94 | 96.08 | 98.00 |
| LARE       | 82.96 | 96.08 | 98.04 |

## COCO-Dense (3089 images)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 13.64 | 27.76 | 35.75 |

## Flickr30K-Dense (2477 images)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 9.93  | 21.15 | 27.05 |

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
