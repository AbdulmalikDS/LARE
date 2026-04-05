# Results

**Text → Image Recall@K (%)**, SigLIP So/14 (`siglip-so400m`), full test split.

## COCO (5000 test images, 25000 captions)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 54.24 | 76.78 | 84.21 |
| LARE     | 54.26 | 76.80 | 84.24 |

## Flickr30K (1000 test images)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 82.94 | 96.08 | 98.00 |
| LARE     | 82.94 | 96.12 | 98.00 |

## COCO-Dense (3089 images)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline | 26.61 | 46.31 | 55.22 |
| LARE     | 29.94 | 50.17 | 59.26 |

## Flickr30K-Dense (2477 images)

| Method   | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| Baseline |  5.05 | 15.50 | 20.96 |
| LARE     | 12.33 | 19.87 | 24.10 |

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
