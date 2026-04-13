# Results

Zero-shot retrieval performance of baseline models and LARE pipeline on COCO and
Flickr30K, along with their Dense variants. **Text → Image Recall@K (%)**.

| Model | ViT | COCO R@1 | R@5 | R@10 | Flickr30K R@1 | R@5 | R@10 | COCO-Dense R@1 | R@5 | R@10 | Flickr30K-Dense R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP    | L/14  | 36.10 | 61.10 | 71.44 | 65.00 | 88.00 | 92.62 | 18.25 | 36.11 | 45.66 |  7.31 | 16.21 | 21.87 |
| SigLIP  | So/14 | 54.24 | 76.78 | 84.21 | 82.94 | 96.08 | 98.00 | 26.61 | 46.31 | 55.22 |  9.93 | 21.15 | 27.05 |
| SigLIP 2 | So/16 | 56.55 | 78.75 | 85.95 | 83.72 | 96.34 | 98.32 | 27.56 | 47.56 | 56.73 | 11.25 | 22.20 | 28.58 |
| LARE (CLIP)    | L/14  | 36.10 | 61.10 | 71.44 | 65.00 | 88.00 | 92.62 | 23.66 | 43.18 | 53.21 |  8.51 | 19.08 | 24.57 |
| LARE (SigLIP)  | So/14 | 54.26 | 76.80 | 84.24 | 82.94 | 96.12 | 98.00 | 29.19 | 49.28 | 58.00 | 10.94 | 21.84 | 28.00 |
| **LARE (SigLIP 2)** | **So/16** | **56.56** | **78.78** | **85.97** | **83.76** | **96.38** | **98.34** | **30.33** | **50.56** | **59.62** | **11.49** | **22.91** | **29.48** |

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
