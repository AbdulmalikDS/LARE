# Results

Zero-shot retrieval performance of baseline models and LARE pipeline on COCO and
Flickr30K, along with their Dense variants. **Text → Image Recall@K (%)**.

| Model | ViT | COCO R@1 | R@5 | R@10 | Flickr30K R@1 | R@5 | R@10 | COCO-Dense R@1 | R@5 | R@10 | Flickr30K-Dense R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP    | L/14  | 36.10 | 61.10 | 71.44 | 65.00 | 88.00 | 92.62 | 17.79 | 35.85 | 45.11 |  3.48 | 11.97 | 16.33 |
| SigLIP  | So/14 | 54.24 | 76.78 | 84.21 | 82.94 | 96.08 | 98.00 | 26.61 | 46.31 | 55.22 |  5.05 | 15.50 | 20.96 |
| SigLIP 2 | So/16 | 56.55 | 78.75 | 85.95 | 83.72 | 96.34 | 98.32 | 27.56 | 47.56 | 56.73 |  5.12 | 16.47 | 21.80 |
| **LARE (CLIP)**    | L/14  | 36.10 | 61.10 | 71.44 | 65.00 | 88.00 | 92.62 | 22.97 | 42.10 | 52.03 |  9.73 | 16.63 | 20.40 |
| **LARE (SigLIP)**  | So/14 | 54.26 | 76.80 | 84.24 | 82.94 | 96.12 | 98.00 | 29.94 | 50.17 | 59.26 | 12.33 | 19.87 | 24.10 |
| **LARE (SigLIP 2)** | So/16 | 56.56 | 78.78 | 85.97 | 83.76 | 96.38 | 98.34 | 31.00 | 51.45 | 60.67 | 13.28 | 21.11 | 25.10 |

## Reproduce

```bash
conda env create -f environment.yml
conda activate lare

COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```
