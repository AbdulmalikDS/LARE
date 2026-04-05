# LARE: Low-Attention Region Encoding for Text–Image Retrieval

**[CVPR 2026 — MULA Workshop](https://mula-workshop.github.io/)** | [Project Page](https://falmeshal.github.io/LARE/) | [Dense-Set Dataset](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

LARE is a training-free method that improves text-image retrieval by detecting overlooked image regions and re-encoding them with the same frozen encoder. A confidence gate decides when global similarity is unreliable and blends in regional evidence.

## Quick start

```bash
conda env create -f environment.yml
conda activate lare
```

```python
from lare import create_lare

pipeline = create_lare(model="siglip-so400m")
score = pipeline.retrieve(image, "a person carrying a red bag")
print(score.score)
```

## Reproducing paper numbers

Target numbers from Table 3 (SigLIP So/14) are listed in [`RESULTS.md`](RESULTS.md).
Run all benchmarks with one command:

```bash
COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images FLICKR_JSON=/path/to/dataset.json \
DENSESET_JSON=/path/to/dense_set_captions.json \
bash scripts/reproduce.sh siglip-so400m
```

Dense-Set caption files are committed in [`benchmarks/captions/`](benchmarks/) — see
[`benchmarks/README.md`](benchmarks/README.md) for exact expected numbers per file.

Or evaluate a single benchmark:

```bash
# COCO Karpathy (5000 images)
python scripts/eval.py --model siglip-so400m --dataset karpathy \
    --images-dir /path/to/coco/val2014 [--pipeline]

# Flickr30K (1000 test images)
python scripts/eval.py --model siglip-so400m --dataset flickr30k \
    --images-dir /path/to/flickr30k-images --flickr-json /path/to/dataset.json [--pipeline]

# COCO-Dense (3089 images)
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json /path/to/dense_set_captions.json [--pipeline]
```

Add `--pipeline` to enable LARE; omit for baseline global-only scoring.
Supported models: `clip`, `clip-large`, `clip-336`, `siglip`, `siglip-so400m`, `siglip2`.

## Repository layout

```
lare/           Paper implementation (frozen — used for reproduction)
lare_plus/      Experimental research branch (DINOv3 backbone, PC1 saliency — unstable)
eval/           Dataset loaders and R@K metrics
scripts/        Evaluation + visualization scripts
```

Only `lare/` is needed to reproduce Table 3 numbers. `lare_plus/` is an in-progress
research direction and is **not** used by `scripts/eval.py` or `scripts/reproduce.sh`.

## Citation

```bibtex
@inproceedings{alquwayfili2026lare,
  title={LARE: Low-Attention Region Encoding for Text--Image Retrieval},
  author={Abdulmalik Alquwayfili and Faisal Almeshal and Jumanah Almajnouni
          and Leena Alotaibi and Huda Alamri and Muhammad Kamran J Khan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
             Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```
