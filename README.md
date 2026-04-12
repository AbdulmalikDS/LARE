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

## Reproduce

Hardware: Evaluation scripts run comfortably out-of-the-box on a single **NVIDIA 24GB GPU** (e.g., RTX 3090, 4090, or A10g).

First, see [`data/README.md`](data/README.md) for quick instructions on where to download the COCO and Flickr30K image sets. 

Expected benchmark numbers are clearly tabulated in [`RESULTS.md`](RESULTS.md).

Run all benchmarks:

```bash
COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh
```

Or a single manual run:

```bash
python scripts/eval.py --model siglip-so400m --dataset karpathy \
    --images-dir /path/to/coco/val2014 [--pipeline]

python scripts/eval.py --model siglip-so400m --dataset flickr30k \
    --images-dir /path/to/flickr30k-images \
    --flickr-json data/captions/flickr30k.json [--pipeline]

python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json data/captions/coco_dense.json [--pipeline]
```

Add `--pipeline` to enable LARE; omit for baseline.
Supported models: `clip`, `clip-large`, `clip-336`, `siglip`, `siglip-so400m`, `siglip2`.

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
