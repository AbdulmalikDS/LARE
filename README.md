# LARE: Low-Attention Region Encoding for Text–Image Retrieval

**CVPRW 2026** | [Project Page](https://falmeshal.github.io/LARE/) | [Dense-Set Dataset](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

LARE is a training-free method that improves text-image retrieval by detecting overlooked image regions and re-encoding them with the same frozen encoder. A confidence gate decides when global similarity is unreliable and blends in regional evidence.

## Setup

```bash
conda env create -f environment.yml
conda activate lare
```

## Usage

```python
from lare import create_lare

pipeline = create_lare(model="siglip")  # or clip, siglip-so400m, siglip2
score = pipeline.retrieve(image, "a person carrying a red bag")
print(score.score)
```

## Evaluation

```bash
# Baseline
python scripts/eval.py --model siglip --dataset karpathy

# LARE
python scripts/eval.py --model siglip --dataset karpathy --pipeline
```

Supported datasets: `karpathy`, `flickr30k`, `dense_set`

## Citation

```bibtex
@inproceedings{alquwayfili2026lare,
  title={LARE: Low-Attention Region Encoding for Text--Image Retrieval},
  author={Abdulmalik Alquwayfili and Faisal Almeshal and Jumanah Almajnouni and Leena Alotaibi and Huda Alamri and Muhammad Kamran J Khan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```
