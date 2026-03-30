# LARE: Low-Attention Region Encoding for Text–Image Retrieval

**[CVPR 2026](https://cvpr.thecvf.com/) — [MULA Workshop](https://mula-workshop.github.io/)** | [Project Page](https://falmeshal.github.io/LARE/) | [Dense-Set Dataset](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

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
# MS-COCO Karpathy
python scripts/eval.py --model siglip --dataset karpathy \
    --images-dir /path/to/coco/val2014

# Flickr30k
python scripts/eval.py --model siglip --dataset flickr30k \
    --images-dir /path/to/flickr30k/images \
    --flickr-json /path/to/dataset.json

# Dense-Set
python scripts/eval.py --model siglip --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json /path/to/captions.json

# Add --pipeline to enable LARE (default is baseline)
```

Supported models: `clip`, `clip-large`, `clip-336`, `siglip`, `siglip-so400m`, `siglip2`

## Citation

```bibtex
@inproceedings{alquwayfili2026lare,
  title={LARE: Low-Attention Region Encoding for Text--Image Retrieval},
  author={Abdulmalik Alquwayfili and Faisal Almeshal and Jumanah Almajnouni and Leena Alotaibi and Huda Alamri and Muhammad Kamran J Khan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```
