<div align="center">

# 🌌 LARE: Low-Attention Region Encoding for Text–Image Retrieval

**[CVPR 2026 — MULA Workshop](https://mula-workshop.github.io/)** | [Project Page](https://falmeshal.github.io/LARE/) | [Dense-Set Dataset](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

**LARE** is a training-free augmentation for text-to-image retrieval in crowded scenes. Our method mines low-attention regions from a frozen vision encoder, encodes these regions alongside the full image, and combines regional embeddings with the global image embedding at inference time. This simple test-time procedure improves retrieval on Dense-Set variants that emphasize subtle and occluded content.

<br>
<img src="docs/image1.png" width="90%" alt="LARE Details">

</div>

## Quick Start

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

## Reproduce Results

> **Hardware**: All benchmarks were evaluated on two NVIDIA Quadro RTX 8000 GPUs. However, a standard 11GB-16GB GPU is sufficient to reproduce the results.

**Step 1. Clone & Environment**
```bash
git clone https://github.com/AbdulmalikDS/LARE.git
cd LARE
conda env create -f environment.yml
conda activate lare
```

**Step 2. Download Datasets**
Create a data directory and download the MS-COCO 2014 validation split (you can also provide Flickr30k):
```bash
mkdir -p datasets/coco
wget http://images.cocodataset.org/zips/val2014.zip -O datasets/coco/val2014.zip
unzip datasets/coco/val2014.zip -d datasets/coco/
```

**Step 3. Run Benchmark Scripts**
Pass the directory to the reproduction script. It will automatically test the baseline and LARE pipeline on standard and Dense-Set annotations.

```bash
COCO_DIR=datasets/coco/val2014 \
# FLICKR_DIR=datasets/flickr30k-images \
bash scripts/reproduce.sh siglip-so400m
```

## 📜 Citation

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
