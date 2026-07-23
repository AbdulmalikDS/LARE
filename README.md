<div align="center">

# 🌌 LARE: Low-Attention Region Encoding for Text–Image Retrieval

**[Paper](https://arxiv.org/abs/2606.18885)** | [ICML 2026 EMMQA Workshop](https://qanta-org.github.io/competition/2026/icml/) | [Dense-Set Dataset](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

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
Head to [`data/README.md`](data/README.md) for quick instructions and `wget` scripts on how to download the MS-COCO and Flickr30K image sets locally.

**Step 3. Run Benchmark Scripts**
Pass the dataset directory to the reproduction script. It will evaluate the baseline and LARE pipeline natively against standard and Dense-Set annotations.

```bash
COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh siglip-so400m
```

## 📜 Citation

```bibtex
@inproceedings{alquwayfili2026lare,
  title={LARE: Low-Attention Region Encoding for Text--Image Retrieval},
  author={Abdulmalik Alquwayfili and Faisal Almeshal and Jumanah Almajnouni
          and Leena Alotaibi and Faisal Alhajari and Mohammed Alkhrashi
          and Alreem Almuhrij and Abdullah Aldwyish and Raied Aljadaany
          and Huda Alamri and Muhammad Kamran J. Khan},
  booktitle={ICML 2026 Workshop on Efficient Multimodal Question Answering (EMMQA)},
  year={2026},
  url={https://openreview.net/forum?id=42bo30qeLe}
}
```
