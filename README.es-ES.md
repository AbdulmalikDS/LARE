

<div align="center">

# 🌌 LARE: Codificación de Regiones de Baja Atención para la Recuperación de Texto–Imagen

**[Artículo](https://arxiv.org/abs/2606.18885)** | [Taller ICML 2026 EMMQA](https://qanta-org.github.io/competition/2026/icml/) | [Conjunto de Datos Dense-Set](https://huggingface.co/datasets/AbdulmalekDS/Dense-Set)

**LARE** es una técnica de aumento sin entrenamiento para la recuperación de texto a imagen en escenas concurridas. Nuestro método extrae regiones de baja atención de un codificador visual congelado, codifica estas regiones junto con la imagen completa y combina las representaciones regionales con la representación global de la imagen durante la inferencia. Este sencillo procedimiento en tiempo de prueba mejora la recuperación en las variantes de Dense-Set que enfatizan contenido sutil y ocluido.

<br>
<img src="docs/image1.png" width="90%" alt="Detalles de LARE">

</div>

## Inicio Rápido

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

## Reproducir Resultados

> **Hardware**: Todos los benchmarks se evaluaron en dos GPU NVIDIA Quadro RTX 8000. Sin embargo, una GPU estándar de 11GB a 16GB es suficiente para reproducir los resultados.

**Paso 1. Clonar y Configurar el Entorno**
```bash
git clone https://github.com/AbdulmalikDS/LARE.git
cd LARE
conda env create -f environment.yml
conda activate lare
```

**Paso 2. Descargar Conjuntos de Datos**
Visita [`data/README.md`](data/README.md) para obtener instrucciones rápidas y scripts de `wget` sobre cómo descargar los conjuntos de imágenes MS-COCO y Flickr30K localmente.

**Paso 3. Ejecutar Scripts de Benchmark**
Pasa el directorio del conjunto de datos al script de reproducción. Evaluará nativamente la línea base y el flujo de trabajo LARE contra las anotaciones estándar y de Dense-Set.

```bash
COCO_DIR=/path/to/coco/val2014 \
FLICKR_DIR=/path/to/flickr30k-images \
bash scripts/reproduce.sh siglip-so400m
```

## 📜 Citación

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
