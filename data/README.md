# Caption files

Images are not committed due to their large size. Please download them locally:

### 1. MS-COCO 2014
To download the MS-COCO 2014 Validation split natively via your terminal:
```bash
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

### 2. Flickr30K
Flickr30K requires explicit permission from researchers. Download the `flickr30k-images` dataset formally via the [Kaggle repository](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) or the [official Illinois University entity page](https://github.com/BryanPlummer/flickr30k_entities).

| File | Dataset | N_img |
|------|---------|------:|
| `captions/coco_dense.json` | COCO-Dense | 3089 |
| `captions/flickr30k.json` | Flickr30K (test split) | 1000 |
| `captions/flickr30k_dense.json` | Flickr30K-Dense | 2477 |

See [`../RESULTS.md`](../RESULTS.md) for expected numbers.
