# Caption files

Images are not committed — download from:
- COCO val2014: <https://cocodataset.org/#download>
- Flickr30K: <https://github.com/BryanPlummer/flickr30k_entities>

| File | Dataset | N_img |
|------|---------|------:|
| `captions/coco.json` | COCO test split | 5000 |
| `captions/coco_dense.json` | COCO-Dense | 3089 |
| `captions/flickr30k.json` | Flickr30K | 31783 |
| `captions/flickr30k_dense.json` | Flickr30K-Dense | 2477 |

> `coco.json` is provided for reference. `scripts/eval.py --dataset karpathy`
> fetches the same split automatically from HuggingFace (`yerevann/coco-karpathy`).

See [`../RESULTS.md`](../RESULTS.md) for expected numbers.
