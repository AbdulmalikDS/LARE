# Benchmark caption files

Caption files used by `scripts/reproduce.sh`. Images are not committed; download:
- COCO val2014: <https://cocodataset.org/#download>
- Flickr30K images: <https://github.com/BryanPlummer/flickr30k_entities>

| File | Used for |
|------|----------|
| `captions/flickr30k_dataset.json` | Flickr30K (Karpathy-style split) |
| `captions/MSCOCO_val2014_denseset_blip.json` | COCO-Dense (3089 images) |
| `captions/flickr30k_denseset_blip_gpt_oss.json` | Flickr30K-Dense (2477 images) |

COCO-Karpathy captions are loaded automatically from HuggingFace
(`yerevann/coco-karpathy`), so no file is committed for it.

See [`../RESULTS.md`](../RESULTS.md) for expected numbers.
