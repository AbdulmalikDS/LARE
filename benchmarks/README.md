# Benchmark caption files

Caption JSON files used to reproduce the numbers in [`../RESULTS.md`](../RESULTS.md).

Images are **not** committed here (too large) — download them separately:
- **COCO val2014** (for COCO-Karpathy + COCO-Dense): <https://cocodataset.org/#download>
- **Flickr30K** (images + `dataset.json`): <https://github.com/BryanPlummer/flickr30k_entities>

## Files

| File | Dataset | N_img | Expected baseline R@1 |
|------|---------|------:|----------------------:|
| `captions/MSCOCO_val2014_denseset_blip.json` | COCO-Dense (BLIP-2 captions) | 3089 | 13.64 |
| `captions/MSCOCO_val2014_denseset_gpt_oss_refined_mscoco_captions.json` | COCO-Dense (GPT-refined MSCOCO) | 3089 | 41.76 |
| `captions/val2014_filtered_denseset_blip_captions.json` | COCO-Dense (filtered BLIP-2) | 2823 | 11.91 |
| `captions/flickr30k_denseset_blip_gpt_oss.json` | Flickr30K-Dense | 2477 | 9.93 |

_All baselines: SigLIP So/14 (`siglip-so400m`), T→I retrieval, full split._

## Reproduce

```bash
# 1) COCO-Dense with BLIP-2 captions
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json benchmarks/captions/MSCOCO_val2014_denseset_blip.json

# 2) COCO-Dense with GPT-refined MSCOCO captions
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/coco/val2014 \
    --captions-json benchmarks/captions/MSCOCO_val2014_denseset_gpt_oss_refined_mscoco_captions.json

# 3) Flickr-Dense
python scripts/eval.py --model siglip-so400m --dataset dense_set \
    --images-dir /path/to/flickr30k-images \
    --captions-json benchmarks/captions/flickr30k_denseset_blip_gpt_oss.json
```

Add `--pipeline` to enable LARE (otherwise baseline global-only scoring).

## Note on paper Table 3

The paper reports COCO-Dense baseline R@1 = **26.61** (3089 imgs). None of the caption
files above reproduce that number exactly — it falls between `denseset_blip.json` (13.64)
and `denseset_gpt_oss_refined` (41.76). We suspect the paper used a third caption file
or a different sub-split. **The captions here reproduce the numbers in this repo's
`RESULTS.md` byte-for-byte**, not the paper's Table 3 Dense-Set numbers.

For COCO-Karpathy and Flickr30K, paper Table 3 reproduces exactly (see `RESULTS.md`).
