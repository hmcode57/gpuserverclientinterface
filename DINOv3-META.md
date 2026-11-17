# DINOv3 (Meta)

The TRELLIS.2 setup in this repository (`worker/trellis2_t4/`) uses Meta's **DINOv3** as its image encoder:
**ViT-L/16, pretrained on LVD-1689M** (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`), from
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3). DINOv3 is licensed under the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) (last updated August 19, 2025),
Copyright © Meta Platforms, Inc. A verbatim copy is in [DINOv3-LICENSE.md](DINOv3-LICENSE.md).

## Where the licence is

The DINOv3 License (section 1.b.i) asks for a copy of the agreement **with the DINO Materials** whenever they are
distributed or made available. So a copy also lives with the weights:

| Copy of the weights | Licence copy |
|---|---|
| Your private Kaggle dataset `$KUSER/dinov3-vitl16-hf` | `LICENSE.md` in the dataset |
| The local folder it was uploaded from | `LICENSE.md` in that folder |

This repository contains no DINO Materials (no weights, no DINOv3 code), only code that loads the weights from
the dataset at run time, and the licence copy above.

## What was done to the weights

The original checkpoint (SHA-256 starts with `8aa4cbdd`, as in the file name) was converted on 2026-11-17 to
the Hugging Face transformers format (`config.json`, `model.safetensors`, `preprocessor_config.json`) with
transformers' own `convert_dinov3_vit_to_hf.py`, run unchanged except for reading the local file and
`map_location='cpu'`. The converter checked the result against Meta's reference outputs (all within 1e-3). The
converted copy is a derivative of the DINO Materials and falls under the same licence.

## Obligations to keep in mind (DINOv3 License, section 1.b)

- Sharing the weights, or anything derived from them, with anyone else: only under the DINOv3 License, with
  a copy of it. The Kaggle dataset is private.
- Publishing research results that used DINOv3: acknowledge DINOv3 in the publication.
- Use must comply with applicable law and trade controls; no reverse engineering; no military, weapons,
  nuclear or espionage uses.

## Credit

DINOv3: Oriane Siméoni et al., *DINOv3*, Meta AI Research, 2025. https://github.com/facebookresearch/dinov3
