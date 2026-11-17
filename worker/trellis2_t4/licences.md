# Sources and licences (trellis2_t4)

Credits for everything except Meta's DINOv3, which has its own file: `DINOv3-META.md` in the repository root.

Where the code and models used by `worker/trellis2_t4/` come from. The files themselves carry no source notes;
this list is the record. Licences checked on 2026-11-17; recheck before publishing anything.

## Code

| Our file | What came from where | Source | Licence |
|---|---|---|---|
| `patches/0001-sparse-sdpa.patch` | The patched files are TRELLIS.2's `trellis2/modules/sparse/config.py` and `trellis2/modules/sparse/attention/full_attn.py` at commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | MIT |
| `patches/0001-sparse-sdpa.patch` | `_sdpa_varlen()` copied unchanged (body and comments) from the nested function of the same name in `trellis2/modules/sparse/attention/full_attn.py`, last changed in commit `848162efc48cafb3c04fede07cb84cfe0fa481b5` (2026-01-02); repo head checked: `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`. Moved to module level; the `elif config.ATTN == 'sdpa'` branch and the `'sdpa'` entries in `config.py` are ours. | [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) | MIT |
| `setup.sh` | Structure (lock, `ready.json`, once per session) follows our own `worker/kimodo/setup.sh` on the `kimodo-t4` branch. Clones TRELLIS.2 at the commit above at run time; nothing from TRELLIS.2 is stored in this repo except the patch context. | this repo | (ours) |
| `pipeline_512.py` | Ours. The idea of dropping the two 1024 flow models from `pipeline.json` and replacing the background remover also appears in [vidproject1/trellis2-kaggle-bootstrap](https://github.com/vidproject1/trellis2-kaggle-bootstrap) `runtime.py` (untested there; no code copied). | this repo | (ours) |
| `tests/test_trellis2.py` | Ours. | this repo | (ours) |
| `tools/notebook/kaggle_run.py` (`--dataset`) | Ours. | this repo | (ours) |

MIT requires keeping the copyright and permission notice with copies of substantial portions: TRELLIS.2
(Copyright (c) Microsoft Corporation) and ComfyUI-Trellis2 (see its `LICENSE`).

## Models (downloaded at run time, not stored here)

| Model | Used for | Licence | Access |
|---|---|---|---|
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | flow models, shape and texture decoders, `pipeline.json` | MIT | open |
| [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | sparse-structure decoder (`ss_dec_conv3d_16l8_fp16`) | MIT | open |
| [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | upstream's default background remover; **not loaded** by `pipeline_512` (`rembg='none'`) | custom ("other"), reported CC BY-NC 4.0 | gated |
| [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | optional background remover (`rembg='ZhengPeng7/BiRefNet'`) | MIT | open |
