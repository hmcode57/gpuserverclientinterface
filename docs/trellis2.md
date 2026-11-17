# TRELLIS.2 on free Kaggle T4s (architecture, not built yet)

Plan for running Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D model through this
queue on the free Kaggle tier, for stylized and cartoon assets used in videos. **Nothing here has run on a GPU
yet.** Statements marked *verified* were checked against the TRELLIS.2 source (cloned November 2026); the rest
are estimates or open questions that the first Kaggle test run (milestone 2) answers.

```text
laptop                                   Kaggle T4 x2 (free tier, temporary)
──────                                   ───────────────────────────────────
viewer.py ──► queue_server.py ◄─tunnel── supervise.py
 (submit,      (unchanged)                 ├─ worker gpu0 ─┐ worker/handlers/trellis2.py
  grid,                                    └─ worker gpu1 ─┘   └─ trellis2-t4 (fork of TRELLIS.2)
  3D view,                                 /kaggle/input/
  compare)                                   ├─ trellis2-t4-wheels    (compiled once)
                                             └─ trellis2-512-weights  (fp16, 512 path only)
```

## Goal and why it fits

- **Job:** one image in (PNG with a transparent background), one `model.glb` plus a `preview.png` out.
- **Stylized output needs much less than the defaults.** The `512` pipeline, 1024–2048 px textures and about
  20k–100k faces are enough for cartoon assets. Upstream defaults are `1024_cascade`, 4096 px textures and
  1M faces. The lower settings are the main reason a 16 GB T4 is realistic at all. Upstream asks for 24 GB and
  has only tested on A100 and H100.
- **Style comes from the input image.** TRELLIS.2 reproduces the look of the picture, so feed it cartoon
  concept art. Its textures are PBR; render with an unlit or toon shader for a flat look.
- **Fits the queue's sweet spot:** independent jobs that take minutes, small files, a model that stays loaded.

## Free Kaggle tier limits that shape the design

Check these against your account settings and a fresh `preflight`; Kaggle changes them.

| Limit | Value | Consequence |
|---|---|---|
| GPU quota | about 30 h/week, varies (account settings). T4 x2 reportedly costs the same as one GPU | Always use T4 x2 and keep both GPUs busy |
| Session length | up to 12 h | Long sessions; `--budget-minutes` |
| RAM | 29–31 GB per session, shared by all workers (measured, `docs/notebooks.md`) | Two workers fit only with the 512-only fp16 loader (problem 1) |
| Disk | `/kaggle/working` about 20 GB, kept between versions | The build run can save the wheels and the 512-only fp16 weights there, then turn them into datasets |
| Python | 3.13 (measured November 2026) | Wheels are built for cp313 and Kaggle's exact torch; pin the notebook's environment |
| Secrets | don't reach API-pushed runs (measured, `docs/notebooks.md`) | The Hugging Face token is needed only in the one-time build run, done in the web UI |
| Accounts | phone verification for GPU and internet; one account per person | Pool quota only with teammates' own accounts (`docs/multi-session.md`) |

Colab's free tier (about 12.7 GB RAM) fits one worker at best. Avoid Kaggle's P100: compute capability 6.0,
no tensor cores.

## Problems and fixes

### 1. Memory

*Verified:* `Pipeline.from_pretrained` loads every model in `pipeline.json`, including the 1024-resolution flow
models the `512` path never uses: five flow models of 1.3B parameters each (`dit_1_3B`, 1536 channels, 30
blocks). The flow model classes default to `dtype='float32'`; whether the checkpoint configs override that is
unverified (huggingface.co was unreachable). So one worker needs roughly 13–26 GB of CPU RAM as upstream
ships it, and two workers do not fit.

*Verified:* `low_vram` (default `True`) moves one sub-model at a time to the GPU and back.

**Fix:** the fork's 512-only loader (`model_names_to_load` already exists as a hook) loads the sparse-structure
flow and decoder, the 512 shape and texture flows and the two decoders, converted to fp16 at load. That is
about 3 × 2.6 GB of flow weights, roughly 8 GB per worker, so two workers fit in 29–31 GB. Keep `low_vram`,
set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, use batch 1, and use the smaller GLB settings above.
VRAM at 512 is expected to fit (largest sub-model about 2.6 GB in fp16 plus activations); the test run
measures it, since the runtime already records `peak_vram_gb` and `peak_rss_gb` per job.

### 2. Attention on T4 (no flash-attn)

*Verified:* dense attention accepts `sdpa`, but sparse attention (`trellis2/modules/sparse/config.py`) accepts
only `xformers`, `flash_attn` and `flash_attn_3`. Setting just `ATTN_BACKEND=sdpa` makes sparse attention
silently keep its `flash_attn` default. Classifier-free guidance runs the conditioned and unconditioned passes
separately and `num_samples` is 1, so every sparse attention call is one sequence. No flow model uses windowed
attention.

| Option | Pros | Cons |
|---|---|---|
| xformers | no code change, supports T4 in fp16 | the wheel must match Kaggle's exact torch; building it takes 30+ min |
| **sdpa patch (chosen)** | about 10 lines for the single-sequence case plus a padded batch path; no extra dependency, survives torch updates | small fork patch |

Settings: `ATTN_BACKEND=sdpa`, `SPARSE_ATTN_BACKEND=sdpa` (after the patch).

### 3. Precision (T4 has no bf16 hardware)

*Verified:* the flow configs are named `*_bf16`; the VAE decoders are already fp16 (`use_fp16: true`); the
flow models use RMS norm on queries and keys (`qk_rms_norm`) and fp32 LayerNorms (`LayerNorm32`), which lowers
the fp16 overflow risk. `convert_to(dtype)` exists on the flow models.

| Option | Verdict |
|---|---|
| keep bf16 | not viable for speed or attention kernels on T4 |
| **fp16 (chosen)** | fast (tensor cores); check every stage for NaN/inf |
| fp32 | safe, about 5.2 GB per flow model, several times slower; used as a per-stage fallback |
| hybrid (selected layers in fp32) | only if one stage keeps failing in fp16 |

The fork checks each stage's output with `torch.isfinite` and re-runs only a failing stage in fp32, recording
that in the job metrics. The parity tool (below) measures fp16 against an fp32 reference stage by stage.

### 4. Install, compile and Triton

- **Compiled extensions:** cumesh, o_voxel, nvdiffrast (and flex_gemm). Build them once for the T4 only
  (`TORCH_CUDA_ARCH_LIST=7.5`), which doesn't need a GPU, so try a CPU session first and fall back to a GPU
  session if `nvcc` is missing there. Save them as the private dataset `trellis2-t4-wheels`; every session
  then installs from it in about a minute via `--requirements`.
- **Triton:** officially supports compute capability 8.0+; the T4 is 7.5. *Verified:* `flex_gemm` (Triton) is
  imported at load by `trellis2/representations/mesh/base.py`, is the default sparse-convolution backend, and
  is used by `o_voxel/postprocess.py` (`grid_sample_3d`) for GLB export. Fixes: `SPARSE_CONV_BACKEND=spconv`
  (`conv_spconv.py` exists; untested with these weights) and a plain-PyTorch `grid_sample_3d` fallback in the
  fork. **This is the highest remaining risk.**

### 5. Gated and licensed models

- **Image encoder:** DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`). Possibly gated; accept its license on
  Hugging Face, download it once in the build run and keep it in the private weights dataset.
- **Background removal:** `pipeline.json` points to `briaai/RMBG-2.0`, reported as CC BY-NC 4.0 (no
  commercial use). *Verified:* it is loaded on every start, and skipped at inference only when the input has
  an alpha channel. Fix: the fork doesn't load it when inputs are RGBA, or swaps to `ZhengPeng7/BiRefNet`
  (MIT, the code's own default).

## Two GPUs: independent workers, not a stage split

Kaggle's two T4s have no NVLink; only a few MB of latents would pass between stages, so the link speed doesn't
matter.

| | 2 independent workers (default) | Stage split (fallback) |
|---|---|---|
| Time for one job | same | same: stages still run one after another |
| Throughput | both GPUs always busy | one GPU waits when stages are unbalanced |
| VRAM headroom per stage | best: one model at a time (`low_vram`) | less: GPU 0 keeps DINOv3 and two flow models resident (about 6 GB) |
| CPU RAM | two weight copies (about 2 × 8 GB with the 512 loader) | one copy |
| Code | `--gpus auto`, works today | fork patch mapping each stage to a device; one worker with `--gpus all` |

Use the stage split only if the test run shows two workers run out of CPU RAM.

## Components

### Repo `trellis2-t4` (fork of microsoft/TRELLIS.2, to be created)

Each patch is separate and switched on by an environment variable, so upstream merges stay easy and the fork
behaves like upstream with nothing set.

| Patch | Contents | Problem |
|---|---|---|
| sparse `sdpa` backend | single-sequence path via `torch.nn.functional.scaled_dot_product_attention`, padded batch path, `'sdpa'` accepted in sparse config | 2 |
| precision setting | `TRELLIS_DTYPE=fp16\|fp32`; convert at load; per-stage NaN check with fp32 retry | 3 |
| 512-only pipeline | loads only the 512-path models; background removal optional or MIT BiRefNet | 1, 5 |
| flex_gemm fallback | plain-PyTorch `grid_sample_3d`; spconv switch for sparse convolution | 4 |
| stage API | `run_stages()` yields after each stage: progress, NaN checks, lease checks between stages | integration |
| stage dumps | save each stage's tensors for parity checks (kimodo.cpp `reference/` method) | 3 |
| device per stage (later, if needed) | stage-split mode | 1 (fallback) |
| CPU tests | sdpa against naive attention, 512 loader, grid_sample fallback | no GPU needed |

### This repo (gpuserverclientinterface)

| Path | Purpose |
|---|---|
| `worker/handlers/trellis2.py` | thin handler over `run_stages()`: reads `image.png`, writes `model.glb` and `preview.png`, reports stage timings, NaN fallbacks, faces and texture size |
| `configs/trellis2-stylized/v1.json` | `512` pipeline, 1024–2048 px textures, about 50k faces, fp16, sampler steps |
| `requirements/trellis2-t4.txt` | `--find-links /kaggle/input/trellis2-t4-wheels` plus pure-Python dependencies (never torch) |
| `tools/trellis2/build_kaggle.py` | one-time build run: compile wheels, export 512-only fp16 weights to `/kaggle/working`, write a SHA-256 manifest |
| `tools/trellis2/parity.py` | fp16 against fp32 per stage: mean, p50, p95 and max divergence plus a non-finite check (after kimodo.cpp `scripts/quantization_metrics.py`) |
| `tools/notebook/session.py` | new `--once` option for custom phases (today `--cmd` phases run once per GPU in parallel) |
| `laptop/viewer.py` | local viewer, see below |
| `docs/trellis2.md` | this document, updated with measured numbers |

**No queue protocol changes.** The viewer and the handler use existing endpoints only.

Handler outline:

```python
# worker/handlers/trellis2.py (sketch)
import os
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from runtime import Handler

class Trellis2(Handler):
    def load(self):        # 512-only pipeline from /kaggle/input/trellis2-512-weights, fp16, low_vram
        ...
    def warmup(self):      # one tiny job so Triton/kernels compile before real jobs
        ...
    def run(self, job, ctx):
        # ctx.input("image.png") -> run_stages(): after each stage ctx.progress(phase=...), NaN check
        # -> GLB with the job's texture size and face count -> ctx.output("model.glb"), ctx.output("preview.png")
        # on CUDA OOM: empty cache, retry once with lower settings, record it in metrics
        ...
```

Submitting a job:

```bash
python3 laptop/submit.py --kind trellis2 --family trellis2 --config trellis2-stylized@1 \
    --params '{"seed": 3}' --file image.png=character.png
```

### Viewer (`laptop/viewer.py`)

Standard library only, on `127.0.0.1`, reads `run/token` and proxies the queue's existing endpoints (job list
and details, file download, job submit, workers).

- **Submit:** drag and drop an image; seed, texture size, faces; N variations with different seeds.
- **Jobs:** live grid with state, current stage, input thumbnail and `preview.png`.
- **3D view:** orbit the GLB with `<model-viewer>`, flat or lit display, download.
- **Compare:** seeds or fp16 against fp32 side by side.
- **Workers:** state, peak VRAM and RAM, stage timings.
- **License labels** per model, including whether commercial use is allowed.

The layout follows kimodo.cpp's `demo/` (history on disk, restoring settings from an old item, one resident
model, `/compare`). Its canvas renderer draws skeletons only, so the 3D view uses `<model-viewer>` instead.

### kimodo.cpp

No changes now. Possible later: a `kimodo` job family in the same queue (text-to-motion), plus an auto-rigging
step so TRELLIS.2 characters can be animated. If code is copied from it rather than ideas, keep its LICENSE
and NOTICE.

## Kaggle workflow

1. **Build run (once, web UI so secrets work):** compile wheels, download DINOv3 with an `HF_TOKEN` secret,
   export the 512-only fp16 weights to `/kaggle/working`. Create the datasets `trellis2-t4-wheels` and
   `trellis2-512-weights` from the output. Pin the notebook's environment.
2. **Test run (once):** parity fp16 against fp32, peak VRAM and RAM with one and two workers, Triton against
   the spconv and PyTorch fallbacks, time per job. Write the numbers into this document.
3. **Production sessions:** attach both datasets, then
   ```python
   SESSION_ARGS = ('--session kaggleA --phases preflight,install,workers '
                   '--requirements requirements/trellis2-t4.txt '
                   '--worker-args "--handler trellis2=handlers.trellis2:Trellis2" '
                   '--worker-idle-exit 600 --budget-minutes 700')
   ```
   driven by `tools/notebook/kaggle_run.py --mode run` (interactive sessions shut down when idle).

Budget: about 2 GPU-hours once for the build and test runs, leaving about 28 h/week. Throughput estimate, to be
replaced by measurements: if a 512 job takes 1–3 minutes on a T4, two GPUs give about 40–120 assets per hour.
Each GLB is a few MB at about 50k faces with 1024 px textures.

## Milestones

1. **Fork and CPU-tested patches:** sdpa backend, 512-only loader, grid_sample fallback.
2. **One Kaggle test run:** build wheels and weights, parity, memory, Triton. Answers problems 1–4.
3. **Handler, config and viewer.** The viewer can be built earlier, tested with the `echo` handler.
4. **Production:** seed sweeps through the viewer; stage-split mode and a kimodo family only if needed.

## Open questions (answered by milestone 2)

- Do the flow checkpoints set `dtype` (bf16) or run in fp32 by default?
- Does fp16 stay finite through all stages, and how far does it drift from fp32?
- Does Triton/flex_gemm work on the T4, and does `spconv` load these weights correctly?
- Is DINOv3 gated for your account?
- Peak VRAM per stage and CPU RAM per worker; seconds per job at the stylized settings.

## Sources

- [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2): `pipelines/base.py`,
  `pipelines/trellis2_image_to_3d.py`, `pipelines/rembg/BiRefNet.py`, `modules/sparse/*`, `configs/gen`,
  `configs/scvae`, `o-voxel/o_voxel/postprocess.py`
- [Triton supported hardware](https://github.com/triton-lang/triton)
- [TRELLIS.2 texturing_pipeline.json (RMBG-2.0 reference)](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/texturing_pipeline.json)
- [Free 29 GB RAM and 4 CPU cores Kaggle GPU notebooks update](https://lilys.ai/en/notes/google-tpu-20251128/free-kaggle-gpu-29gb-4-cpu-update)
- [Free cloud GPUs in 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- This repo's `docs/notebooks.md`, section 6 (Kaggle measurements, November 2026)
