# TRELLIS.2 on Kaggle T4 x2: run protocol (terminal)

The same steps every time, from the laptop's terminal. Background and design: [trellis2.md](trellis2.md).

**State (2026-11-17):** everything below is built and tested on the laptop (with fakes), but **has not run on
Kaggle yet**. The first session is the real test; expect to fix things after it.

Everything runs **in a terminal on the laptop, from the project folder**:

```bash
cd ~/video\&kaggle-setups/trellis2-t4
git switch trellis2-t4
KUSER=$(kaggle config view | awk '/username:/ {print $3}'); echo "$KUSER"   # every new terminal
```

| When | Do |
|---|---|
| Once | Part 1 |
| Every session | Part 2 (laptop), Part 3 (start on Kaggle), Part 4 (check), Part 5 (generate) |
| At the end | Part 6 |

---

## Part 1: one-time setup

### 1.1 Laptop tools

| Tool | Check |
|---|---|
| Python 3.9+ | `python3 --version` |
| `cloudflared`, `openssl`, `curl` | `cloudflared --version; openssl version; curl --version` |
| Kaggle CLI ≥ 1.8 with `~/.kaggle/access_token` | `kaggle config view` prints your username |

### 1.2 Queue token: reuse the one from `gpuserverclientinterface`

This folder is a separate clone, and `run/` (the tokens) isn't in git. Copy the existing tokens so the
`gpuq-token` dataset on Kaggle keeps matching both folders:

```bash
mkdir -p run && cp -p ../gpuserverclientinterface/run/token ../gpuserverclientinterface/run/worker_token run/
```

Only one queue runs at a time: `start.sh` in either folder stops the other folder's queue.

### 1.3 Kaggle datasets

| Dataset | What | Status |
|---|---|---|
| `$KUSER/gpuq-token` | the queue token (API-pushed runs get no Kaggle secrets) | exists |
| `$KUSER/dinov3-vitl16-hf` | DINOv3 ViT-L/16, converted, private, with Meta's `LICENSE.md` | exists |

The TRELLIS.2 weights need no dataset: they are open and download faster from Hugging Face (`trellis2.md`,
Kaggle workflow).

---

## Part 2: every session, on the laptop

```bash
./laptop/start.sh                 # queue + tunnel; writes the new address to run/url
python3 laptop/submit.py --list   # empty list, no error = queue is up
python3 laptop/make_bundle.py     # only the first time and after code under worker/ or tools/notebook/ changed
```

The tunnel address changes every time `start.sh` runs; `kaggle_run.py` reads it from `run/url`, so start
Kaggle (Part 3) after `start.sh`. Keep the laptop awake and online during the session.

---

## Part 3: start the Kaggle session

### 3.1 First time only: check the connection (CPU, about a minute)

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/trellis2-t4 --mode probe --token-dataset $KUSER/gpuq-token
```

`/health 200` and `/bundle 200` = the notebook reaches the laptop. `/health 401` = token mismatch (redo 1.2, or
add `--make-token-dataset` to upload this folder's token). `PROBE secret error: ConnectionError` is normal.

### 3.2 Start the session (T4 x2)

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/trellis2-t4 --mode run \
    --token-dataset $KUSER/gpuq-token --dataset $KUSER/dinov3-vitl16-hf \
    --session-args '--session kaggleA --phases preflight,install,trellis2,workers --requirements worker/trellis2_t4/requirements.txt --cmd "trellis2=bash worker/trellis2_t4/setup.sh" --worker-args "--handler trellis2=handlers.trellis2:Trellis2" --worker-idle-exit 900 --budget-minutes 120'
```

- After 3.3 (saved build) add `--dataset $KUSER/trellis2-t4-wheels`; `python3 tools/notebook/trellis2_datasets.py status`
  prints the flags. Every push replaces the notebook's inputs, so all `--dataset`/`--token-dataset` flags go on
  every command.
- Only the setup, no workers: `--phases preflight,install,trellis2` (drop `--worker-args`).
- It pushes the notebook as a new version, starts it and follows its log. `Ctrl-C` only stops following:
  `python3 tools/notebook/kaggle_run.py --kernel $KUSER/trellis2-t4 --logs` shows it again.
- `QUEUED` = waiting for a free T4 x2 (GPU hours don't count yet; don't push again).

Phases: `preflight` (hardware), `install` (pure-Python packages), `trellis2` (setup.sh: TRELLIS.2 source and
patch, weights download and DINOv3 copy in the background, the CUDA extension build), `workers` (one per GPU,
each loads the pipeline, runs a short warm-up, then takes jobs). **The first session compiles the extensions;
how long that takes is not measured yet.**

### 3.3 Save the build (once, after the first session whose `trellis2` phase ended `ok=True`)

```bash
python3 tools/notebook/trellis2_datasets.py wheels
```

It takes the compiled wheels from the latest run of `$KUSER/trellis2-t4` and uploads them as the private
dataset `$KUSER/trellis2-t4-wheels`. setup.sh installs from it only when it was built for the same Python,
torch, CUDA and source commits (`trellis2-t4-wheels.txt`); after Kaggle updates its image it builds again,
and you run this again.

---

## Part 4: check the setup

```bash
python3 laptop/submit.py --workers                  # ready = trellis2 loaded on that GPU
python3 tools/notebook/report.py                    # latest session: phases, ok, times, RAM peaks
less data/reports/kaggleA/trellis2_gpu*.log         # one GPU's log does the work, the other says "already set up"
```

The setup log that did the work ends with lines like:

```text
[trellis2-setup +Ns] extensions: built CuMesh in Ns          (or: installing the wheels from /kaggle/input/...)
[trellis2-setup +Ns] extensions: cumesh, flex_gemm, nvdiffrast, o_voxel import
[trellis2-setup +Ns] weights done: 9.8G
[trellis2-setup +Ns] DINOv3 done: 1.2G
[trellis2-setup +Ns] ready: {...}
```

Write the times into `trellis2.md` (they are the first real numbers). If a build step fails, the log names
`build_<package>.log`; it is in the notebook output under `trellis2/`.

---

## Part 5: generate 3D models

Input: a PNG with a **transparent background** (RGBA). For images with a background, start the workers with
`--worker-args "--handler trellis2=handlers.trellis2:Trellis2 --opt rembg=ZhengPeng7/BiRefNet"` (MIT model).

```bash
python3 laptop/submit.py --kind generate --family trellis2 --config trellis2-stylized@1 \
    --params '{"seed": 3}' --file image.png=character.png --output model.glb --output result.json
# prints the job id; then
python3 laptop/submit.py --list                                     # state: queued / running / done / failed
python3 laptop/submit.py --get JOB_ID model.glb > character.glb
python3 laptop/submit.py --get JOB_ID result.json                   # settings, timings, faces, fp32 fallback
```

Settings (`--params`, override `configs/trellis2-stylized/v1.json`): `seed`, `texture_size` (default 1024),
`decimation_target` (faces, default 50000), `remesh`, `remesh_band`, `remesh_project`, and sampler steps
`sparse_structure_steps`, `shape_steps`, `tex_steps` (upstream default 12 each). Several seeds: submit several
jobs; both GPUs take jobs in parallel. Jobs can be submitted before the workers are ready; they wait.

---

## Part 6: end the session

1. Workers stop after 15 idle minutes (`--worker-idle-exit 900`) or at `--budget-minutes`, and the session
   uploads its report first. Stop earlier on kaggle.com → the notebook → *Cancel run*.
2. The queue on the laptop can keep running for the next session; stop it with `Ctrl-C` in its terminal.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no DINOv3 under /kaggle/input` | dataset not attached to this push | add `--dataset $KUSER/dinov3-vitl16-hf` (3.2) |
| `/health 401` in the probe | `run/token` differs from the `gpuq-token` dataset | redo 1.2, or add `--make-token-dataset` once |
| `HTTP Error 530` / connection errors | `start.sh` was restarted, the address changed | push again (3.2); it reads the new `run/url` |
| `git apply --check` fails in the setup log | upstream moved under the patch (only if `TRELLIS2_COMMIT` was changed) | keep the pinned commit, or update the patch |
| Hugging Face errors in the setup log | Hugging Face down or rate-limited | run again later |
| Run stays `QUEUED` long | no free T4 x2, an open interactive GPU session, or quota used up | close the notebook editor; check Settings → *Quotas* |
