# TRELLIS.2 on Kaggle T4 x2: run protocol (terminal)

The same steps every time, from the laptop's terminal. Background and design: [trellis2.md](trellis2.md).
Meta's DINOv3: [../DINOv3-META.md](../DINOv3-META.md).

**State (2026-11-17):** what runs today is the **setup check**: a Kaggle session that clones and patches
TRELLIS.2, downloads the 512 weights from Hugging Face and copies DINOv3 from your private dataset, then stops.
**Generating 3D models is not built yet** (needs the CUDA extension build and the handler; Part 5 is the
placeholder). This document is updated as those land.

Everything runs **in a terminal on the laptop, from the project folder**:

```bash
cd ~/video\&kaggle-setups/trellis2-t4
git switch trellis2-t4
KUSER=$(kaggle config view | awk '/username:/ {print $3}'); echo "$KUSER"   # every new terminal
```

| When | Do |
|---|---|
| Once | Part 1 |
| Every session | Part 2 (laptop), Part 3 (start on Kaggle), Part 4 (check) |
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

### 3.2 Setup check (T4 x2, about 5–10 minutes, no workers)

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/trellis2-t4 --mode run \
    --token-dataset $KUSER/gpuq-token --dataset $KUSER/dinov3-vitl16-hf \
    --session-args '--session kaggleA --phases preflight,trellis2 --cmd "trellis2=bash worker/trellis2_t4/setup.sh" --budget-minutes 30'
```

Every push replaces the notebook's inputs, so the two `--dataset`/`--token-dataset` flags must be on every
command. It pushes the notebook as a new version, starts it and follows its log. `Ctrl-C` only stops
following; show it again with:

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/trellis2-t4 --logs
```

`QUEUED` = waiting for a free T4 x2 (GPU hours don't count yet; don't push again).

---

## Part 4: check the result

```bash
python3 tools/notebook/report.py                    # latest session: phases, ok, times, RAM peaks
less data/reports/kaggleA/trellis2_gpu*.log         # one GPU's log does the work, the other says "already set up"
```

The log that did the work ends with:

```text
[trellis2-setup +Ns] DINOv3: copying /kaggle/input/.../dinov3-vitl16-hf -> /tmp/dinov3-vitl16-hf in the background
[trellis2-setup +Ns] source at 75fbf01..., patches applied
[trellis2-setup +Ns] weights done: 9.8G
[trellis2-setup +Ns] DINOv3 done: 1.2G
[trellis2-setup +Ns] ready: {"commit":...,"dino":"/tmp/dinov3-vitl16-hf","weights":{...}}
```

Write the times of `weights done` and `DINOv3 done` into `trellis2.md` (they are the first real numbers).

---

## Part 5: generate 3D models (not built yet)

Needs the CUDA extension build (cumesh, o_voxel, nvdiffrast, flex_gemm) and `worker/handlers/trellis2.py`.
Then the session gets a `workers` phase and you submit images with `laptop/submit.py`; this part will
describe it.

---

## Part 6: end the session

1. The setup check stops by itself after its phases (or at `--budget-minutes`) and uploads its report first.
   A running session can be stopped on kaggle.com → the notebook → *Cancel run*.
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
