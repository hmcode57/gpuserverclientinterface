# Training one LoRA adapter across sessions (DiLoCo)

Several notebook sessions (several Kaggle T4 x2, Colab, teammates' accounts) train **one** LoRA adapter
together. Sessions never talk to each other, and weights never pass through your laptop or home connection.

## How it works

Standard multi-GPU training syncs gradients after every step, which needs GPUs that can reach each other in
milliseconds. Notebook sessions can't. DiLoCo (Douillard et al., 2023, "DiLoCo: Distributed Low-Communication
Training of Language Models") syncs every few hundred steps instead:

```
 laptop: diloco_train.py (schedules only, a few KB per round)
     │ jobs with presigned URLs
     ▼
 ┌────────── round r ───────────────────────────────────────────────────────────────┐
 │ session A: download θ_r ─► torchrun on both T4s, H AdamW steps on shard A ─► upload Δ_A │
 │ session B: download θ_r ─► torchrun on both T4s, H AdamW steps on shard B ─► upload Δ_B │
 │ any session: reduce job: Δ = mean(Δ_k), Nesterov outer step ─► upload θ_{r+1}            │
 └──────────────────────────────────── all through Cloudflare R2 ──────────────────┘
```

- **Inside a session** the two T4s run normal DDP (torchrun), so it's true data parallelism there.
- **Between sessions** each replica trains on its own data shard for H steps; the pseudo-gradients
  Δ_k = θ_r − θ_k are averaged and applied with an outer Nesterov step (lr 0.7, momentum 0.9 in the paper).
- **Δ is sent as int8** (per-row scales, 4x smaller than fp32). θ and the momentum stay fp32.
- **Each replica's AdamW state** is kept between rounds in R2 (as in the paper), so whichever session runs
  replica k next round continues where it left off. `--no-inner-state` turns that off to save storage.
- **Round r of replica k always gets the same data**, whichever session runs it, so retries are exact.

Fault tolerance comes from the queue: a session dying mid-round requeues its job; a replica that keeps
failing is resubmitted (`--replica-retries`) and then dropped from that round; `--straggler-timeout` cancels
replicas still running that long after the first one finished (their GPU is freed immediately). A round
closes with at least `--min-replicas` (default half). New sessions can join any time and just pick up jobs.

## Data volumes

For the example config (Qwen2.5-1.5B, LoRA r=16 on all linear layers, about 18.5M trainable parameters):

| Object | Size | Per round |
|---|---|---|
| θ (fp32) | ~74 MB | downloaded by every replica and the reducer, uploaded once |
| Δ_k (int8) | ~19 MB | uploaded by each replica, downloaded by the reducer |
| AdamW state per replica | ~148 MB | downloaded and uploaded by each replica (`--no-inner-state` skips it) |
| **through your home connection** | | **a few KB of job JSON** |

Sessions move a few hundred MB per round to and from R2, which takes seconds from Kaggle or Colab.
The driver deletes what isn't needed any more (`--keep-rounds 2` keeps the last two rounds' weights), so a
run stays well inside R2's free 10 GB. Add a bucket lifecycle rule on the `diloco/` prefix (for example,
delete after 14 days) to catch leftovers from cancelled stragglers.

## One-time R2 setup

1. Cloudflare dashboard → **R2** → create a bucket, e.g. `gpuq`.
2. **R2 → Manage API tokens → Create API token**: permission **Object Read & Write**, limited to that bucket.
   Note the **Access Key ID**, **Secret Access Key** and your **Account ID**.
3. On the laptop only (never in a notebook):
   ```bash
   export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=gpuq
   ```
   Notebooks only ever see presigned URLs for the objects of their current job, valid for `--url-hours`
   (48 by default, at most 168).

## Running

Laptop:

```bash
./laptop/start.sh                                   # queue + tunnel (see docs/multi-session.md)
python3 laptop/make_bundle.py                       # includes worker/diloco and the recipes
python3 laptop/diloco_train.py --run qwen-lora --cfg examples/diloco/qwen2.5-1.5b-lora.json \
    --replicas 3 --rounds 30 --steps 100
```

Each notebook (one per session; a teammate's uses `run/worker_token`):

```python
SESSION_ARGS = ('--session kaggleA --phases preflight,install,workers '
                '--requirements worker/diloco/requirements.txt --gpus all '
                '--worker-args "--handler diloco=diloco.handler:DiLoCo" --budget-minutes 700')
```

`--gpus all` is important: it starts **one** worker that sees both T4s, and the handler runs torchrun across
them. Without it you'd get two single-GPU workers, which also works but each replica uses one GPU.

The driver prints a line per replica and per round, and keeps its state in `data/diloco/<run>/state.json`.
Stop it any time and run the same command again: it resumes, reusing jobs that are already queued or
running. Raise `--rounds` on a finished run to train longer. At the end an export job writes a standard PEFT
adapter to `data/diloco/<run>/adapter/`:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_id), "data/diloco/qwen-lora/adapter")
```

## Choosing the settings

| Setting | Guidance |
|---|---|
| `--replicas` | the number of sessions you expect to run at once; more replicas than sessions just queue |
| `--steps` (H) | aim for 10–30 minutes of training per round, so transfer time is small next to it. The paper used 500 |
| `--outer-lr`, `--outer-momentum` | the paper's 0.7 / 0.9 assumes many rounds and large H. On a small test problem with few rounds and H=15 they overshot (loss went up for a few rounds before settling), while `--outer-lr 1.0 --outer-momentum 0.5` fell every round. With fewer than ~10 rounds, try 1.0 / 0.5 first, or 1.0 / 0 (plain averaging) |
| `lr`, `warmup` in the cfg | the inner AdamW settings, as for single-GPU LoRA. The schedule is warmup + cosine over all rounds |
| `micro_batch`, `grad_accum` | per GPU. Global batch per inner step = micro_batch × grad_accum × GPUs in the session |
| `load_in_4bit` | QLoRA, for bases that don't fit a T4 in fp16 (above about 3B). Add `bitsandbytes` to the requirements |

## Your own model or data

The built-in recipe `diloco.recipes.hf_lora` covers Hugging Face causal LMs with text or chat data (any HF
dataset, or `"dataset": "json", "data_files": "..."` for a file in the bundle). For anything else write a
recipe: a module with `load_model`, `batches` and `loss` (see `worker/diloco/recipes/__init__.py`), include
it in the bundle, and pass `--recipe yourmodule`. Every replica must build the same trainable parameter
names and shapes, so pin `revision` for the base model.

## Limits

- **Not yet run on real Kaggle GPUs or a real R2 bucket.** Verified on CPU: the math against
  `torch.optim.SGD`, the full pipeline over HTTP with two workers (one using 2-process DDP), failures,
  stragglers, resume and cleanup with a toy problem, and the HF recipe with a tiny Llama (init, 1- and
  2-process training, reduce, export loaded back with `PeftModel.from_pretrained`). The SigV4 presigning
  matches AWS's published example.
- **Each GPU holds the whole base model** (DDP, not FSDP). With T4s that means about 3B in fp16 or 7–8B in
  4-bit.
- **Quality vs. a single machine:** DiLoCo papers report results close to fully synchronous training at
  H≈500 with 8 replicas on pre-training. LoRA fine-tuning with a few sessions is a gentler setting, but it
  hasn't been measured here; compare against a short single-session run on the same data first.
- **Gated models** need `HF_TOKEN` in the notebook's environment (the cell only loads `QUEUE_TOKEN`).
