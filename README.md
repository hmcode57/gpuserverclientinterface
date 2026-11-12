# gpuserverclientinterface

Run GPU jobs on free or temporary notebook GPUs (Kaggle, Colab, any rented box) while your laptop stays in
charge. The laptop holds a durable job queue; notebook workers connect **out** to it through an HTTPS tunnel,
pull jobs, and push results back. Notebooks can die at any moment without losing work.

```text
Laptop (persistent)                                  GPU notebook session (temporary, outbound only)
-------------------                                  ----------------------------------------------
submit.py / your app ---> queue_server.py  <--HTTPS tunnel---  supervise.py
                          SQLite + files                         ├── runtime.py on GPU 0 ── your handler
                          configs/, reports/                     └── runtime.py on GPU 1 ── your handler
```

- **Laptop owns all durable state**: queue database, input and output files, configs, session reports.
- **Workers own only temporary GPU state**: loaded models and the job in progress.
- **The laptop never connects into a notebook.** Workers poll, so they work behind any NAT or firewall.
- **Nothing to install on the laptop side**: the queue and client are Python standard library only.

The framework is model-agnostic. You write a small **handler** class for your workload (an LLM, a diffusion
model, a training step, a benchmark) and the framework handles transport, scheduling, retries, crash
recovery, multi-GPU splitting and monitoring.

## When to use it

**Good fit:** batches of independent GPU jobs that take seconds to tens of minutes each, on GPUs you don't
control and that can vanish: image, video and audio generation, batch LLM inference, embedding or captioning
a dataset, transcription, benchmarks, hyperparameter sweeps of short runs, multi-step pipelines. Several free
providers can serve one queue at once.

**Poor fit:**
- **Live serving.** Polling through a tunnel adds seconds of delay.
- **Multi-day training.** There are no checkpoints yet, so a retried job starts over.
- **Multi-GPU training of one model.** Workers don't talk to each other.
- **Gigabytes per job.** Every file goes through the laptop's connection.
- **Multi-user or sensitive workloads.** There's one shared token, and data passes through Cloudflare and the provider.
- **Millisecond jobs.** Per-job HTTP overhead dominates, so batch them inside a handler.

## What it handles for you

| Concern | How |
|---|---|
| Communication | Authenticated HTTP API (Bearer token) on 127.0.0.1, exposed by a Cloudflare quick tunnel; clients retry transient tunnel errors (502/503/504/52x/530) safely |
| Exactly-one execution | Atomic claim in SQLite; every claim returns a **lease** that heartbeat, upload, complete and fail must carry |
| Dead notebooks | Heartbeats every 30 s; a job with no heartbeat for 170 s goes back to the queue; 2 failed attempts mark it `failed` |
| Stale processes | A worker whose job was requeued is fenced off (409 `lease_lost`) even if it reuses the same name, and stops wasting GPU time |
| Lost responses | Claims retried with the same `request_id` return the same job; retried complete/fail return the stored result |
| Load splitting | One worker process per GPU (`CUDA_VISIBLE_DEVICES`); every worker pulls the next job it can run, so no GPU idles while work is queued |
| Model residency | One model family resident per worker; the queue prefers jobs for the family already loaded, so models are rarely swapped |
| Routing | Jobs name a `family`; workers declare which families they `accept`; `waiting_for` explains why a job isn't running |
| Ordering | `priority`, then age; `after` makes a job wait for another one to be `done` |
| Crash recovery | A job that raises is reported with its traceback, then the worker exits and `supervise.py` starts a clean process (backoff, crash-loop limit) |
| Reproducibility | Immutable versioned `configs/<name>/v<n>.json`, referenced by jobs as `name@n` |
| Files | Streaming upload/download with size limits, atomic writes, path-traversal protection; declared outputs can only be written by the lease holder |
| Code delivery | `make_bundle.py` packs the current code; notebooks download it from the queue (no git, no uploads) |
| Observability | Worker status (`starting/ready/busy/error/stopped/gone`), job progress, per-job metrics (run time, peak RAM, peak VRAM), session reports with RAM/VRAM/utilisation sampled every 2 s |

## Quick start (everything on the laptop)

Needs Python 3.9+. This runs the full protocol locally with the example `echo` handler, no GPU or tunnel needed:

```bash
export QUEUE_TOKEN=$(openssl rand -hex 32)
python3 laptop/queue_server.py &                       # http://127.0.0.1:8787, state in data/

echo hello > /tmp/in.txt
python3 laptop/submit.py --kind demo --family echo --params '{"text": "hi", "seconds": 3}' --file in.txt=/tmp/in.txt

QUEUE_URL=http://127.0.0.1:8787 python3 worker/runtime.py --handler echo=handlers.echo:Echo --once

python3 laptop/submit.py --list
python3 laptop/submit.py --get <job id> result.json
```

## Running on a notebook GPU

1. **Start the queue and tunnel** on the laptop. This writes the token to `run/token` and the public address
   to `run/url`:
   ```bash
   ./laptop/start.sh          # needs cloudflared, openssl, curl
   ```
2. **Build the bundle** (the code the notebook downloads). Add your own handler package with `--include`:
   ```bash
   python3 laptop/make_bundle.py --include ../my-model/handlers=worker/my_handlers
   ```
3. **In the notebook**: put the token in the provider's secrets as `QUEUE_TOKEN` (never in the cell), paste
   `tools/notebook/notebook_cell.py` into one cell, set `QUEUE_URL` and `SESSION_ARGS`, run it.
   It starts one worker per GPU and uploads a session report as it goes.
4. **Submit jobs** from the laptop with `submit.py` (or the `Queue` client from your own code), watch with
   `submit.py --list` and `--workers`, and summarise the session with `python3 tools/notebook/report.py`.

Step-by-step Kaggle and Colab instructions, the CLI-driven Kaggle runner and provider gotchas are in
[docs/notebooks.md](docs/notebooks.md).

## Writing a handler

```python
# worker/my_handlers/llm.py
from runtime import Handler

class Generate(Handler):
    def load(self):                       # once, when this family becomes resident on the GPU
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=self.options.get("model", "gpt2"), device=self.device)

    def info(self):
        return {"model": self.options.get("model", "gpt2")}

    def run(self, job, ctx):              # once per job
        prompt = ctx.params["prompt"]
        ctx.progress(phase="generate")    # also stops here if the queue took the job away
        text = self.pipe(prompt, max_new_tokens=ctx.params.get("max_new_tokens", 64))[0]["generated_text"]
        ctx.output("out.txt", text)
        return {"chars": len(text)}       # stored as the job's metrics

    def close(self):                      # before another family is loaded
        del self.pipe
```

Start workers with `--handler llm=my_handlers.llm:Generate --opt model=gpt2`, then submit jobs with
`--family llm`. Inside `run`, `ctx` gives you `params`, `input(name)`, `output(name, data)`, `config()`,
`progress(**fields)` and `check()`. Full reference: [docs/architecture.md](docs/architecture.md#handlers).
Two examples ship with the repo: `worker/handlers/echo.py` (no dependencies) and
`worker/handlers/torch_matmul.py` (a GPU benchmark reporting TFLOPS).

## Layout

| Path | What it is |
|---|---|
| `laptop/queue_server.py` | The queue: HTTP API, SQLite state, file store, lease and heartbeat logic |
| `laptop/start.sh` | Starts the queue and a Cloudflare quick tunnel, prints `QUEUE_URL` |
| `laptop/submit.py` | CLI: submit jobs with files, list jobs and workers, download results |
| `laptop/make_bundle.py` | Packs the code notebooks download (`GET /bundle`) |
| `worker/client.py` | Queue client for workers, the CLI and your own code (stdlib only) |
| `worker/runtime.py` | Long-lived worker: claim loop, heartbeats, handler loading and swapping, metrics, exit codes |
| `worker/supervise.py` | One worker per GPU, restart on crash |
| `worker/handlers/` | Example handlers |
| `worker/fake_worker.py` | Bare protocol test worker (no runtime) |
| `tools/notebook/` | Notebook cell, session driver, report summariser, Kaggle CLI runner |
| `tools/public_tunnel_acceptance.sh` | End-to-end test through a real Cloudflare tunnel |
| `configs/` | Immutable versioned job configs |
| `tests/` | Queue, protocol-hardening and runtime tests |
| `docs/` | Architecture and protocol reference, notebook guide |

## Tests

```bash
python3 tests/test_queue.py            # queue behaviour over real HTTP
python3 tests/test_queue_hardening.py  # leases, retries, limits, timeouts, traversal
python3 tests/test_runtime.py          # worker runtime + supervisor (CPU only)
bash tools/public_tunnel_acceptance.sh # through a real Cloudflare quick tunnel (needs cloudflared + internet)
```

## Limits

- **One shared token.** Laptop and workers use the same token, so any token holder can read every job. Keep it
  in the provider's secrets, rotate it by deleting `run/token`, and don't share notebooks that contain it.
- **One laptop.** The queue is a single process with SQLite; it's built for one person's GPU sessions, not a
  multi-tenant cluster.
- **Quick-tunnel URLs change** on every `start.sh`; update `QUEUE_URL` in the notebook cell (the Kaggle
  runner does this automatically).
- `torch_matmul.py` and multi-GPU sessions have not been run in this repository's tests; the CPU paths have.
