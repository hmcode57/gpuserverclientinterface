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

## What this framework is useful for

It pays off when **the GPU is temporary and unreliable, the work splits into independent jobs that take
seconds to tens of minutes each, and you want the results and state to stay on your own machine**.

### Good fit

| Task | Why it fits |
|---|---|
| **Batch generative media**: image or video generation, upscaling, style transfer, TTS, voice conversion | Jobs are independent with files in and out. Models are expensive to load, so they stay loaded between jobs. Long sessions die, and lost jobs are requeued. |
| **Offline batch inference**: embedding a corpus, captioning or tagging a dataset, transcribing audio, batch LLM summarization or classification | Thousands of independent jobs spread across several free GPUs and several notebooks at once. Priority and family routing let different models share one queue. |
| **Benchmarks and hardware checks**: comparing providers or GPUs, testing memory limits | The session driver records hardware, RAM and VRAM peaks, and the timing of each phase, and its reports survive the session dying. `torch_matmul` is a starting point. |
| **Hyperparameter sweeps and small experiments**: many short training or fine-tuning runs of up to about an hour | One job per configuration. Configs are immutable, so runs are reproducible. A crashed run is retried once, then marked failed. |
| **Multi-step pipelines**, such as preprocess → generate → postprocess | `after` dependencies, plus different families on different GPUs |
| **Several free providers at once**: Kaggle, Colab and a rented box together | Every worker pulls from the same queue. There's no central scheduler to configure. |

### Poor fit

| Task | Why not | Use instead |
|---|---|---|
| **Live or interactive serving**: chat UIs, APIs with response-time limits | The tunnel and polling add seconds of delay, and the notebook can disappear at any time | A dedicated inference host or serverless GPUs |
| **Long training runs with large checkpoints** (hours to days, multi-GB weights and optimizer state) | A job can resume after a session dies only if its handler saves checkpoints itself, and each checkpoint goes through your home connection (512 MB cap by default, held in memory). Session cuts and real crashes share one attempt limit. Small checkpoints such as LoRA adapters work well; see "Resuming long jobs" below. | The provider's own storage with framework-native resume (Hugging Face Trainer, Lightning), or a paid host |
| **Multi-GPU training of one model** (data or tensor parallel) | Each worker runs alone on one GPU, and workers don't talk to each other | torchrun, DeepSpeed or FSDP on one machine |
| **Gigabytes of data per job** | Every file goes through your home connection, and uploads are capped at 512 MB by default | Cloud storage, or datasets the provider hosts |
| **Multi-user or sensitive workloads** | Only two coarse tokens: anyone with the producer token can read every job, and worker-token holders see the inputs of jobs they run. Job data passes through Cloudflare and the provider, and the laptop is a single point of failure. | Celery, Redis or Ray with proper authentication |
| **Very small jobs** (milliseconds each) | Several HTTP round trips through the tunnel per job, so the overhead is bigger than the work | Batch them into bigger jobs inside your handler |

### Compared with other tools

- **Ray, Celery and Dask** are more powerful, but they expect a cluster they can reach. They don't cope well with workers that sit behind notebook firewalls and disappear.
- **Modal and RunPod serverless** are smoother, but you pay for them.
- **This framework's niche** is free or cheap GPUs you don't control, coordinated from your laptop, with nothing to install on the laptop.

## Gaps worth closing

In order of value:

1. **Built-in checkpoint support.** Resuming already works by hand (see "Resuming long jobs" below), but
   `ctx.checkpoint()` and `ctx.resume()` helpers would make it one line each. Large checkpoints would also
   need streaming uploads (not held in memory), and session cuts would need to count separately from real
   crashes, so retrying expected cuts many times doesn't also retry genuine bugs many times.
2. **Batch submission and bulk download.** A `submit.py --batch jobs.jsonl` command, and a way to fetch all
   results at once, for runs with thousands of jobs.
3. **Cleanup of finished jobs.** `data/files/` currently grows forever. It needs a retention setting or a
   prune command.
4. **A real GPU run.** The multi-GPU path and the `torch_matmul` handler haven't been run on GPUs yet. One
   Kaggle T4 x2 session would confirm them.

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

### Several notebooks at once

Run the same cell in as many notebooks as you like (several Kaggle T4 x2 sessions, Colab, teammates'
accounts); every worker pulls from the same queue. Give each notebook its own `--session`, give notebooks you
don't fully trust the restricted `run/worker_token`, and use a named tunnel or `QUEUE_URL_GIST` so a laptop
restart doesn't strand them. Details in [docs/multi-session.md](docs/multi-session.md).

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

## Resuming long jobs

Files a worker uploads stay attached to the job when it's requeued, so a handler can resume after a
notebook session dies:

```python
def run(self, job, ctx):
    state = self.fresh_state()
    if "ckpt.pt" in job["files"]:                 # an earlier run of this job saved one
        state = self.restore(ctx.input("ckpt.pt"))
    for step in range(state.step, ctx.params["steps"]):
        self.train_step(state)
        ctx.check()                               # stop if the queue gave the job to someone else
        if step % 500 == 0:
            ctx.output("ckpt.pt", self.save(state))   # model, optimizer, scheduler, RNG, data position
    ctx.output("model.safetensors", self.export(state))
```

Raise the attempt limit for this (`QUEUE_MAX_ATTEMPTS=20` on the laptop), because every session cut counts
as one attempt. Keep checkpoints under `QUEUE_MAX_FILE_MB` (512 MB by default). Each checkpoint goes through
your connection, and the laptop has to stay on.

## Layout

| Path | What it is |
|---|---|
| `laptop/queue_server.py` | The queue: HTTP API, SQLite state, file store, lease and heartbeat logic |
| `laptop/start.sh` | Starts the queue and a Cloudflare tunnel (quick or named), creates both tokens, prints `QUEUE_URL`, optionally publishes it to a gist |
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
| `docs/` | Architecture and protocol reference, notebook guide, multi-session guide |

## Tests

```bash
python3 tests/test_queue.py            # queue behaviour over real HTTP
python3 tests/test_queue_hardening.py  # leases, retries, limits, timeouts, traversal
python3 tests/test_runtime.py          # worker runtime + supervisor (CPU only)
python3 tests/test_multi_session.py    # worker token, URL discovery, unique session ids
bash tools/public_tunnel_acceptance.sh # through a real Cloudflare quick tunnel (needs cloudflared + internet)
```

## Limits

- **Two tokens, coarse permissions.** `run/token` can do everything. `run/worker_token` can't list, read or
  submit jobs, but it can claim any job and read that job's inputs. Keep both in the provider's secrets,
  rotate them by deleting the files, and don't share notebooks that contain them.
- **One laptop.** The queue is a single process with SQLite; it's built for one person's GPU sessions, not a
  multi-tenant cluster.
- **Quick-tunnel URLs change** on every `start.sh`. Use a named tunnel (`TUNNEL_NAME`/`TUNNEL_HOSTNAME`) or
  publish the URL to a gist (`QUEUE_URL_GIST` + `QUEUE_URL_SOURCE`) so running notebooks follow it; otherwise
  update `QUEUE_URL` in the cell (the Kaggle runner does this automatically).
