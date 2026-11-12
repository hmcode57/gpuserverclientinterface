# Architecture and protocol reference

## 1. Boundaries

```text
Laptop (persistent)                                  GPU notebook session (temporary, outbound only)
-------------------                                  ----------------------------------------------
producers (submit.py, your app, a UI)                 supervise.py
        |                                               ├── runtime.py  CUDA_VISIBLE_DEVICES=0  handler
        v                                               └── runtime.py  CUDA_VISIBLE_DEVICES=1  handler
queue_server.py  127.0.0.1:8787  <---- HTTPS quick tunnel (cloudflared) ----  workers pull, push results
  data/queue.db   jobs, workers, leases
  data/files/<job>/<name>   inputs and outputs
  data/reports/<session>/   notebook session reports
  data/bundle.tar.gz        code for notebooks
configs/<name>/v<n>.json    immutable job settings
```

- The laptop owns all durable state. Workers own only temporary GPU state.
- The tunnel exposes only the authenticated queue API. It is not a control channel into the notebook.
- A worker always pulls work and pushes results; nothing ever connects into a notebook.
- Producers (CLI, scripts, a future UI) are just queue clients. They never talk to workers and never hold
  model state, so a UI can be added later without touching worker code.

## 2. Job model

| Field | Set by | Meaning |
|---|---|---|
| `id` | queue | Time-ordered unique id (`<ns timestamp>-<random>`) |
| `kind` | producer | What to do, free-form (`generate`, `embed`, `bench`) |
| `family` | producer | Which resident model/runtime can do it; routes the job to workers that accept it |
| `params` | producer | Free-form JSON object, interpreted only by the handler (1 MiB request limit) |
| `config` | producer | Optional `name@version` of an immutable config file |
| `priority` | producer | Integer, higher first (default 0) |
| `after` | producer | Optional job id that must be `done` first |
| `outputs` | producer, worker | File names only the lease holder may write; anything a worker uploads is added |
| `files` | both | `{name: url}` of inputs and outputs |
| `state` | queue | `queued` → `running` → `done` or `failed` (a failed run goes back to `queued` until attempts run out) |
| `attempts`, `last_error` | queue | Failed runs so far (max `QUEUE_MAX_ATTEMPTS`, default 2) and the latest traceback |
| `worker`, `heartbeat`, `progress` | worker | Who runs it, last sign of life, latest progress JSON |
| `score`, `metrics` | worker | Set on completion |
| `waiting_for` | queue (read only) | Why a queued job isn't running: dependency pending or failed, or no live worker accepts its family |

**Claim order** for a worker: jobs of the family it has loaded first (avoids model swaps), then higher
`priority`, then oldest. Only jobs whose `after` dependency is `done` and whose family the worker accepts are
eligible. A job whose dependency failed is reported as blocked, and new jobs can't depend on a failed job.

**Configs** are versioned and never edited: a change is a new file (`v2.json`). A retried or repeated job
therefore always runs with exactly the settings it was submitted with.

## 3. Lease protocol (protocol 2)

1. `POST /claim` atomically picks a job, sets it `running` and returns it with a fresh **lease**.
2. Every write for that run carries the lease: heartbeat, output upload (`X-Lease` header), complete, fail.
3. If heartbeats stop for `QUEUE_HEARTBEAT_TIMEOUT` (170 s; the reaper runs every 10 s, so at most 180 s),
   the run ends as one failed attempt and the lease is dead. Any later call with it gets **409 `lease_lost`**,
   even from a restarted process with the same worker name. The client raises `LeaseLost`, and the runtime
   abandons the job without reporting anything.
4. **Retry safety.** A claim retried with the same `request_id` returns the same job and lease (the client
   keeps the id until it sees an answer). Complete and fail retried with the same lease return the stored
   result instead of an error. Heartbeats aren't retried; the next one is the retry.
5. A worker that claimed but never sent a heartbeat or upload is reported as such in `last_error` (the
   worker died right after claiming, or the claim answer was lost and never retried).
6. Workers older than protocol 2 get **426** on `/claim`.

Inputs can be uploaded by any token holder while the job is `queued`. Once it is running, only the lease
holder can upload. Names in `outputs` can never be written without the lease.

## 4. HTTP API

All requests need `Authorization: Bearer <token>`. Unauthenticated bodies are never read.

Two tokens: `QUEUE_TOKEN` (producer) may call everything. The optional `QUEUE_WORKER_TOKEN` may only call
`GET /health`, `GET /bundle`, `GET /configs/<ref>`, `POST /workers`, `PUT /reports/...`, `POST /claim`,
`POST /jobs/<id>/heartbeat|complete|fail` (lease required, as always), and `GET`/`PUT /jobs/<id>/files/<name>`
with the job's active lease in `X-Lease`. Anything else answers 403 `{"code": "forbidden"}`. See
[multi-session.md](multi-session.md).

| Method and path | Body / result |
|---|---|
| `GET /health` | `{ok, protocol, max_file_bytes}` |
| `POST /jobs` | `{kind, family, params?, config?, priority?, after?, outputs?}` → job (201) |
| `GET /jobs` | All jobs with `waiting_for` |
| `GET /jobs/<id>` | One job |
| `PUT /jobs/<id>/files/<name>` | Raw bytes, `Content-Length` required, `X-Lease` for outputs → job |
| `GET /jobs/<id>/files/<name>` | Streams the file |
| `POST /claim` | `{worker, family (loaded), accept (list or null), request_id, protocol}` → job with `lease`, or 204 |
| `POST /jobs/<id>/heartbeat` | `{worker, lease, progress}` |
| `POST /jobs/<id>/complete` | `{worker, lease, score, metrics}` → job |
| `POST /jobs/<id>/fail` | `{worker, lease, error}` → job |
| `GET /workers` | Workers with `state` (`gone` after the heartbeat timeout), `family`, `accept`, `info`, `job`, `age_s` |
| `POST /workers` | `{worker, state, family, info, job, protocol}` (`state`: starting, ready, busy, error, stopped) |
| `GET /configs/<name>@<v>` | The config JSON |
| `GET /bundle` | `data/bundle.tar.gz` |
| `PUT /reports/<session>/<file>` | Raw bytes, stored in `data/reports/<session>/` (200 MB limit) |

Errors are JSON `{error, code?}`: 400 invalid input, 401 token, 404 unknown, 409 conflict (`lease_lost`,
`not_queued`, `output_file`), 411 no length, 413 too large, 426 outdated client.

**Limits and robustness.** Uploads up to `QUEUE_MAX_FILE_MB` (512 MiB default) and JSON up to 1 MiB are
refused with 413 *before* the body is read. Stalled or idle sockets time out after `QUEUE_READ_TIMEOUT` (60 s).
Files are streamed to a unique temp file, fsynced and renamed, so concurrent uploads never corrupt each other
and a crash never leaves a half-written file under the real name. Job ids and file names are validated, so
paths can't escape `data/files/`.

## 5. Worker runtime

```text
start -> status starting -> load first handler -> warmup -> status ready
ready -> claim (preferring the loaded family) -> status busy -> heartbeat thread starts
busy  -> load/swap handler if needed -> handler.run(job, ctx) -> complete(metrics) -> clear caches -> ready
lease lost   -> abandon silently, stay up
job raises   -> fail(traceback) -> status error -> exit 1 (supervisor restarts a clean process)
bad token, outdated client, bad handler spec -> exit 2 (supervisor does not restart)
```

A status thread reports the worker's state every 30 s, so `GET /workers` shows idle workers as alive. Network
errors while claiming keep the loaded model and retry; only 400/401/426 stop the worker.

### Handlers

```python
from runtime import Handler

class MyHandler(Handler):
    def load(self): ...              # once per residency: load weights (self.device, self.options)
    def warmup(self): return {}      # optional: first run, so its cost isn't charged to a job
    def info(self): return {}        # static facts for the worker status
    def run(self, job, ctx): ...     # per job: return a metrics dict ("score" key -> job score)
    def close(self): ...             # before another family loads
```

`ctx` (JobContext):

| Member | Use |
|---|---|
| `ctx.params` | The job's params dict |
| `ctx.input(name)` | Download an input file (bytes) |
| `ctx.output(name, data)` | Upload a result file (bytes or str) |
| `ctx.config()` | The job's config dict, or None |
| `ctx.progress(**fields)` | Set progress for the next heartbeat; raises `LeaseLost` if the job was taken away |
| `ctx.check()` | Only the lease check. Call it inside long loops (each denoising step, each batch) |

The runtime adds `run_s`, `peak_rss_gb`, `worker` and (on CUDA) `peak_vram_gb` to the returned metrics.

### GPU residency rules

These came out of running a 1.3B-parameter video model on Kaggle T4s and apply to any large model:

- One worker process per GPU, selected with `CUDA_VISIBLE_DEVICES`. Never share a GPU between workers.
- One model family resident per worker. Swapping is allowed but costs a full load, which is why the queue
  prefers jobs for the loaded family.
- Clear caches (`gc.collect()`, `torch.cuda.empty_cache()`) **between jobs**, never between steps: the
  allocator is fastest when it can reuse blocks within a job. The runtime does this for you.
- Session RAM is shared by all workers on the machine. Two processes that each load a large text encoder
  can exceed a notebook's ~29–31 GB even when each fits alone. Measure with a session's RAM peaks before
  running two workers, or keep big CPU-side components in one process.
- After an exception, exit instead of continuing: a GPU after an OOM or CUDA error is in an unknown state.
- On T4 (sm_75) use float16 (no fast bfloat16) and PyTorch SDPA (no FlashAttention 2).

## 6. Load splitting across GPUs

`supervise.py --gpus auto` starts one `runtime.py` per GPU it finds (`nvidia-smi -L`), named
`<name>-gpu<i>`. There is no central scheduler: each worker claims the next eligible job when it is free, so
work spreads across GPUs by itself, and a GPU that dies only loses its in-flight job (which is requeued).
Different GPUs or machines can accept different families (`--handler` lists), and the queue routes jobs
accordingly. Workers from several notebooks (Kaggle + Colab at once) can serve the same queue.

The supervisor restarts a worker that exits with code 1 after 5 s, doubling up to 60 s, and gives up on a GPU
after `--max-restarts` (5) crashes within 10 minutes. Exit code 2 is never restarted.

## 7. Notebook sessions

`tools/notebook/session.py` runs inside the notebook as phases: `preflight` (hardware, drivers, torch,
disk, cgroup limits, internet, queue round trip), `install` (pip with your requirements, refusing to swap
the provider's CUDA torch silently), `workers` (the supervisor), and any custom phases (`--cmd
name="command"`, run once per GPU in parallel). A sampler records RAM and per-GPU VRAM, utilisation,
temperature, power and clocks every 2 s. After every phase `session.json` and the samples are uploaded to
`data/reports/<session>/`, so a notebook that dies mid-run still leaves its data on the laptop.
`--budget-minutes` stops before the provider's time limit while keeping time to upload.

## 8. Environment variables

| Variable | Where | Default |
|---|---|---|
| `QUEUE_TOKEN` | everywhere | required, at least 16 characters |
| `QUEUE_WORKER_TOKEN` | queue | unset (no worker token); `start.sh` sets it from `run/worker_token` |
| `QUEUE_URL` | workers, clients | `http://127.0.0.1:8787` for `submit.py` |
| `QUEUE_URL_SOURCE` | workers, clients | unset; file path or URL to re-read the queue URL from after transport errors |
| `QUEUE_URL_RESOLVE_EVERY` | workers, clients | 60 s (lookup cache per machine) |
| `TUNNEL_NAME`, `TUNNEL_HOSTNAME` | `start.sh` | unset (quick tunnel) |
| `QUEUE_URL_GIST` | `start.sh` | unset; gist id to publish the URL to |
| `QUEUE_HEARTBEAT_TIMEOUT` | queue | 170 s |
| `QUEUE_MAX_ATTEMPTS` | queue | 2 |
| `QUEUE_MAX_FILE_MB` | queue | 512 |
| `QUEUE_READ_TIMEOUT` | queue | 60 s |
| `QUEUE_CONFIGS` | queue | `configs/` |
| `QUEUE_QUIET` | queue | unset (`1` silences request logging) |
