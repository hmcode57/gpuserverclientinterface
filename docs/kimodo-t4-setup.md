# kimodo encoders on Kaggle T4 x2: setup and use

The steps to go from nothing to a running Kaggle session that turns kimodo prompts into embeddings, with prompts sent
from your laptop's terminal at any time while it runs. Background and design: [kimodo-t4.md](kimodo-t4.md).

Everything below runs **in a terminal on the laptop**, from the repository folder, unless it says *Kaggle*.

```bash
cd ~/gpuserverclientinterface
git checkout kimodo-t4
```

| When | Do |
|---|---|
| Once | Part 1 |
| Every session | Part 2 (laptop), Part 3 (start Kaggle: 3A browser **or** 3B terminal), Part 4 (wait) |
| While it runs | Part 5 (send prompts, get embeddings) |
| At the end | Part 6 |

---

## Part 1: one-time setup

### 1.1 Laptop tools

| Tool | Check | Needed for |
|---|---|---|
| Python 3.9+ | `python3 --version` | queue, submitting |
| `cloudflared` | `cloudflared --version` | the tunnel the notebook connects through |
| `openssl`, `curl` | `openssl version; curl --version` | `start.sh` |
| `numpy` (optional) | `python3 -c "import numpy"` | reading the embeddings |
| Kaggle CLI ≥ 1.8 (optional) | `kaggle --version` | 1.5 (datasets) and 3B (starting runs from the terminal) |

### 1.2 Kaggle account

1. Verify your phone number at kaggle.com → Settings (needed for GPUs and Internet in notebooks).
2. Check you have GPU quota left this week (Settings → *Quotas*). One T4 x2 session uses both GPUs' hours.

### 1.3 Kaggle CLI token and username (for 1.5 and 3B)

Already set up if this prints your username:

```bash
kaggle config view
```

Otherwise: kaggle.com → Settings → *API* → create a token, save it as `~/.kaggle/access_token`, then run:

```bash
chmod 600 ~/.kaggle/access_token
```

The commands below use `$KUSER` for your Kaggle username. Set it **in every new terminal**:

```bash
KUSER=$(kaggle config view | awk '/username:/ {print $3}'); echo "$KUSER"
```

### 1.4 Queue token

Start the queue once (`./laptop/start.sh`, see 2.1); it creates the token in `run/token`. It stays the same
between sessions. Show it with:

```bash
cat run/token
```

Put it where Kaggle can read it:

- **Browser start (3A):** in the Kaggle notebook editor open *Add-ons → Secrets → Add secret*.
  Label: `QUEUE_TOKEN` (exactly), value: the line printed by `cat run/token`. Then switch it **on** for this
  notebook (every new notebook needs it switched on).
- **Terminal start (3B):** secrets don't reach API-pushed runs, so `kaggle_run.py` puts the token in a
  **private** dataset for you (`--make-token-dataset`, step 3B.1).

Never paste the token into the notebook cell itself.

### 1.5 Optional: make later sessions start in ~2.5 minutes instead of ~25

Two private datasets: the CUDA build (skips the ~20 min build, **use this one**) and the weights.
`setup.sh` finds both under `/kaggle/input`. The weights dataset is slower than downloading: loading the 8 GB
from the dataset mount took ~6 min, the Hugging Face download ~2 min plus 7 s to load (measured on T4 x2), so
attach it only as a fallback when Hugging Face is unreachable. Kaggle unpacks the uploaded
`kimodo-cuda-bin.tar.gz`; `setup.sh` uses the unpacked files. `tools/notebook/kimodo_datasets.py` creates them; it reads your
Kaggle username from the Kaggle CLI (1.3), so the commands are the same for everyone.

**Weights, any time** (8 GB upload, once; Kaggle then needs a few minutes to process it):

```bash
python3 tools/notebook/kimodo_datasets.py weights              # uses ~/kimodo/*.gguf, checks the sha256
python3 tools/notebook/kimodo_datasets.py weights --download   # no local copy: fetch from Hugging Face first
python3 tools/notebook/kimodo_datasets.py weights --dir /path  # the .gguf files are somewhere else
```

**CUDA build, after the first session whose `kimodo` phase ended `ok=True`** (Part 4):

```bash
python3 tools/notebook/kimodo_datasets.py bin
```

It takes `kimodo-cuda-bin.tar.gz` from the latest run of `<your user>/kimodo-t4` (`--kernel` for another
notebook). After changing `KIMODO_COMMIT` or the build flags in `setup.sh`, run one session without this
dataset and then `bin` again, or the old binary keeps being used.

**Check what's ready** and get the flags to attach them:

```bash
python3 tools/notebook/kimodo_datasets.py status
```

**Attach them:**
- Terminal start (3B): add the `--dataset ...` flags that `status` prints to the `kaggle_run.py --mode run`
  command. Every push replaces the notebook's inputs, so inputs added in the web editor don't carry over to
  terminal runs.
- Browser start (3A): *Add Input* → your datasets → add `kimodo-cuda-bin`.

The `kimodo` log (`data/reports/kaggleA/kimodo_gpu0.log`) then shows `binary from /kaggle/input/kimodo-cuda-bin (unpacked by Kaggle)`
instead of a build.

---

## Part 2: every session, on the laptop

### 2.1 Start the queue and the tunnel

```bash
./laptop/start.sh
```

It prints the public address and writes it to `run/url`. **The address changes every time you run
`start.sh`**, so the notebook must be (re)started with the current one. Leave the laptop awake and online
during the session. Short internet drops are fine: workers keep the encoder loaded and reconnect.

Check:

```bash
cat run/url                       # https://<random>.trycloudflare.com
python3 laptop/submit.py --list   # empty list, no error = queue is up
```

### 2.2 Build the code bundle (only after code changes)

```bash
python3 laptop/make_bundle.py
```

The notebook downloads this bundle from the queue at start. It stays in `data/bundle.tar.gz` and is reused,
so rebuild it only the first time and after code under `worker/` or `tools/notebook/` changed (e.g. after a
`git pull`). Changes to `laptop/` and docs run on the laptop and don't need a rebuild. The notebook log prints
the bundle's commit (`BUNDLE.json`) at the start.

---

## Part 3: start the Kaggle session

Do **either** 3A (browser) **or** 3B (terminal), not both. Two sessions with the same `--session` name
(`kaggleA`) at once clash; a second notebook needs its own name (`kaggleB`).

### Part 3A: from the browser

1. Create a new notebook on kaggle.com. Settings: **Accelerator: GPU T4 x2**, **Internet: on**.
2. Make sure the `QUEUE_TOKEN` secret is attached (1.4).
3. Open `tools/notebook/notebook_cell.py` on the laptop, copy all of it and paste it into one cell of the
   *Kaggle notebook*. (`SESSION_ARGS` is in this file, not in `session.py`; don't edit `session.py`.)
4. **In the Kaggle cell** (this is Python, don't run it in a terminal), change two lines near the top:
   - line 5, `QUEUE_URL = "https://PASTE-THE-ADDRESS-..."`: put the address from `cat run/url` between the quotes:
     ```python
     QUEUE_URL = "https://<random>.trycloudflare.com"
     ```
   - line 11, `SESSION_ARGS = '--phases preflight,workers --worker-args "--handler echo=handlers.echo:Echo" --budget-minutes 20'`
     (the echo test): replace that whole line with:
     ```python
     SESSION_ARGS = ('--session kaggleA --phases preflight,kimodo,workers '
                     '--cmd "kimodo=bash worker/kimodo/setup.sh" '
                     '--worker-args "--handler kimodo=handlers.kimodo_encode:KimodoEncode" '
                     '--budget-minutes 540')
     ```
5. Optional: attach the datasets from 1.5 (*Add Input*).
6. Run the cell and leave the tab open (or use *Save Version → Save & Run All* to run without the tab).

Kaggle keeps the cell with the notebook. Next time only `QUEUE_URL` needs changing, and only if `start.sh`
was restarted (the address changes then).

Continue with Part 4.

### Part 3B: from the terminal

Needs 1.3 (Kaggle CLI and `$KUSER`).

#### 3B.1 First time only: check the connection (CPU, about a minute)

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/kimodo-t4 --mode probe \
    --token-dataset $KUSER/gpuq-token --make-token-dataset
```

`/health 200` and `/bundle 200` in the output mean the notebook can reach your laptop.
`/health 401` means the token is wrong; a connection error means the tunnel is down (rerun 2.1).

Normal in the probe output, ignore it:
- `PROBE secret error: ConnectionError ...`: API-pushed runs get no secrets; the next line shows the token
  was read from the dataset instead.
- debugger warnings, `SyntaxWarning`, `NbConvertApp` lines: Kaggle's own noise.

The probe ends with `status COMPLETE` and starts no workers.

#### 3B.2 Start the GPU session

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/kimodo-t4 --mode run \
    --token-dataset $KUSER/gpuq-token \
    --session-args '--session kaggleA --phases preflight,kimodo,workers --cmd "kimodo=bash worker/kimodo/setup.sh" --worker-args "--handler kimodo=handlers.kimodo_encode:KimodoEncode" --budget-minutes 540'
```

With the datasets from 1.5, add the flags `kimodo_datasets.py status` prints after `--token-dataset ...`,
normally just `--dataset $KUSER/kimodo-cuda-bin` (weights from Hugging Face, see 1.5).

It pushes the notebook (T4 x2, `QUEUE_URL` filled in from `run/url`) as a new version, starts it and follows
its log.

#### 3B.3 Follow it

- `QUEUED`: Kaggle is waiting for a free T4 x2 machine; GPU hours don't count yet. Usually seconds to
  minutes, longer at busy times. Don't push again (that puts you at the back) and don't open the notebook in
  the Kaggle editor with the GPU on: an interactive GPU session takes one of your GPU slots.
- `RUNNING`: the log shows `==== phase preflight`, `==== phase kimodo` (the long one), `==== phase workers`.
  Kaggle's log can lag a minute or two; Part 4 is the quickest check.

`Ctrl-C` only stops following; the session keeps running. Show status and log again later:

```bash
python3 tools/notebook/kaggle_run.py --kernel $KUSER/kimodo-t4 --logs
```

---

## Part 4: wait for the workers

```bash
watch -n 10 python3 laptop/submit.py --workers
```

Ready looks like this (one worker per T4):

```text
kaggleA-gpu0   ready  family=kimodo job=-  seen 3s ago  {"backend": "CUDA0", "text_model": "Llama-3-Kimodo-Q8_0.gguf", ...}
kaggleA-gpu1   ready  family=kimodo job=-  seen 2s ago  {"backend": "CUDA0", ...}
```

`CUDA0` on both is correct: each worker sees only its own GPU. How long it takes:

| Session | Until ready |
|---|---|
| First (build + 8 GB download) | about 24 min (build ~20 min, download overlaps it) |
| Weights dataset only (build still runs) | about 26 min (6 min load from the dataset mount) |
| CUDA build dataset, weights from Hugging Face | about 2.5 min (expected: 2 min download + 7 s load) |

You can submit prompts (Part 5) before the workers are ready; they wait in the queue.

First session that built successfully (`phase kimodo: ok=True`)? Save the build now so the next sessions
skip it: `python3 tools/notebook/kimodo_datasets.py bin` (1.5).

---

## Part 5: use it

All results go to `prompts-for-encoder-files/out/` (change with `--out DIR`), downloaded as soon as each job is done:
`<job>.f32` holds the embeddings, `<job>.json` the prompts in the same row order. By default every prompt is
its own job, so each prompt gets its own `.f32`/`.json` pair. `--batch 32` packs 32 prompts into one job
(one file pair for all 32, fewer tunnel round trips): worth it for files with hundreds of prompts.

### Single prompts

```bash
python3 laptop/kimodo_prompts.py "a person walks forward and waves" "someone jumps twice" --wait
```

### A file (one prompt per line)

```bash
python3 laptop/kimodo_prompts.py --file prompts.txt --wait             # one file pair per prompt
python3 laptop/kimodo_prompts.py --file prompts.txt --batch 32 --wait  # large files: one pair per 32
```

### One file you keep adding to (recommended)

```bash
python3 laptop/kimodo_prompts.py --follow prompts-for-encoder-files/prompts.txt
```

Add lines to the file however you like (editor, `echo "a person waves." >> ...`, another program). About a
second after each save, every line that wasn't sent before goes out as its own job, and each result is saved
as its own labelled pair, numbered in sending order:

```text
prompts-for-encoder-files/out/0001-a-person-jumps-over-a-small-obstacle.f32   # 4096 float32
prompts-for-encoder-files/out/0001-a-person-jumps-over-a-small-obstacle.json  # {"prompt": ..., "n": 1, "job": ...}
```

- Only new lines: the lines already in the file when you first start are skipped (`--include-existing`
  sends them too).
- It remembers what was sent in `out/.follow-<file>.json`: saving from an editor, deleting or moving lines,
  and stopping and restarting never send a prompt twice. Lines added while it was stopped are sent on the next
  start, and results still running are collected then.
- A prompt text is sent once. To send one again, change the line. A failed prompt (e.g. too long) is shown
  as `[failed]`; edit it and save to send the fixed version.
- `--batch 32` packs new lines into fewer jobs (faster for big pastes); the files are still one per prompt.
- Stop with `Ctrl-C`.

### Typed or piped in

Type prompts, one per line; each line is sent right away. (With `--batch N`, lines are collected until N
fill or you stop typing for 2 s.) `Ctrl-D` ends and waits for the last results, `Ctrl-C` stops at once.

```bash
python3 laptop/kimodo_prompts.py --watch
my_prompt_generator | python3 laptop/kimodo_prompts.py --watch --batch 16
```

`--batch` is prompts per job, not a limit: `--watch` keeps reading until you stop it. Results go to
`out/<job>.f32` / `.json` as each job finishes; jobs still running at `Ctrl-C` are listed and can be collected
later with `--fetch`. For a file, use `--follow` above rather than `tail -F file | ... --watch`: when an editor
saves by replacing the file, `tail -F` sends the whole file again.

Or watch a folder: every `*.txt` dropped into it becomes a job and is moved to `inbox/sent/`:

```bash
mkdir -p inbox
python3 laptop/kimodo_prompts.py --watch-dir inbox/
# in another terminal / from another program:
printf 'a person kicks\na person sits down\n' > inbox/batch-001.txt
```

### Submit now, collect later

```bash
python3 laptop/kimodo_prompts.py --file prompts.txt          # prints [queued] <job id> per job
python3 laptop/kimodo_prompts.py --fetch <job id> <job id>   # downloads them when done
```

### Read the embeddings

```python
import json, numpy as np
emb = np.fromfile("prompts-for-encoder-files/out/<job>.f32", "<f4").reshape(-1, 4096)
prompts = json.load(open("prompts-for-encoder-files/out/<job>.json"))["prompts"]   # prompts[i] -> emb[i]
```

Each row is the embedding `kimodo_generate_embedding` takes (and `kmd-sample-embedding` with a raw `.f32`).

### Watch progress

```bash
python3 laptop/submit.py --list      # all jobs: queued / running / done / failed
python3 laptop/submit.py --workers   # what each GPU is doing
```

---

## Part 6: end the session

1. Stop sending prompts and wait until `submit.py --list` shows nothing `queued` or `running`.
2. Stop the notebook: on kaggle.com open the notebook → *Cancel run* (or *Stop session* for an interactive one).
   Otherwise it stops by itself when `--budget-minutes` runs out, and it still uploads its report first.
3. Session summary (timings, RAM/VRAM peaks):
   ```bash
   python3 tools/notebook/report.py
   ```
4. The queue on the laptop can keep running for the next session; stop it with `Ctrl-C` in its terminal or
   by killing `start.sh`'s processes.
5. After using Part 3B: `rm ~/.kaggle/access_token` and revoke the token at kaggle.com → Settings when
   you're done for a while; delete the `gpuq-token` dataset after rotating the queue token.

Jobs that were running when a notebook died go back into the queue automatically (after about 3 minutes) and
are picked up by the next session.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Notebook: `QUEUE_TOKEN was not available` | secret not attached to *this* notebook, or API-pushed run | attach the secret (1.4), or use `--token-dataset` (3B) |
| Notebook: `HTTP Error 530` / connection errors | `start.sh` was restarted, address changed | rerun the notebook with the new `run/url` |
| Phase `kimodo` fails, `nvcc not found` | Kaggle image without CUDA toolkit | check the accelerator is a GPU; report the image version |
| Phase `kimodo` fails: `Target "ggml-cuda" links to: CUDA::cuda_driver but the target was not found` | old bundle without `GGML_CUDA_NO_VMM=ON` | `python3 laptop/make_bundle.py`, start a new session |
| Phase `kimodo` fails during cmake/build | compiler or CUDA version on the current image | on the laptop: `data/reports/kaggleA/kimodo_gpu0.log` / `kimodo_gpu1.log` (end of `cmake.log` / `build.log`); full logs in the notebook output under `kimodo/` |
| `phase kimodo: ok=False`, then workers crash with `kimodo not set up: no .../ready.json` | the build failed; workers have nothing to load and give up after 6 crashes | fix the `kimodo` error above; the worker crashes are only a follow-on |
| Worker crash `binbcast.cu: GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16)` on the first prompt | build of the fork before `e6f6a2d` (BF16 norm weights repeated on CUDA) | `python3 laptop/make_bundle.py`, start a new session without the old `kimodo-cuda-bin` dataset |
| Worker error `came up on CPU, not CUDA` | the build has no CUDA, or no GPU visible | check `nvidia-smi` in the notebook and the build log |
| Worker error `out of memory` | something else holds VRAM on that GPU | the encoder needs about 9 GB; don't run other GPU work in the notebook |
| Job `failed`: `prompt N (...): prompt token count must be in 1..511` | prompt too long | shorten that prompt and resubmit |
| Prompts stay `queued` | no worker ready yet, or the session ended | `submit.py --workers`; start a new session (Part 2–3) |
| `kimodo_prompts.py` connection refused | queue not running on the laptop | `./laptop/start.sh` |
| `bash: kaggle-user: No such file or directory` | a `<kaggle-user>` placeholder from an old copy of a command | use `$KUSER` (1.3) or your username |
| Run stays `QUEUED` for a long time | no free T4 x2, an open interactive GPU session, or GPU quota used up | close the notebook editor; check Settings → *Quotas*; wait or try later |
| Log shows old times / an earlier run | `--logs` shows the latest run; the new one wasn't pushed | check `kaggle kernels status $KUSER/kimodo-t4`, push again with 3B.2 |
