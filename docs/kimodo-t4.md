# kimodo text encoder on Kaggle T4 x2

Encodes kimodo prompts (the LLM2Vec / Llama-3 text encoder, Q8_0) into the 4096-float embeddings that kimodo's
motion models take (`kimodo_generate_embedding`, `kmd-sample-embedding`). Each T4 runs one worker with
**one fully resident Q8_0 encoder** (about 9 GB of its 15 GB). Two Q8_0 encoders on one T4 don't fit (about
17.8 GB). Streaming the layers would fit, but at about 1 s per prompt instead of about 50 ms, so it's slower
than one resident encoder per GPU.

The workers keep the encoder loaded and keep asking the queue for work. **You can submit prompts at any time
while the session runs**, and they're picked up within a second or two, with no reload.

```text
laptop                                             Kaggle T4 x2
kimodo_prompts.py --watch  ──► queue_server ◄──── worker gpu0 ── kmd-encode-serve (CUDA0, Q8_0 resident)
  (stdin, folder, files)       (jobs, results)  └─ worker gpu1 ── kmd-encode-serve (CUDA0 = 2nd T4)
```

## What's in the branch

| File | Role |
|---|---|
| [hmcode57/kimodo.cpp](https://github.com/hmcode57/kimodo.cpp/tree/kaggle-cuda) `kaggle-cuda` | Fork of kimodo.cpp (upstream has only Vulkan and CPU) with a GGML CUDA backend and `kmd-encode-serve`, a process that loads the encoder once and answers prompts over stdin/stdout. |
| `worker/kimodo/setup.sh` | Session phase: clones the fork at `f853bcc`, builds for sm_75 with nvcc, downloads and verifies the Q8_0 weights in parallel with the build. Runs once per notebook even though custom phases start once per GPU (a file lock). |
| `worker/handlers/kimodo_encode.py` | Handler `kimodo`: prompts → `embeddings.f32` (`[n, 4096]` float32) + `result.json`. Refuses to run if the encoder came up on CPU instead of CUDA. |
| `laptop/kimodo_prompts.py` | Submits prompts (arguments, a file, stdin continuously, or a watched folder) and downloads the embeddings into `data/kimodo_embeddings/`. |

## Running it

The full step-by-step protocol (one-time setup, web or terminal start, all ways to send prompts,
shutdown, troubleshooting) is in [kimodo-t4-setup.md](kimodo-t4-setup.md). The short version:

1. Laptop:
   ```bash
   ./laptop/start.sh
   python3 laptop/make_bundle.py
   ```
2. Kaggle notebook: **GPU T4 x2**, Internet on, `QUEUE_TOKEN` secret as in [notebooks.md](notebooks.md).
   Paste `tools/notebook/notebook_cell.py` and set:
   ```python
   SESSION_ARGS = ('--session kaggleA --phases preflight,kimodo,workers '
                   '--cmd "kimodo=bash worker/kimodo/setup.sh" '
                   '--worker-args "--handler kimodo=handlers.kimodo_encode:KimodoEncode" '
                   '--budget-minutes 540')
   ```
   There's deliberately no `--worker-idle-exit`: the workers wait for new prompts until the budget ends. Add
   `--worker-idle-exit 1800` if you'd rather have the session stop after 30 idle minutes.
   From the command line instead: `python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode run
   --session-args '<the same string>'`.
3. Wait until `python3 laptop/submit.py --workers` shows `kaggleA-gpu0` and `kaggleA-gpu1` as `ready` with
   `"backend": "CUDA0"`. The first session spends about 10 minutes on the build plus the 8 GB download.
4. Send prompts, at any time and as often as you like:
   ```bash
   python3 laptop/kimodo_prompts.py "a person walks forward and waves" --wait     # one-off
   python3 laptop/kimodo_prompts.py --file prompts.txt --batch 32 --wait         # a file
   tail -f prompts.txt | python3 laptop/kimodo_prompts.py --watch                # every new line
   python3 laptop/kimodo_prompts.py --watch-dir inbox/                           # every *.txt dropped in
   ```
   Results: `data/kimodo_embeddings/<job>.f32` and `<job>.json` (prompts in row order).
   `np.fromfile(f, "<f4").reshape(-1, 4096)`.

Batch prompts (`--batch`, default 32): each job costs a few tunnel round trips, which take longer than the
~50 ms of GPU work per prompt. `--watch` flushes a partial batch after 2 s of quiet.

## Making later sessions start faster

`setup.sh` uses these if they're attached as Kaggle datasets (found anywhere under `/kaggle/input`):

- **Weights**: a dataset with `Llama-3-Kimodo-Q8_0.gguf` and `tokenizer.gguf`, so there's no 8 GB download.
- **Binary**: the `kimodo-cuda-bin.tar.gz` that the first build writes to `/kaggle/working/kimodo/`
  (download it from the notebook's output), so there's no ~10 minute CUDA build.

## Status

Tested on the laptop: the patched kimodo builds (CPU configuration), `kmd-encode-serve` output is
bit-identical to `kmd-encode`, and the handler works end to end through the queue on CPU, including prompts
submitted while the worker is running. **Not yet run on a Kaggle T4**: the CUDA build (nvcc and compiler
versions on the current image) and the VRAM and speed numbers above are still to be confirmed. `setup.sh` prints
the compiler and nvcc versions it uses and keeps `cmake.log` / `build.log` in `/kaggle/working/kimodo/`.
