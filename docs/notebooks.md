# Running workers on notebook GPUs (Kaggle, Colab)

## 1. Laptop

```bash
./laptop/start.sh                     # queue + Cloudflare quick tunnel
python3 laptop/make_bundle.py         # code for the notebook; add --include for your handlers
cat run/url                           # -> QUEUE_URL for the notebook cell
cat run/token                         # -> the QUEUE_TOKEN secret. Never paste it into the notebook itself
```

`start.sh` needs `python3`, `openssl`, `curl` and
[`cloudflared`](https://github.com/cloudflare/cloudflared/releases) on your PATH. It creates `run/token` once,
restarts the queue and tunnel, waits until the public address really answers, and writes it to `run/url`.
**The quick-tunnel address changes every time `start.sh` runs.** The token stays the same until you delete
`run/token`.

Keep the laptop awake and online while notebooks are working. If it goes offline, workers keep retrying
their claims with their models loaded; running jobs are requeued if their heartbeats can't get through for
170 s.

## 2. Kaggle

1. Create a notebook. In **Settings**, turn **Internet on** and set **Accelerator** to a GPU (for example
   **GPU T4 x2**).
2. **Add-ons → Secrets**: add `QUEUE_TOKEN` with the full contents of `run/token`, and switch it on for this
   notebook. Creating the secret in your account isn't enough by itself.
3. Check the secret in a cell (prints only the length):
   ```python
   from kaggle_secrets import UserSecretsClient
   token = UserSecretsClient().get_secret("QUEUE_TOKEN")
   print("Secret loaded:", bool(token), "length:", len(token or ""))
   ```
4. Paste `tools/notebook/notebook_cell.py` into a cell. Set `QUEUE_URL` to `run/url` and `SESSION_ARGS` to
   what the session should do, then run it and leave it running.

## 3. Colab

Same cell. Put the token in Colab's **Secrets** (key icon) as `QUEUE_TOKEN` and enable notebook access. Pick
a GPU runtime. The cell finds the Colab secret automatically.

## 4. Session arguments

`SESSION_ARGS` are passed to `tools/notebook/session.py`. Examples:

```python
# test the plumbing: echo workers for 20 minutes
SESSION_ARGS = '--phases preflight,workers --worker-args "--handler echo=handlers.echo:Echo" --budget-minutes 20'

# measure the GPU first, then serve real jobs until the queue has been idle for 10 minutes
SESSION_ARGS = ('--phases preflight,install,workers --requirements worker/my_handlers/requirements.txt '
                '--worker-args "--handler llm=my_handlers.llm:Generate --opt model=gpt2" --worker-idle-exit 600')

# a custom phase, run once per GPU in parallel, before the workers
SESSION_ARGS = ('--phases preflight,smoke,workers --cmd "smoke=python worker/my_handlers/smoke.py" '
                '--worker-args "--handler llm=my_handlers.llm:Generate"')
```

On the laptop, `python3 laptop/submit.py --workers` shows the workers as soon as they start, and
`python3 tools/notebook/report.py` summarises the latest session from `data/reports/<session>/`.

## 5. Driving Kaggle from the command line

`tools/notebook/kaggle_run.py` pushes the cell as a Kaggle notebook version, runs it, and follows its status
and logs, so there is nothing to copy and paste. It needs the Kaggle CLI (1.8 or newer) with an API token
in `~/.kaggle/access_token`.

```bash
python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode probe   # CPU: secret + tunnel + /bundle
python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode run --session-args '--phases preflight,workers'
python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --logs         # status + logs of the latest run
```

It fills in `QUEUE_URL` from `run/url`. Probe results: `secret error: ConnectionError` = no secret reaches the
run; `/health 401` = the tunnel works but the token is missing or wrong; `/health 200` and `/bundle 200` = ready.

**Kaggle secrets don't reach API-pushed runs** (measured 2026-11-09: the secret was attached and switched on,
and `UserSecretsClient` still raised `ConnectionError` in runs started by `kaggle kernels push`). The
workaround is a **private** dataset that holds only a file named `QUEUE_TOKEN`, attached as a data source.
The notebook cell reads `/kaggle/input/**/QUEUE_TOKEN` when no secret is available:

```bash
python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode run \
    --token-dataset <user>/gpuq-token --make-token-dataset
```

Delete that dataset and the API token (`rm ~/.kaggle/access_token`, then revoke it at kaggle.com/settings)
when you're done or after rotating the queue token.

## 6. Provider notes

From Kaggle runs in November 2026. Check them against a fresh `preflight` before relying on them.

- **Disk:** `/kaggle/working` is limited to about 20 GB and is kept between versions. `/tmp` is a larger overlay
  that is lost when the session ends. Put large model downloads in `/tmp` if they don't fit.
- **RAM:** about 29–31 GB for the whole session, shared by all worker processes. `preflight` records the
  cgroup limit, and every phase records its RAM peak.
- **Python:** images move to new Python versions (3.13 in November 2026). Pin package versions that have wheels
  for that version, or the install falls back to slow or failing source builds.
- **pip** prints resolver warnings that begin with `ERROR:` about unrelated preinstalled packages and still
  exits 0. Judge it by the exit code, which `session.py` does.
- **torch:** keep it out of your requirements file and use the provider's CUDA build. `session.py` records the
  torch version before and after the install and warns if pip replaced it.
- **Time limits:** providers cap session length and weekly GPU hours; check their current limits. Use
  `--budget-minutes` so reports are uploaded before the cut.
- **First model download is cold:** about 20 GB of model files can take most of a 20-minute budget. Measure
  it in its own phase before planning a session around it.
