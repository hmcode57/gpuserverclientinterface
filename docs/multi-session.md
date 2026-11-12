# Several notebooks at once

Any number of notebook sessions can work on one queue at the same time: several Kaggle T4 x2 sessions,
Colab, a rented box, and sessions on other people's accounts. Every worker pulls the next job it can run, so
there's nothing to configure centrally. Three T4 x2 sessions are six workers on one queue.

This scales **throughput of independent jobs**. It doesn't make one job faster, and it can't train one
model across sessions (they can't talk to each other; see the README's "Poor fit" table).

## Checklist

1. **One `--session` per notebook.** It prefixes the worker names (`kaggleA-gpu0`, `kaggleA-gpu1`, ...), so
   `submit.py --workers` and the reports in `data/reports/<session>/` keep sessions apart. Without it the
   session id is `nb-<timestamp>-<random>`, which is also unique, just harder to read.
   ```python
   SESSION_ARGS = '--session kaggleA --phases preflight,workers --worker-args "--handler ..." --budget-minutes 600'
   ```
2. **The right token per notebook** (see below).
3. **A stable or discoverable URL**, so a laptop restart doesn't strand the notebooks (see below).
4. **Size your home connection.** All files go through the laptop. Many workers at once only pay off when
   jobs spend much longer on the GPU than on their uploads and downloads: prompts in and text, embeddings or
   small images out are fine; video in and out isn't.

## Tokens: producer and worker

`laptop/start.sh` creates two tokens:

| File | Can | Give it to |
|---|---|---|
| `run/token` | everything: submit, list and read jobs, download any file | your laptop tools and notebooks on your own account |
| `run/worker_token` | claim jobs, read the inputs of the job it holds a lease for, upload outputs, heartbeat/complete/fail, report status and session reports, fetch the bundle and configs | notebooks on accounts you don't fully trust, e.g. friends contributing their GPU quota |

A notebook uses whichever token is in its `QUEUE_TOKEN` secret; the cell is the same. A worker-token holder
can still see the inputs of jobs it claims and could claim jobs just to read them, so don't queue anything
you wouldn't hand to the people running those notebooks. It can't list the queue, read jobs by id, submit,
or download files of jobs it isn't running.

Rotate either token by deleting its file and running `start.sh` again; then update the secrets.

## Accounts

The framework doesn't care whose account a worker runs on. **Kaggle allows one account per person**, so
pooling means other people running sessions on their own accounts, not you opening several accounts for
more quota (that breaks Kaggle's terms and can get every account banned).

For a teammate's notebook: send them the notebook cell and the **worker token** through a private channel.
They put the token in their own Kaggle secret `QUEUE_TOKEN`, set a distinct `--session`, and run the cell.

## Keeping the URL working

Quick-tunnel addresses change on every `start.sh`. Two fixes; either lets running notebooks survive a laptop
restart.

### Named tunnel (fixed address)

Needs a free Cloudflare account and a domain on it. One-time setup:

```bash
cloudflared tunnel login
cloudflared tunnel create gpuq
cloudflared tunnel route dns gpuq queue.example.com
```

Then always start with:

```bash
TUNNEL_NAME=gpuq TUNNEL_HOSTNAME=queue.example.com ./laptop/start.sh
```

`QUEUE_URL = "https://queue.example.com"` in the cell never changes again.

### URL published to a gist (no domain needed)

Create a secret gist once with a file named `queue_url` (any content), and note its id:

```bash
echo placeholder | gh gist create -f queue_url -    # prints https://gist.github.com/<user>/<id>
QUEUE_URL_GIST=<id> ./laptop/start.sh               # writes the current URL into the gist
```

In each notebook cell set:

```python
QUEUE_URL_SOURCE = "https://api.github.com/gists/<id>"
```

The cell reads the URL from the gist at start, and every worker re-reads it when the queue stops answering,
then switches over (logged as `[client] queue moved: old -> new`). Lookups are cached for 60 s per machine
(`QUEUE_URL_RESOLVE_EVERY`) to stay inside GitHub's anonymous limit of 60 requests an hour. The gist only
holds the URL; the queue still needs a token. But workers send their token to whatever URL the source
names, so use a gist or file only you can edit. Resolved URLs must be `https` (plain `http` only for
localhost).

`QUEUE_URL_SOURCE` also accepts any https URL that returns the address as plain text, or a file path.

The queue keeps its state across restarts. Jobs whose heartbeats didn't get through for 170 s during the
switch are requeued and run again; idle workers just reconnect with their models still loaded.

### Kaggle CLI runner

`tools/notebook/kaggle_run.py` fills in `QUEUE_URL` from `run/url` and `QUEUE_URL_SOURCE` from
`run/url_source` (written by `start.sh` when `QUEUE_URL_GIST` is set). `--worker-token` makes
`--make-token-dataset` package `run/worker_token` instead of `run/token`.

```bash
python3 tools/notebook/kaggle_run.py --kernel <user>/gpuq-a --mode run \
    --session-args '--session kaggleA --phases preflight,workers --worker-args "--handler ..."'
python3 tools/notebook/kaggle_run.py --kernel <user>/gpuq-b --mode run \
    --session-args '--session kaggleB --phases preflight,workers --worker-args "--handler ..."'
```

## Watching

```bash
python3 laptop/submit.py --workers     # every worker of every session, with state and current job
python3 laptop/submit.py --list
python3 tools/notebook/report.py data/reports/kaggleA
```
