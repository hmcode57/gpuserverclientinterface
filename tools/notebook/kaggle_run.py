"""Push the notebook to Kaggle via the Kaggle CLI, then follow its status and logs (no copy-paste needed).

  python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode probe   # CPU: secret + tunnel + /bundle check
  python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --mode run     # GPU T4 x2: notebook_cell.py
  python3 tools/notebook/kaggle_run.py --kernel <user>/<slug> --logs         # status + logs of the latest run

Needs the Kaggle CLI (>= 1.8) with a token in ~/.kaggle/access_token (temporary, delete when done).
QUEUE_URL is read from run/url (written by laptop/start.sh). The queue token is NEVER put in the notebook: the
Kaggle secret QUEUE_TOKEN must already be attached to this notebook (Add-ons > Secrets, once per notebook in the
web UI; the API cannot attach secrets). Pushing creates a new saved version and runs it ("Save & Run All").
"""
import argparse, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

PROBE = '''# CPU probe pushed by tools/notebook/kaggle_run.py: checks the secret and the tunnel, prints no secret values.
import urllib.request
QUEUE_URL = "__QUEUE_URL__"
token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = (UserSecretsClient().get_secret("QUEUE_TOKEN") or "").strip()
    print("PROBE secret loaded:", bool(token), "length:", len(token))
except Exception as e:
    print("PROBE secret error:", type(e).__name__, e)
if not token:
    import glob
    for f in sorted(glob.glob("/kaggle/input/**/QUEUE_TOKEN", recursive=True)):
        token = open(f).read().strip()
        print("PROBE token file:", f, "length:", len(token))
        break
    else:
        print("PROBE no /kaggle/input/**/QUEUE_TOKEN file")
for path in ("/health", "/bundle"):
    try:
        r = urllib.request.urlopen(urllib.request.Request(QUEUE_URL + path,
                                   headers={"Authorization": f"Bearer {token}"}), timeout=60)
        body = r.read()
        print("PROBE", path, r.status, len(body), "bytes", body[:200] if path == "/health" else "")
    except Exception as e:
        print("PROBE", path, "failed:", type(e).__name__, e)
'''


def kaggle(*args, check=True):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def queue_url():
    try:
        u = open(os.path.join(ROOT, "run/url")).read().strip().rstrip("/")
    except FileNotFoundError:
        sys.exit("run/url missing: start the queue with ./laptop/start.sh first")
    if not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", u) and "--any-url" not in sys.argv:
        sys.exit(f"run/url does not look like a quick-tunnel URL: {u!r}")
    return u


def make_token_dataset(ds):
    """Create or update a PRIVATE dataset that contains only the file QUEUE_TOKEN (= run/token)."""
    tok = open(os.path.join(ROOT, "run/token")).read().strip()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "QUEUE_TOKEN"), "w") as f:
            f.write(tok)
        json.dump({"title": ds.split("/", 1)[1], "id": ds, "licenses": [{"name": "other"}]},
                  open(os.path.join(d, "dataset-metadata.json"), "w"))
        exists = kaggle("datasets", "status", ds, check=False)
        if "ready" in exists.lower():
            print(kaggle("datasets", "version", "-p", d, "-m", "token update").strip())
        else:  # datasets are private unless --public is given
            print(kaggle("datasets", "create", "-p", d).strip())


def push(kernel, mode, timeout, token_dataset=None, extra="", accelerator="NvidiaTeslaT4"):
    url = queue_url()
    if mode == "probe":
        src = PROBE.replace("__QUEUE_URL__", url)
    else:
        src = open(os.path.join(HERE, "notebook_cell.py")).read()
        src, n = re.subn(r'^QUEUE_URL = ".*?"', f'QUEUE_URL = "{url}"', src, count=1, flags=re.M)
        if n != 1:
            sys.exit("could not find the QUEUE_URL line in notebook_cell.py")
        if extra:
            src, n = re.subn(r"^SESSION_ARGS = .*$", f"SESSION_ARGS = {extra!r}", src, count=1, flags=re.M)
            if n != 1:
                sys.exit("could not find the SESSION_ARGS line in notebook_cell.py")
    slug = kernel.split("/", 1)[1]
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    meta = {"id": kernel, "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": mode == "run", "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [token_dataset] if token_dataset else [], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}
    with tempfile.TemporaryDirectory() as d:
        json.dump(nb, open(os.path.join(d, f"{slug}.ipynb"), "w"))
        json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
        args = ["kernels", "push", "-p", d]
        if mode == "run":
            args += ["--accelerator", accelerator]
        if timeout:
            args += ["-t", str(timeout)]
        print(kaggle(*args).strip())
    print(f"pushed {mode} with QUEUE_URL={url}")


def status(kernel):
    out = kaggle("kernels", "status", kernel, check=False)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out)
    return (m.group(1) if m else "UNKNOWN"), out.strip()


def logs(kernel, since=0.0):
    raw = kaggle("kernels", "logs", kernel, check=False)
    try:
        entries = json.loads(raw)
    except ValueError:
        return since, raw.strip()
    new = [e for e in entries if e.get("time", 0) > since]
    text = "".join(("" if e.get("stream_name") == "stdout" else "[err] ") + e.get("data", "") for e in new)
    return (new[-1]["time"] if new else since), text


def follow(kernel, poll, fresh=False):
    since, last = 0.0, None
    if fresh:  # right after a push, status can still show the previous version: wait until the new run starts
        for _ in range(30):
            if status(kernel)[0] in ("QUEUED", "RUNNING"):
                break
            time.sleep(5)
    while True:
        st, raw = status(kernel)
        if st != last:
            print(f"--- status {st}", flush=True)
            last = st
        since, text = logs(kernel, since)
        if text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "UNKNOWN"):
            if st == "UNKNOWN":
                print(raw)
            return st
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, help="<user>/<notebook-slug>")
    ap.add_argument("--mode", choices=("probe", "run"))
    ap.add_argument("--logs", action="store_true", help="do not push, only show status and logs")
    ap.add_argument("--timeout", type=int, help="kernel run time limit in seconds")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--token-dataset", help="<user>/<slug> of a private dataset with a QUEUE_TOKEN file, attached to "
                    "the run (API-pushed runs do not receive Kaggle secrets)")
    ap.add_argument("--make-token-dataset", action="store_true",
                    help="create/update that private dataset from run/token first")
    ap.add_argument("--session-args", dest="extra", default="",
                    help="replace SESSION_ARGS in notebook_cell.py (tools/notebook/session.py arguments)")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle accelerator for --mode run "
                    "(NvidiaTeslaT4 = T4 x2, NvidiaTeslaP100, ...)")
    ap.add_argument("--any-url", action="store_true", help="allow a run/url that is not a trycloudflare.com address")
    a = ap.parse_args()
    if "/" not in a.kernel:
        sys.exit("--kernel must be <user>/<slug>")
    if not a.logs:
        if not a.mode:
            sys.exit("give --mode probe|run, or --logs")
        if a.make_token_dataset:
            if not a.token_dataset:
                sys.exit("--make-token-dataset needs --token-dataset <user>/<slug>")
            make_token_dataset(a.token_dataset)
        push(a.kernel, a.mode, a.timeout, a.token_dataset, a.extra, a.accelerator)
        time.sleep(5)
    if a.logs or a.no_follow:
        print(status(a.kernel)[1])
        print(logs(a.kernel)[1])
        return
    sys.exit(0 if follow(a.kernel, a.poll, fresh=not a.logs) == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
