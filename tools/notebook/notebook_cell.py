# One notebook cell (Kaggle or Colab). Settings: a GPU accelerator, Internet ON.
# Secrets (provider's secrets feature, NOT in this cell): QUEUE_TOKEN = contents of run/token on the laptop,
# or of run/worker_token for notebooks on accounts you don't fully trust (it can't list, read or submit jobs).
# Downloads the code bundle from the laptop queue (laptop/make_bundle.py) and runs tools/notebook/session.py.
QUEUE_URL = "https://PASTE-THE-ADDRESS-FROM-laptop/start.sh.trycloudflare.com"   # changes every tunnel restart
# Optional: where to look up the current URL, so this cell and its workers survive a laptop tunnel restart.
# e.g. "https://api.github.com/gists/<id>" (laptop/start.sh updates it when QUEUE_URL_GIST is set). Wins over
# QUEUE_URL when it answers. Leave "" to use QUEUE_URL only.
QUEUE_URL_SOURCE = ""
# Give every notebook its own --session (e.g. kaggleA, kaggleB): it prefixes the worker names on the laptop.
SESSION_ARGS = '--phases preflight,workers --worker-args "--handler echo=handlers.echo:Echo" --budget-minutes 20'

import os, subprocess, sys, tarfile, io, urllib.request
token = os.environ.get("QUEUE_TOKEN", "").strip()
secret_errors = []
if not token:
    try:  # Kaggle-style secrets
        from kaggle_secrets import UserSecretsClient
        token = (UserSecretsClient().get_secret("QUEUE_TOKEN") or "").strip()
    except Exception as e:
        secret_errors.append(f"Kaggle Secrets: {type(e).__name__}: {e}")
if not token:
    try:  # Colab-style secrets
        from google.colab import userdata
        token = (userdata.get("QUEUE_TOKEN") or "").strip()
    except Exception as e:
        secret_errors.append(f"Colab Secrets: {type(e).__name__}: {e}")
if not token:  # private Kaggle dataset holding a file named QUEUE_TOKEN (API-pushed runs get no secrets)
    import glob
    for f in sorted(glob.glob("/kaggle/input/**/QUEUE_TOKEN", recursive=True)):
        token = open(f).read().strip()
        if token:
            break
    else:
        secret_errors.append("no /kaggle/input/**/QUEUE_TOKEN file")
if not token:
    detail = " | ".join(secret_errors) if secret_errors else "no secret provider returned a value"
    raise RuntimeError(
        "QUEUE_TOKEN was not available. In Kaggle open Add-ons > Secrets, create or attach a secret named "
        "exactly QUEUE_TOKEN, paste the contents of the laptop run/token file as its value, then run this cell again. "
        "Do not paste the token into this notebook. Detail: " + detail)
if QUEUE_URL_SOURCE:
    import json, re
    try:
        r = urllib.request.urlopen(urllib.request.Request(QUEUE_URL_SOURCE, headers={"User-Agent": "gpuq-client"}),
                                   timeout=20).read().decode()
        if r.lstrip().startswith("{"):  # GitHub gist API: file "queue_url", else the first file
            fs = json.loads(r)["files"]
            r = (fs.get("queue_url") or next(iter(fs.values())))["content"]
        QUEUE_URL = re.search(r"https://[A-Za-z0-9.-]+(?::\d+)?", r).group(0)  # https only: the token goes there
        print("QUEUE_URL from QUEUE_URL_SOURCE:", QUEUE_URL)
    except Exception as e:
        print(f"QUEUE_URL_SOURCE unreadable ({type(e).__name__}: {e}), using QUEUE_URL")
    os.environ["QUEUE_URL_SOURCE"] = QUEUE_URL_SOURCE  # workers re-read it if the tunnel moves
os.environ["QUEUE_URL"], os.environ["QUEUE_TOKEN"] = QUEUE_URL.rstrip("/"), token
req = urllib.request.Request(QUEUE_URL.rstrip("/") + "/bundle", headers={"Authorization": f"Bearer {token}"})
work = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.makedirs(f"{work}/gpuq", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(req, timeout=120).read())).extractall(f"{work}/gpuq")
print(open(f"{work}/gpuq/BUNDLE.json").read())
p = subprocess.Popen(f"{sys.executable} -u tools/notebook/session.py --out {work}/session_out {SESSION_ARGS}",
                     shell=True, cwd=f"{work}/gpuq", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
print("exit code", p.wait())
