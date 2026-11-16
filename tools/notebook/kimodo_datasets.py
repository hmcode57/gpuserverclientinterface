"""Create (or update) the two private Kaggle datasets that make kimodo sessions start in seconds.

  python3 tools/notebook/kimodo_datasets.py weights            # Q8_0 weights + tokenizer (8 GB, once)
  python3 tools/notebook/kimodo_datasets.py weights --download # fetch them from Hugging Face first
  python3 tools/notebook/kimodo_datasets.py bin                # CUDA build from the last successful session
  python3 tools/notebook/kimodo_datasets.py status             # which exist, and the --dataset flags to use

worker/kimodo/setup.sh finds both under /kaggle/input and skips the 8 GB download and the ~10 min build.
Attach them to terminal runs with kaggle_run.py --dataset USER/kimodo-q8 --dataset USER/kimodo-cuda-bin.

Needs the Kaggle CLI with a token (~/.kaggle/access_token). The Kaggle username is read from
`kaggle config view` unless --user is given. Datasets are created private.
"""
import argparse, hashlib, os, re, shutil, subprocess, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kaggle_run import kaggle  # noqa: E402

HF = "https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/main"
FILES = {  # same files and checksums as worker/kimodo/setup.sh
    "Llama-3-Kimodo-Q8_0.gguf": "b26d0e74b115b33a7f2df5aca04426548365e99217cd590986f8abc4172e4c5c",
    "tokenizer.gguf": "81614aca62a98846c02b72cc2e5378e5bdce8b4d507b88f96eb1faa90dae607e",
}
BIN = "kimodo-cuda-bin.tar.gz"


def kaggle_user(given):
    if given:
        return given
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False))
    if not m or m.group(1) == "None":
        sys.exit("no Kaggle username: set up the Kaggle CLI token (docs/kimodo-t4-setup.md 3B.1) or pass --user")
    return m.group(1)


def exists(ds):
    return "ready" in kaggle("datasets", "status", ds, check=False).lower()


def publish(ds, folder, message):
    """Create the private dataset from folder, or add a new version if it already exists."""
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        f.write('{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' % (ds.split("/")[1], ds))
    print(f"uploading to {ds} (this can take a while) ...", flush=True)
    if exists(ds):
        out = kaggle("datasets", "version", "-p", folder, "-m", message)
    else:
        out = kaggle("datasets", "create", "-p", folder)  # private unless --public is given
    print(out.strip())
    print(f"done. Kaggle needs a few minutes to process it; check with: kaggle datasets status {ds}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    part = dest + ".part"
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
    os.replace(part, dest)


def cmd_weights(a, user):
    src = os.path.expanduser(a.dir)
    if a.download:
        os.makedirs(src, exist_ok=True)
    for name, digest in FILES.items():
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            if not a.download:
                sys.exit(f"{path} missing: point --dir at the folder that has it, or add --download")
            download(f"{HF}/{name}", path)
        if not a.no_verify:
            print(f"checking {name} ...", flush=True)
            if sha256(path) != digest:
                sys.exit(f"{path}: checksum mismatch, delete it and rerun with --download")
    stage = os.path.join(src, "kaggle-q8")  # same filesystem as the weights, so hard links work
    os.makedirs(stage, exist_ok=True)
    for name in FILES:
        target = os.path.join(stage, name)
        if os.path.exists(target):
            os.remove(target)
        try:
            os.link(os.path.join(src, name), target)
        except OSError as e:
            sys.exit(f"could not hard-link {name} into {stage} ({e}); copy it there yourself and rerun")
    publish(f"{user}/{a.name}", stage, "weights update")


def cmd_bin(a, user):
    kernel = a.kernel or f"{user}/kimodo-t4"
    with tempfile.TemporaryDirectory() as d:
        print(f"fetching {BIN} from the latest run of {kernel} ...", flush=True)
        kaggle("kernels", "output", kernel, "-p", d, "--file-pattern", re.escape(BIN))
        found = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f == BIN]
        if not found:
            sys.exit(f"no {BIN} in the output of {kernel}: the latest run has to be one whose kimodo phase "
                     "ended ok=True (and built, not used a prebuilt binary)")
        stage = os.path.join(d, "stage")
        os.makedirs(stage)
        shutil.move(found[0], os.path.join(stage, BIN))
        publish(f"{user}/{a.name}", stage, "CUDA build update")


def cmd_status(a, user):
    flags = []
    for name in ("kimodo-q8", "kimodo-cuda-bin"):
        ds = f"{user}/{name}"
        ok = exists(ds)
        print(f"{ds:40s} {'ready' if ok else 'missing or still processing'}")
        if ok:
            flags.append(f"--dataset {ds}")
    if flags:
        print("\nadd to kaggle_run.py --mode run:  " + " ".join(flags))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="Kaggle username (default: from `kaggle config view`)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights", help="upload Llama-3-Kimodo-Q8_0.gguf + tokenizer.gguf")
    w.add_argument("--dir", default="~/kimodo", help="folder holding the two .gguf files (default ~/kimodo)")
    w.add_argument("--download", action="store_true", help="download missing files from Hugging Face first")
    w.add_argument("--no-verify", action="store_true", help="skip the sha256 check (about 30 s for 8 GB)")
    w.add_argument("--name", default="kimodo-q8", help="dataset slug (default kimodo-q8)")
    b = sub.add_parser("bin", help=f"upload {BIN} from the latest successful session")
    b.add_argument("--kernel", help="USER/SLUG of the notebook (default USER/kimodo-t4)")
    b.add_argument("--name", default="kimodo-cuda-bin", help="dataset slug (default kimodo-cuda-bin)")
    sub.add_parser("status", help="show which datasets exist")
    a = ap.parse_args()
    user = kaggle_user(a.user)
    {"weights": cmd_weights, "bin": cmd_bin, "status": cmd_status}[a.cmd](a, user)


if __name__ == "__main__":
    main()
