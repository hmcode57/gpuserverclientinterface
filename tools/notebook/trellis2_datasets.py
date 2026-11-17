"""Save the TRELLIS.2 CUDA extension build as a private Kaggle dataset, so later sessions skip compiling.

  python3 tools/notebook/trellis2_datasets.py wheels     # wheels from the latest run of USER/trellis2-t4
  python3 tools/notebook/trellis2_datasets.py status     # which datasets exist, and the --dataset flags to use

worker/trellis2_t4/setup.sh compiles cumesh, flex_gemm, nvdiffrast and o_voxel when no matching wheels are
attached and leaves them in /kaggle/working/trellis2/wheels, with trellis2-t4-wheels.txt describing what they
were built for (Python, torch, CUDA, GPU arch, source commits). Attach the dataset with
kaggle_run.py --dataset USER/trellis2-t4-wheels; setup.sh installs from it only if that description matches the
session, and otherwise builds again.

Needs the Kaggle CLI with a token (~/.kaggle/access_token). The username is read from `kaggle config view`
unless --user is given. Datasets are created private. Same approach as kimodo_datasets.py on the kimodo-t4
branch.
"""
import argparse, os, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kaggle_run import kaggle  # noqa: E402

MARKER = "trellis2-t4-wheels.txt"
DATASETS = ("dinov3-vitl16-hf", "trellis2-t4-wheels")


def kaggle_user(given):
    if given:
        return given
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False))
    if not m or m.group(1) == "None":
        sys.exit("no Kaggle username: set up the Kaggle CLI token or pass --user")
    return m.group(1)


def exists(ds):
    return "ready" in kaggle("datasets", "status", ds, check=False).lower()


def publish(ds, folder, message):
    """Create the private dataset from folder, or add a new version if it already exists."""
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        f.write('{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' % (ds.split("/")[1], ds))
    print(f"uploading to {ds} ...", flush=True)
    out = kaggle("datasets", "version", "-p", folder, "-m", message) if exists(ds) else \
        kaggle("datasets", "create", "-p", folder)          # private unless --public is given
    print(out.strip())
    print(f"done. Kaggle needs a few minutes to process it; check with: kaggle datasets status {ds}")


def cmd_wheels(a, user):
    kernel = a.kernel or f"{user}/trellis2-t4"
    with tempfile.TemporaryDirectory() as d:
        print(f"fetching the wheels from the latest run of {kernel} ...", flush=True)
        kaggle("kernels", "output", kernel, "-p", d, "--file-pattern", r".*trellis2/wheels/.*")
        markers = [os.path.join(r, MARKER) for r, _, fs in os.walk(d) if MARKER in fs]
        if not markers:
            sys.exit(f"no wheels in the output of {kernel}: the latest run has to be one whose trellis2 phase "
                     "built the extensions (not one that installed them from this dataset)")
        src = os.path.dirname(markers[0])
        wheels = sorted(f for f in os.listdir(src) if f.endswith(".whl"))
        print(f"built for: {open(markers[0]).read().strip()}\nwheels: {', '.join(wheels)}")
        stage = os.path.join(d, "stage")
        os.makedirs(stage)
        for f in wheels + [MARKER]:
            shutil.move(os.path.join(src, f), os.path.join(stage, f))
        publish(f"{user}/{a.name}", stage, "extension build update")


def cmd_status(a, user):
    flags = []
    for name in DATASETS:
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
    w = sub.add_parser("wheels", help="upload the extension wheels from the latest session that built them")
    w.add_argument("--kernel", help="USER/SLUG of the notebook (default USER/trellis2-t4)")
    w.add_argument("--name", default="trellis2-t4-wheels", help="dataset slug (default trellis2-t4-wheels)")
    sub.add_parser("status", help="show which datasets exist")
    a = ap.parse_args()
    user = kaggle_user(a.user)
    {"wheels": cmd_wheels, "status": cmd_status}[a.cmd](a, user)


if __name__ == "__main__":
    main()
