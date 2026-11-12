"""Builds data/bundle.tar.gz: the code (and optional extra files) a notebook session downloads from the queue
(GET /bundle), so notebooks always run exactly the laptop's current code without git access or uploads.

  python3 laptop/make_bundle.py [--include path[=name-in-bundle] ...]

Always contains worker/ (runtime, client, handlers), tools/notebook/, configs/ and BUNDLE.json (build time,
git commit, whether those folders had uncommitted changes). Use --include for your own handler packages,
requirements files or test inputs, e.g. --include ../my-model/handlers=worker/my_handlers.
"""
import argparse, io, json, os, subprocess, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALWAYS = ("worker", "tools/notebook", "configs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", action="append", default=[], help="path or path=name-in-bundle (repeatable)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/bundle.tar.gz"))
    a = ap.parse_args()
    extra = []
    for spec in a.include:
        src, _, arc = spec.partition("=")
        if not os.path.exists(src):
            sys.exit(f"--include {src}: no such file or folder")
        extra.append((src, arc or os.path.basename(os.path.normpath(src))))
    git = lambda *args: subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", *ALWAYS))
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty,
            "included": [arc for _, arc in extra]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".part"

    def skip(ti):
        return None if "__pycache__" in ti.name or ti.name.endswith((".pyc", ".part")) else ti

    with tarfile.open(tmp, "w:gz") as t:
        for d in ALWAYS:
            t.add(os.path.join(ROOT, d), arcname=d, filter=skip)
        for src, arc in extra:
            t.add(src, arcname=arc, filter=skip)
        b = json.dumps(meta, indent=2).encode()
        ti = tarfile.TarInfo("BUNDLE.json"); ti.size = len(b); ti.mtime = int(time.time())
        t.addfile(ti, io.BytesIO(b))
    os.replace(tmp, a.out)
    print(f"bundle {a.out} ({os.path.getsize(a.out) // 1024} KB), commit {commit or 'none'}"
          f"{' +local changes' if dirty else ''}" + (f", extra: {', '.join(meta['included'])}" if extra else ""))


if __name__ == "__main__":
    main()
