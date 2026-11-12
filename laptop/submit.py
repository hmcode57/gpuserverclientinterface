"""Submit a job (and its input files) to the queue, or list jobs and workers.

python3 laptop/submit.py --kind echo --family echo --params '{"text": "hi"}' --file in.txt=README.md \
    [--config example@1] [--priority 5] [--after JOB_ID] [--output result.json]
python3 laptop/submit.py --list
python3 laptop/submit.py --workers
python3 laptop/submit.py --get JOB_ID result.json > result.json

QUEUE_URL defaults to the local queue; QUEUE_TOKEN is read from the environment or run/token.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "worker"))
from client import Queue


def token():
    t = os.environ.get("QUEUE_TOKEN")
    if not t and os.path.exists(os.path.join(ROOT, "run/token")):
        t = open(os.path.join(ROOT, "run/token")).read().strip()
    if not t:
        sys.exit("set QUEUE_TOKEN (or start the queue with laptop/start.sh, which writes run/token)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("QUEUE_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--list", action="store_true", help="list jobs")
    ap.add_argument("--workers", action="store_true", help="list workers")
    ap.add_argument("--get", nargs=2, metavar=("JOB_ID", "FILE"), help="write a job file to stdout")
    ap.add_argument("--kind"); ap.add_argument("--family")
    ap.add_argument("--params", default="{}", help="JSON object")
    ap.add_argument("--config", help="name@version (configs/<name>/v<version>.json)")
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--after", help="job id that must be done first")
    ap.add_argument("--file", action="append", default=[], help="input file: name=path (repeatable)")
    ap.add_argument("--output", action="append", default=[], help="declare an output file name (repeatable)")
    a = ap.parse_args()
    q = Queue(a.url, token(), "laptop")
    if a.list:
        for j in q.jobs():
            print(f"{j['id']} {j['state']:8} {j['kind']}/{j['family']} prio={j['priority']} attempts={j['attempts']} "
                  f"after={j['after'] or '-'} files={','.join(j['files'])}"
                  + (f"  [waiting: {j['waiting_for']}]" if j.get("waiting_for") else "")
                  + (f"  [error: {j['last_error'].strip().splitlines()[-1][:120]}]" if j.get("last_error") else ""))
        return
    if a.workers:
        for w in q.workers():
            print(f"{w['name']:28} {w['state']:8} family={w['family'] or '-'} job={w['job'] or '-'} "
                  f"seen {w['age_s']}s ago  {json.dumps(w['info'])[:120]}")
        return
    if a.get:
        sys.stdout.buffer.write(q.download(*a.get))
        return
    if not a.kind or not a.family:
        ap.error("--kind and --family are required to submit (or use --list / --workers / --get)")
    files = {}
    for spec in a.file:
        name, path = spec.split("=", 1)
        files[name] = open(path, "rb").read()
    print(q.submit(a.kind, a.family, json.loads(a.params), files, a.config, a.priority, a.after, a.output))


if __name__ == "__main__":
    main()
