"""Send kimodo prompts to the queue at any time during a session and collect the embeddings.

The Kaggle workers keep their encoders loaded and keep pulling jobs, so new prompts can be added whenever you
like. Each job holds a batch of prompts (one HTTP round trip per job, ~50 ms of GPU per prompt).

  # one-off: submit, wait, write embeddings to out/
  python3 laptop/kimodo_prompts.py "a person walks forward and waves" "someone jumps twice" --wait

  # a file, 32 prompts per job
  python3 laptop/kimodo_prompts.py --file prompts.txt --batch 32 --wait

  # continuous: every line typed or piped in becomes a prompt; flushed per --batch or after --flush s quiet
  tail -f new_prompts.txt | python3 laptop/kimodo_prompts.py --watch

  # continuous: every *.txt dropped into inbox/ becomes a job (one prompt per line); moved to inbox/sent/
  python3 laptop/kimodo_prompts.py --watch-dir inbox/

  # fetch results of earlier jobs that weren't waited for
  python3 laptop/kimodo_prompts.py --fetch JOB_ID [JOB_ID ...]

Results: <out>/<job>.f32 (float32 [n, 4096]; np.fromfile(f, "<f4").reshape(-1, 4096)) and <out>/<job>.json
(the prompts in row order). --watch and --watch-dir always collect results in the background.
QUEUE_URL / QUEUE_TOKEN as for submit.py.
"""
import argparse, glob, json, os, queue, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "worker"))
sys.path.insert(0, os.path.join(ROOT, "laptop"))
from client import Queue
from submit import token

FAMILY, KIND = "kimodo", "encode"


class Collector:
    """Polls submitted jobs and downloads each one's embeddings when it is done."""

    def __init__(self, q, out, poll=2.0):
        self.q, self.out, self.poll = q, out, poll
        self.pending, self.lock = set(), threading.Lock()
        os.makedirs(out, exist_ok=True)

    def add(self, job_id):
        with self.lock:
            self.pending.add(job_id)

    def step(self):
        with self.lock:
            ids = list(self.pending)
        for jid in ids:
            try:
                j = self.q.job(jid)
            except Exception as e:
                print(f"[collect] {jid}: {e}", file=sys.stderr); continue
            if j["state"] == "done":
                emb = self.q.download(jid, "embeddings.f32")
                meta = json.loads(self.q.download(jid, "result.json"))
                with open(os.path.join(self.out, f"{jid}.f32"), "wb") as f:
                    f.write(emb)
                with open(os.path.join(self.out, f"{jid}.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                ms = (j.get("metrics") or {}).get("encode_ms_mean")
                print(f"[done] {jid}: {len(meta['prompts'])} prompts -> {self.out}/{jid}.f32"
                      + (f"  ({ms} ms/prompt on {j.get('worker') or meta.get('backend')})" if ms else ""), flush=True)
            elif j["state"] in ("failed", "cancelled"):
                err = (j.get("last_error") or "").strip().splitlines()[-1:] or [""]
                print(f"[{j['state']}] {jid}: {err[0][:200]}", file=sys.stderr, flush=True)
            else:
                continue
            with self.lock:
                self.pending.discard(jid)

    def run_forever(self):
        while True:
            self.step()
            time.sleep(self.poll)

    def wait_all(self):
        while True:
            self.step()
            with self.lock:
                if not self.pending:
                    return
            time.sleep(self.poll)


def submit(q, prompts, priority, collector):
    jid = q.submit(KIND, FAMILY, {"prompts": prompts}, priority=priority,
                   outputs=["embeddings.f32", "result.json"])
    print(f"[queued] {jid}: {len(prompts)} prompt(s)", flush=True)
    if collector:
        collector.add(jid)
    return jid


def watch_stdin(q, a, collector):
    lines = queue.Queue()

    def reader():
        for line in sys.stdin:
            lines.put(line.strip())
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    batch, last = [], time.time()
    while True:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            line = ""
        if line is None:
            break
        if line:
            batch.append(line); last = time.time()
        if batch and (len(batch) >= a.batch or time.time() - last >= a.flush):
            submit(q, batch, a.priority, collector); batch = []
    if batch:
        submit(q, batch, a.priority, collector)


def watch_dir(q, a, collector):
    sent = os.path.join(a.watch_dir, "sent")
    os.makedirs(sent, exist_ok=True)
    print(f"watching {a.watch_dir}/*.txt (Ctrl-C to stop)", flush=True)
    while True:
        for path in sorted(glob.glob(os.path.join(a.watch_dir, "*.txt"))):
            if time.time() - os.path.getmtime(path) < 1.0:  # still being written
                continue
            prompts = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
            for i in range(0, len(prompts), a.batch):
                submit(q, prompts[i:i + a.batch], a.priority, collector)
            os.replace(path, os.path.join(sent, os.path.basename(path)))
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--url", default=os.environ.get("QUEUE_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--file", help="one prompt per line")
    ap.add_argument("--batch", type=int, default=32, help="prompts per job (default 32)")
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/kimodo_embeddings"))
    ap.add_argument("--wait", action="store_true", help="wait for the jobs and download their embeddings")
    ap.add_argument("--watch", action="store_true", help="read prompts from stdin continuously")
    ap.add_argument("--flush", type=float, default=2.0, help="--watch: submit a partial batch after this many s")
    ap.add_argument("--watch-dir", help="submit every *.txt that appears in this folder")
    ap.add_argument("--fetch", nargs="+", metavar="JOB_ID", help="download results of earlier jobs")
    a = ap.parse_args()
    q = Queue(a.url, token(), "laptop")

    if a.fetch:
        c = Collector(q, a.out)
        for jid in a.fetch:
            c.add(jid)
        return c.wait_all()
    if a.watch or a.watch_dir:
        c = Collector(q, a.out)
        threading.Thread(target=c.run_forever, daemon=True).start()
        try:
            if a.watch_dir:
                watch_dir(q, a, c)
            watch_stdin(q, a, c)
            c.wait_all()  # stdin closed: collect what's still running
        except KeyboardInterrupt:
            print(f"\nstopped; still running: {' '.join(sorted(c.pending)) or 'none'} "
                  "(collect later with --fetch)", file=sys.stderr)
        return
    prompts = list(a.prompts)
    if a.file:
        prompts += [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    if not prompts:
        ap.error("give prompts, --file, --watch, --watch-dir or --fetch")
    c = Collector(q, a.out) if a.wait else None
    for i in range(0, len(prompts), a.batch):
        submit(q, prompts[i:i + a.batch], a.priority, c)
    if c:
        c.wait_all()


if __name__ == "__main__":
    main()
