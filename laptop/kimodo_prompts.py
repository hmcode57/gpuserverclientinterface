"""Send kimodo prompts to the queue at any time during a session and collect the embeddings.

The Kaggle workers keep their encoders loaded and keep pulling jobs, so new prompts can be added whenever you
like. By default each prompt is its own job, so it arrives as its own <job>.f32/.json pair as soon as it is
encoded. --batch N packs N prompts into one job (fewer tunnel round trips; ~50 ms of GPU per prompt).

  # one-off: submit, wait, write embeddings to out/
  python3 laptop/kimodo_prompts.py "a person walks forward and waves" "someone jumps twice" --wait

  # a file, one job (and one result file pair) per prompt
  python3 laptop/kimodo_prompts.py --file prompts.txt --wait

  # a large file, 32 prompts per job (faster overall, one result file pair per 32 prompts)
  python3 laptop/kimodo_prompts.py --file prompts.txt --batch 32 --wait

  # continuous: every line typed or piped in becomes a prompt; flushed per --batch or after --flush s quiet
  my_prompt_generator | python3 laptop/kimodo_prompts.py --watch

  # continuous: every *.txt dropped into inbox/ becomes a job (one prompt per line); moved to inbox/sent/
  python3 laptop/kimodo_prompts.py --watch-dir inbox/

  # follow one file you keep adding to: only prompts not sent before, one labelled file pair per prompt
  # (out/0001-a-person-waves.f32 + .json); safe with any editor, remembers what was sent across restarts
  python3 laptop/kimodo_prompts.py --follow prompts.txt

  # fetch results of earlier jobs that weren't waited for
  python3 laptop/kimodo_prompts.py --fetch JOB_ID [JOB_ID ...]

Results: <out>/<job>.f32 (float32 [n, 4096]; np.fromfile(f, "<f4").reshape(-1, 4096)) and <out>/<job>.json
(the prompts in row order). --watch and --watch-dir always collect results in the background.
QUEUE_URL / QUEUE_TOKEN as for submit.py.
"""
import argparse, glob, json, os, queue, re, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "worker"))
sys.path.insert(0, os.path.join(ROOT, "laptop"))
from client import Queue
from submit import token

FAMILY, KIND = "kimodo", "encode"


class Collector:
    """Polls submitted jobs and downloads each one's embeddings when it is done."""

    def __init__(self, q, out, poll=2.0, on_done=None, on_failed=None):
        self.q, self.out, self.poll = q, out, poll
        self.on_done, self.on_failed = on_done, on_failed  # replace the default <job>.f32/.json writing
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
                if self.on_done:
                    where = self.on_done(jid, emb, meta)
                else:
                    with open(os.path.join(self.out, f"{jid}.f32"), "wb") as f:
                        f.write(emb)
                    with open(os.path.join(self.out, f"{jid}.json"), "w") as f:
                        json.dump(meta, f, indent=2)
                    where = f"{self.out}/{jid}.f32"
                ms = (j.get("metrics") or {}).get("encode_ms_mean")
                print(f"[done] {jid}: {len(meta['prompts'])} prompts -> {where}"
                      + (f"  ({ms} ms/prompt on {j.get('worker') or meta.get('backend')})" if ms else ""), flush=True)
            elif j["state"] in ("failed", "cancelled"):
                err = (j.get("last_error") or "").strip().splitlines()[-1:] or [""]
                print(f"[{j['state']}] {jid}: {err[0][:200]}", file=sys.stderr, flush=True)
                if self.on_failed:
                    self.on_failed(jid)
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


def slug(text, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "prompt"


class Follow:
    """--follow FILE: send every prompt line that wasn't sent before, write one file pair per prompt.

    State (which prompt texts were sent, their number, job and row) lives in <out>/.follow-<file>.json, so it
    works with editors that replace the file on save, across restarts, and when lines are deleted or moved.
    A prompt text is sent once; edit a line to send it as a new prompt."""

    def __init__(self, q, a):
        self.q, self.a, self.lock = q, a, threading.Lock()
        self.path = a.follow
        if not os.path.isfile(self.path):
            sys.exit(f"--follow: {self.path} does not exist (check the path; create an empty file to start from "
                     "nothing: touch {self.path})".replace("{self.path}", self.path))
        os.makedirs(a.out, exist_ok=True)
        self.state_path = os.path.join(a.out, f".follow-{os.path.basename(self.path)}.json")
        if os.path.exists(self.state_path):
            self.st = json.load(open(self.state_path, encoding="utf-8"))
        else:
            self.st = {"file": os.path.abspath(self.path), "next": 1, "prompts": {}}
            if not a.include_existing:  # first start: lines already in the file count as done
                for p in self.lines():
                    self.st["prompts"][p] = {"skipped": True}
                n = len(self.st["prompts"])
                if n:
                    print(f"[follow] {n} prompt(s) already in {self.path} skipped "
                          "(--include-existing sends them)", flush=True)
            self.save()
        self.collector = Collector(q, a.out, on_done=self.done, on_failed=self.failed)
        for e in self.st["prompts"].values():  # results still outstanding from an earlier run
            if e.get("job") and not e.get("done") and not e.get("failed"):
                self.collector.add(e["job"])

    def lines(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return list(dict.fromkeys(l.strip() for l in f if l.strip()))
        except FileNotFoundError:
            return []

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.st, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.state_path)

    def entries(self, jid):
        return {e["row"]: (p, e) for p, e in self.st["prompts"].items() if e.get("job") == jid}

    def done(self, jid, emb, meta):
        dim = meta.get("shape", [0, 4096])[1]
        names = []
        with self.lock:
            for row, (p, e) in sorted(self.entries(jid).items()):
                name = f"{e['n']:04d}-{slug(p)}"
                with open(os.path.join(self.a.out, name + ".f32"), "wb") as f:
                    f.write(emb[row * dim * 4:(row + 1) * dim * 4])
                with open(os.path.join(self.a.out, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"prompt": p, "n": e["n"], "job": jid, "row": row, "shape": [dim], "dtype": "<f4",
                               "backend": meta.get("backend")}, f, indent=2, ensure_ascii=False)
                e["done"] = True
                names.append(name)
            self.save()
        return ", ".join(f"{self.a.out}/{n}.f32" for n in names)

    def failed(self, jid):
        with self.lock:
            for _, e in self.entries(jid).values():
                e["failed"] = True
            self.save()
        print("[follow] edit the line to send it again as a new prompt", file=sys.stderr, flush=True)

    def send_new(self):
        new = [p for p in self.lines() if p not in self.st["prompts"]]
        for i in range(0, len(new), self.a.batch):
            chunk = new[i:i + self.a.batch]
            with self.lock:
                jid = self.q.submit(KIND, FAMILY, {"prompts": chunk}, priority=self.a.priority,
                                    outputs=["embeddings.f32", "result.json"])
                for row, p in enumerate(chunk):
                    self.st["prompts"][p] = {"n": self.st["next"], "job": jid, "row": row}
                    self.st["next"] += 1
                self.save()
            print(f"[queued] {jid}: " + "; ".join(chunk)[:150], flush=True)
            self.collector.add(jid)

    def run(self):
        threading.Thread(target=self.collector.run_forever, daemon=True).start()
        print(f"[follow] watching {self.path} for new lines -> {self.a.out}/NNNN-<prompt>.f32 (Ctrl-C to stop)",
              flush=True)
        seen, changed_at = None, 0.0
        try:
            while True:
                try:
                    st = os.stat(self.path)
                    sig = (st.st_mtime_ns, st.st_size, st.st_ino)
                except FileNotFoundError:
                    if seen is not None:
                        print(f"[follow] {self.path} disappeared; waiting for it to come back", file=sys.stderr,
                              flush=True)
                    sig = None
                if sig != seen:
                    seen, changed_at = sig, time.time()
                elif changed_at and time.time() - changed_at >= self.a.settle:  # unchanged for a moment: read it
                    changed_at = 0.0
                    self.send_new()
                time.sleep(0.25)
        except KeyboardInterrupt:
            with self.lock:
                open_ = sorted({e["job"] for e in self.st["prompts"].values()
                                if e.get("job") and not e.get("done") and not e.get("failed")})
            print(f"\nstopped; {len(open_)} job(s) still running, collected on the next --follow run",
                  file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--url", default=os.environ.get("QUEUE_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--file", help="one prompt per line")
    ap.add_argument("--batch", type=int, default=1,
                    help="prompts per job (default 1: one result file pair per prompt)")
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "prompts-for-encoder-files/out"))
    ap.add_argument("--wait", action="store_true", help="wait for the jobs and download their embeddings")
    ap.add_argument("--watch", action="store_true", help="read prompts from stdin continuously")
    ap.add_argument("--flush", type=float, default=2.0, help="--watch: submit a partial batch after this many s")
    ap.add_argument("--watch-dir", help="submit every *.txt that appears in this folder")
    ap.add_argument("--follow", metavar="FILE", help="send new prompt lines of FILE as you add them; one labelled "
                    "file pair per prompt; remembers what was sent (works with editors and across restarts)")
    ap.add_argument("--include-existing", action="store_true",
                    help="--follow, first start: also send the lines already in FILE")
    ap.add_argument("--settle", type=float, default=1.0, help="--follow: read FILE after it is unchanged this long")
    ap.add_argument("--fetch", nargs="+", metavar="JOB_ID", help="download results of earlier jobs")
    a = ap.parse_args()
    q = Queue(a.url, token(), "laptop")

    if a.fetch:
        c = Collector(q, a.out)
        for jid in a.fetch:
            c.add(jid)
        return c.wait_all()
    if a.follow:
        return Follow(q, a).run()
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
        ap.error("give prompts, --file, --follow, --watch, --watch-dir or --fetch")
    c = Collector(q, a.out) if a.wait else None
    for i in range(0, len(prompts), a.batch):
        submit(q, prompts[i:i + a.batch], a.priority, c)
    if c:
        c.wait_all()


if __name__ == "__main__":
    main()
