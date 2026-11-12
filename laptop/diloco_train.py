"""Train one LoRA adapter across several notebook sessions with DiLoCo. The laptop only schedules: weights
and deltas move between the sessions and the object store (Cloudflare R2) through presigned URLs.

  python3 laptop/diloco_train.py --run qwen-lora --cfg cfg.json --replicas 3 --rounds 20 --steps 200

Per round r:  K train jobs (replica k: download theta_r, H AdamW steps on its data shard on all GPUs of its
session, upload the int8 pseudo-gradient)  ->  one reduce job (average, Nesterov outer step, upload
theta_{r+1}). After the last round an export job writes a PEFT adapter, which is downloaded to
data/diloco/<run>/adapter/.

State lives in data/diloco/<run>/state.json: run the same command again to resume after a laptop restart
(running jobs are picked up, not resubmitted). Sessions run:
  supervise.py --gpus all -- --handler diloco=diloco.handler:DiLoCo
"""
import argparse, json, math, os, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker")]
from client import Queue  # noqa: E402
import objstore  # noqa: E402

FAMILY = "diloco"
RUN_ARGS = ("recipe", "cfg", "replicas", "rounds", "steps", "outer_lr", "outer_momentum", "delta_bits", "seed",
            "prefix", "inner_state", "weight_by_tokens")


def log(*a):
    print(f"[diloco {time.strftime('%H:%M:%S')}]", *a, flush=True)


def token():
    t = os.environ.get("QUEUE_TOKEN")
    if not t and os.path.exists(os.path.join(ROOT, "run/token")):
        t = open(os.path.join(ROOT, "run/token")).read().strip()
    if not t:
        sys.exit("set QUEUE_TOKEN (or start the queue with laptop/start.sh, which writes run/token)")
    return t


class Trainer:
    def __init__(self, a, q, store):
        self.a, self.q, self.store = a, q, store
        self.dir = os.path.join(a.data, "diloco", a.run)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "state.json")
        cfg = vars(a)
        run = {k: cfg[k] for k in RUN_ARGS}
        if os.path.exists(self.path):
            self.st = json.load(open(self.path))
            changed = {k: (self.st["run"].get(k), v) for k, v in run.items() if self.st["run"].get(k) != v
                       and k not in ("rounds",)}  # rounds may be raised to train longer
            if changed and not a.force_resume:
                sys.exit(f"run {a.run} exists with different settings {changed}; use a new --run, or --force-resume")
            self.st["run"].update(run)
            log(f"resuming {a.run} at round {self.st['round']}/{a.rounds}")
        else:
            self.st = {"run": run, "round": 0, "theta": None, "momentum": None, "inner_state": {}, "history": [],
                       "active": {}}
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.st, open(tmp, "w"), indent=2)
        os.replace(tmp, self.path)

    def key(self, r, name):
        return f"{self.a.prefix}/{self.a.run}/r{r:05d}/{name}"

    def spec(self, **kw):
        return {"recipe": self.a.recipe, "cfg": self.a.cfg, "seed": self.a.seed, **kw}

    # ------------------------------------------------------------ jobs
    def submit(self, kind, params, priority=0):
        return self.q.submit(f"diloco-{kind}", FAMILY, params, priority=self.a.priority + priority)

    def job_for(self, slot, make):
        """Resume-safe submit: reuse the job recorded for `slot` unless it ended badly."""
        jid = self.st["active"].get(slot)
        if jid:
            j = self.q.job(jid)
            if j and j["state"] in ("queued", "running", "done"):
                return jid
        jid = make()
        self.st["active"][slot] = jid
        self.save()
        return jid

    def wait_one(self, jid, what):
        while True:
            j = self.q.job(jid)
            if j["state"] == "done":
                return j
            if j["state"] in ("failed", "cancelled"):
                raise SystemExit(f"{what} job {jid} {j['state']}: {(j.get('last_error') or '')[-1500:]}")
            time.sleep(self.a.poll)

    def init(self):
        if self.st["theta"]:
            return
        k = self.key(0, "theta.pt")
        jid = self.job_for("init", lambda: self.submit("init", {
            "op": "init", "spec": self.spec(), "put": {"theta.pt": self.store.put_url(k)}}, priority=3))
        log(f"init job {jid}: building the initial adapter")
        j = self.wait_one(jid, "init")
        log(f"initial adapter: {j['metrics'].get('trainable_params')} trainable params")
        self.st["theta"] = k
        self.st["active"].pop("init", None)
        self.save()

    def train_job(self, r, k):
        a, get = self.a, {"theta.pt": self.store.get_url(self.st["theta"])}
        prev = self.st["inner_state"].get(str(k))
        if a.inner_state and prev:
            get["inner_state.pt"] = self.store.get_url(prev)
        params = {"op": "train", "get": get,
                  "spec": self.spec(round=r, replica=k, num_replicas=a.replicas, steps=a.steps,
                                    total_steps=a.rounds * a.steps, delta_bits=a.delta_bits,
                                    keep_inner_state=a.inner_state),
                  "put": {"delta.pt": self.store.put_url(self.key(r, f"delta_{k}.pt"))}}
        if a.inner_state:
            params["put_optional"] = {"inner_state_next.pt": self.store.put_url(self.key(r, f"inner_state_{k}.pt"))}
        return self.submit("train", params, priority=1)

    def run_round(self, r):
        a = self.a
        slots = {k: f"r{r}-train-{k}" for k in range(a.replicas)}
        jobs = {k: self.job_for(s, lambda k=k: self.train_job(r, k)) for k, s in slots.items()}
        retries = {k: 0 for k in jobs}
        ok, failed, first_done = {}, {}, None
        log(f"round {r + 1}/{a.rounds}: {len(jobs)} replicas queued")
        while True:
            for k, jid in list(jobs.items()):
                if k in ok or k in failed:
                    continue
                j = self.q.job(jid)
                if j["state"] == "done":
                    ok[k] = j
                    first_done = first_done or time.time()
                    m = j.get("metrics") or {}
                    log(f"  replica {k} done on {m.get('worker')}: loss {m.get('loss_first')} -> {m.get('loss_last')}"
                        f", {m.get('train_s')}s train, {m.get('upload_mb')} MB up")
                elif j["state"] in ("failed", "cancelled"):
                    why = ((j.get("last_error") or "").strip().splitlines() or [j["state"]])[-1]
                    if retries[k] < a.replica_retries:
                        retries[k] += 1
                        jobs[k] = self.train_job(r, k)
                        self.st["active"][slots[k]] = jobs[k]
                        self.save()
                        log(f"  replica {k} {j['state']} ({why}), resubmitted as {jobs[k]}")
                    else:
                        failed[k] = j
                        log(f"  replica {k} gave up: {why}")
            if len(ok) + len(failed) == len(jobs):
                break
            if (a.straggler_timeout and first_done and len(ok) >= a.min_replicas
                    and time.time() - first_done > a.straggler_timeout):
                for k, jid in jobs.items():
                    if k not in ok and k not in failed:
                        self.q.cancel(jid)
                        log(f"  replica {k} cancelled: straggler (> {a.straggler_timeout:.0f}s after the first)")
                break
            time.sleep(a.poll)
        if len(ok) < a.min_replicas:
            raise SystemExit(f"round {r + 1}: only {len(ok)} replicas finished, need {a.min_replicas} "
                             "(state saved; fix the cause and run the same command to retry this round)")
        return ok

    def reduce(self, r, ok):
        a = self.a
        ks = sorted(ok)
        get = {"theta.pt": self.store.get_url(self.st["theta"])}
        if self.st["momentum"]:
            get["momentum.pt"] = self.store.get_url(self.st["momentum"])
        for k in ks:
            get[f"delta_{k}.pt"] = self.store.get_url(self.key(r, f"delta_{k}.pt"))
        weights = [float((ok[k].get("metrics") or {}).get("tokens") or 1) for k in ks] if a.weight_by_tokens else None
        nt, nm = self.key(r + 1, "theta.pt"), self.key(r + 1, "momentum.pt")
        params = {"op": "reduce", "get": get,
                  "spec": {"deltas": [f"delta_{k}.pt" for k in ks], "weights": weights,
                           "outer_lr": a.outer_lr, "outer_momentum": a.outer_momentum},
                  "put": {"theta_next.pt": self.store.put_url(nt), "momentum_next.pt": self.store.put_url(nm)}}
        jid = self.job_for(f"r{r}-reduce", lambda: self.submit("reduce", params, priority=2))
        j = self.wait_one(jid, "reduce")
        return nt, nm, j.get("metrics") or {}

    def cleanup(self, r, old_inner):
        """Delete objects nothing needs any more (R2's free tier is 10 GB): this round's deltas, superseded
        inner states, and weights older than --keep-rounds. Late writes from cancelled stragglers can leave a
        few orphans; a bucket lifecycle rule on the prefix catches those."""
        a = self.a
        if not a.keep_rounds:
            return
        gone = [self.key(r, f"delta_{k}.pt") for k in range(a.replicas)]
        gone += [v for k, v in old_inner.items() if self.st["inner_state"].get(k) != v]
        old = r + 1 - a.keep_rounds
        if old >= 0:
            gone += [self.key(old, "theta.pt"), self.key(old, "momentum.pt")]
        for key in gone:
            self.store.delete(key)

    def export(self):
        a = self.a
        k = self.key(self.st["round"], "adapter.tar.gz")
        jid = self.job_for("export", lambda: self.submit("export", {
            "op": "export", "spec": self.spec(), "get": {"theta.pt": self.store.get_url(self.st["theta"])},
            "put": {"adapter.tar.gz": self.store.put_url(k)}}, priority=3))
        log(f"export job {jid}")
        self.wait_one(jid, "export")
        out = os.path.join(self.dir, "adapter")
        tgz = os.path.join(self.dir, "adapter.tar.gz")
        with open(tgz, "wb") as f:
            f.write(self.store.get(k))
        os.makedirs(out, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(out, filter="data")
        self.st["active"].pop("export", None)
        self.st["exported"] = k
        self.save()
        log(f"adapter written to {out} (also in the store at {k})")

    def run(self):
        a = self.a
        self.init()
        while self.st["round"] < a.rounds:
            r = self.st["round"]
            t0 = time.time()
            ok = self.run_round(r)
            nt, nm, stats = self.reduce(r, ok)
            losses = [(v.get("metrics") or {}).get("loss_last") for v in ok.values()]
            losses = [x for x in losses if x is not None]
            entry = {"round": r + 1, "replicas": sorted(ok), "loss": round(sum(losses) / len(losses), 5) if losses else None,
                     "pseudo_grad_norm": stats.get("pseudo_grad_norm"), "seconds": round(time.time() - t0, 1),
                     "workers": {str(k): (v.get("metrics") or {}).get("worker") for k, v in ok.items()}}
            self.st["history"].append(entry)
            old_inner = dict(self.st["inner_state"])
            if a.inner_state:
                for k in ok:
                    self.st["inner_state"][str(k)] = self.key(r, f"inner_state_{k}.pt")
            self.st.update(theta=nt, momentum=nm, round=r + 1)
            self.st["active"] = {s: j for s, j in self.st["active"].items() if not s.startswith(f"r{r}-")}
            self.save()
            log(f"round {r + 1}/{a.rounds} done: mean loss {entry['loss']}, |g| {entry['pseudo_grad_norm']:.4g}, "
                f"{len(ok)}/{a.replicas} replicas, {entry['seconds']}s")
            self.cleanup(r, old_inner)
        if a.export and not self.st.get("exported"):
            self.export()
        log("finished")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name (resume key; also the object-store folder)")
    ap.add_argument("--recipe", default="diloco.recipes.hf_lora")
    ap.add_argument("--cfg", required=True, help="recipe config: JSON file or inline JSON")
    ap.add_argument("--replicas", type=int, default=2, help="K: replicas per round (usually = sessions)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200, help="H: inner optimizer steps per replica per round")
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--delta-bits", type=int, choices=(8, 32), default=8)
    ap.add_argument("--min-replicas", type=int, help="replicas needed to close a round (default: half, rounded up)")
    ap.add_argument("--straggler-timeout", type=float, default=0,
                    help="s after the first replica finishes to cancel the rest (0 = wait for all)")
    ap.add_argument("--replica-retries", type=int, default=1, help="resubmits per failed replica per round")
    ap.add_argument("--no-inner-state", dest="inner_state", action="store_false",
                    help="don't carry each replica's AdamW state between rounds (saves 2x adapter size per replica)")
    ap.add_argument("--weight-by-tokens", action="store_true", help="weight replicas by tokens seen")
    ap.add_argument("--no-export", dest="export", action="store_false")
    ap.add_argument("--keep-rounds", type=int, default=2,
                    help="keep the weights of the last N rounds in the store, delete older ones and used deltas "
                         "(0 = keep everything)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=os.environ.get("DILOCO_STORE", "r2"), help="r2 (R2_* env) or an http base URL")
    ap.add_argument("--prefix", default="diloco", help="object-store key prefix")
    ap.add_argument("--url-hours", type=float, default=48, help="presigned URL lifetime (R2 max 168)")
    ap.add_argument("--priority", type=int, default=10)
    ap.add_argument("--poll", type=float, default=5)
    ap.add_argument("--url", default=os.environ.get("QUEUE_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--force-resume", action="store_true")
    a = ap.parse_args(argv)
    a.cfg = json.load(open(a.cfg)) if os.path.exists(a.cfg) else json.loads(a.cfg)
    if a.min_replicas is None:
        a.min_replicas = max(1, math.ceil(a.replicas / 2))
    if not 1 <= a.min_replicas <= a.replicas:
        ap.error("--min-replicas must be between 1 and --replicas")
    return a


def main(argv=None):
    a = parse(argv)
    store = objstore.from_spec(a.store)
    store.expires = int(a.url_hours * 3600)
    Trainer(a, Queue(a.url, token(), "diloco-trainer"), store).run()


if __name__ == "__main__":
    main()
