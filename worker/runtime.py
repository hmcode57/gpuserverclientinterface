"""Generic long-lived worker: one process per GPU, pulls jobs from the queue, runs them with a handler.

  QUEUE_URL=... QUEUE_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
      python3 worker/runtime.py --handler echo=handlers.echo:Echo [--handler gemm=handlers.torch_matmul:Matmul]

Each --handler maps a job family to a Handler class ("module:Class", importable from worker/ or the current
directory). Only one handler (one model family) is resident at a time: the first one loads at startup, the
others load on demand and replace it. The queue prefers jobs of the family already loaded, so swaps are rare.

Lifecycle: starting -> load -> warmup -> ready -> claim -> busy -> run -> complete -> clear caches -> ready.
A background status thread reports the state every 30 s so the queue (and any UI) can see the worker.
While a job runs, a heartbeat thread keeps its lease; if the queue says the lease is gone (job requeued after a
missed heartbeat, or finished elsewhere), the handler's next ctx.progress()/ctx.check() raises LeaseLost and the
job is abandoned without reporting anything.

Exit codes (the contract with worker/supervise.py):
  0  finished normally (--once, --idle-exit, Ctrl-C)
  1  a job raised: the failure was reported, restart a clean process (GPU state may be corrupt)
  2  misconfiguration (bad token, outdated client, bad arguments): restarting will not help
"""
import argparse, gc, importlib, json, os, resource, socket, sys, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.getcwd()]
from client import Heartbeat, LeaseLost, Queue  # noqa: E402

EXIT_OK, EXIT_JOB_FAILED, EXIT_CONFIG = 0, 1, 2


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)  # Linux: KiB


def torch_or_none():
    """torch is optional: CPU-only handlers and tests run without it."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def default_device():
    torch = torch_or_none()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


class Handler:
    """Base class for a job family. Subclass it and override `load` and `run`; everything else is optional.

    One instance stays resident while the worker runs jobs of its family, so load models in `load`, not `run`.
    """
    family = None  # set by the runtime from the --handler mapping

    def __init__(self, device, options):
        self.device = device      # "cuda" or "cpu" (the process sees exactly one GPU via CUDA_VISIBLE_DEVICES)
        self.options = options    # {key: value} from repeated --opt key=value

    def load(self):
        """Load models/weights. Called once when this family becomes resident."""

    def warmup(self):
        """Optional first run so compile/allocation cost is not charged to the first job. Returns metrics."""
        return {}

    def info(self):
        """Static facts reported with the worker status (GPU name, dtype, model revision, ...)."""
        return {}

    def run(self, job, ctx):
        """Do the job. Read inputs with ctx.input(), write results with ctx.output(), report progress with
        ctx.progress(). Return a dict of metrics (stored on the job). Raise to fail the job."""
        raise NotImplementedError

    def close(self):
        """Release models before another family is loaded. The runtime also runs gc and empties the CUDA cache."""


class JobContext:
    def __init__(self, q, job, hb):
        self.q, self.job, self.hb = q, job, hb
        self.id = job["id"]
        self.params = job.get("params") or {}
        self._config = None

    def input(self, name):
        """Bytes of an input file uploaded with the job."""
        if name not in self.job["files"]:
            raise ValueError(f"job {self.id} has no input file {name!r} (has: {sorted(self.job['files'])})")
        return self.q.download(self.id, name)

    def output(self, name, data):
        """Upload a result file. Only this worker (holding the lease) can write it."""
        self.hb.check()
        if isinstance(data, str):
            data = data.encode()
        self.q.upload(self.id, name, data)

    def config(self):
        """The job's immutable config (dict), or None if it was submitted without one."""
        if self._config is None and self.job.get("config"):
            self._config = self.q.config(self.job["config"])
        return self._config

    def progress(self, **fields):
        """Report progress (sent with the next heartbeat) and stop if the lease was lost."""
        self.hb.progress = fields
        self.hb.check()

    def check(self):
        self.hb.check()


def load_class(spec):
    mod, _, cls = spec.partition(":")
    if not mod or not cls:
        raise ValueError(f"handler must look like module:Class, got {spec!r}")
    return getattr(importlib.import_module(mod), cls)


def parse_handlers(items):
    out = {}
    for item in items:
        fam, sep, spec = item.partition("=")
        if not sep or not fam or not spec:
            raise ValueError(f"--handler must look like family=module:Class, got {item!r}")
        out[fam] = spec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handler", action="append", required=True, help="family=module:Class (repeatable)")
    ap.add_argument("--name", default=f"{socket.gethostname()}-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if torch sees a GPU)")
    ap.add_argument("--opt", action="append", default=[], help="key=value passed to handlers (repeatable)")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between claims while idle")
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--status-every", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="exit after one job (or when the queue is empty)")
    ap.add_argument("--idle-exit", type=float, default=0, help="exit 0 after this many idle seconds (0 = never)")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    try:
        specs = parse_handlers(a.handler)
        options = dict(o.split("=", 1) for o in a.opt)
        classes = {fam: load_class(spec) for fam, spec in specs.items()}
    except (ValueError, ImportError, AttributeError) as e:
        log("bad arguments:", e)
        return EXIT_CONFIG
    device = a.device or default_device()
    fams = list(specs)
    q = Queue(os.environ["QUEUE_URL"], os.environ["QUEUE_TOKEN"], a.name)
    try:
        q.status("starting", info={"host": socket.gethostname(), "device": device, "families": fams})
    except Exception as e:
        if getattr(e, "code", None) in (400, 401, 426):
            log("queue refused this worker:", e)
            return EXIT_CONFIG
        raise
    torch = torch_or_none()
    on_cuda = device.startswith("cuda") and torch is not None
    resident = {"h": None}

    def clear_caches():
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

    def load(fam):
        """One family resident at a time; swapping closes the old handler first."""
        h = resident["h"]
        if h is not None and h.family == fam:
            return h
        if h is not None:
            log(f"unloading {h.family} for a {fam} job")
            h.close()
            resident["h"] = None
            clear_caches()
        t = time.time()
        h = classes[fam](device, options)
        h.family = fam
        h.load()
        resident["h"] = h
        log(f"loaded {fam} in {time.time() - t:.1f}s")
        return h

    h = load(fams[0])
    info = {"device": device, **h.info()}
    if not a.no_warmup:
        w = h.warmup() or {}
        if w:
            info["warmup"] = w
            log("warmup", w)
    q.status("ready", h.family, info)
    log(f"ready as {a.name}, families {','.join(fams)}, device {device}")

    stop = threading.Event()
    state = {"state": "ready", "job": None}

    def alive():  # keep-alive status; reports the current state, never overwrites busy
        while not stop.wait(a.status_every):
            try:
                q.status(state["state"], resident["h"].family if resident["h"] else None, job=state["job"])
            except Exception as e:
                log("status failed:", e)
    threading.Thread(target=alive, daemon=True).start()

    idle_since = time.time()
    try:
        while True:
            try:
                job = q.claim(resident["h"].family, accept=fams)
            except Exception as e:  # tunnel down: keep the loaded model and retry (the claim keeps its request_id)
                if getattr(e, "code", None) in (400, 401, 426):
                    log("queue refused the claim:", e)
                    return EXIT_CONFIG
                log("claim failed, retrying:", e)
                time.sleep(max(a.poll, 10))
                continue
            if not job:
                if a.once or (a.idle_exit and time.time() - idle_since > a.idle_exit):
                    log("idle, exiting")
                    return EXIT_OK
                time.sleep(a.poll)
                continue
            jid = job["id"]
            log(f"claimed {jid} kind={job['kind']} family={job['family']}"
                + (" (recovered lost claim response)" if job.get("replayed") else ""))
            state.update(state="busy", job=jid)
            q.status("busy", resident["h"].family, job=jid)
            try:
                with Heartbeat(q, jid, a.heartbeat) as hb:
                    if job["family"] not in classes:
                        raise ValueError(f"worker has no handler for family {job['family']!r}")
                    hb.progress = {"phase": "load", "family": job["family"]}
                    h = load(job["family"])
                    if on_cuda:
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    hb.progress = {"phase": "run"}
                    metrics = h.run(job, JobContext(q, job, hb)) or {}
                    metrics = {"run_s": round(time.time() - t0, 3), "peak_rss_gb": peak_rss_gb(),
                               "worker": a.name, **metrics}
                    if on_cuda:
                        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    hb.check()
                q.complete(jid, score=metrics.pop("score", None), metrics=metrics)
                log("done", jid, json.dumps(metrics, default=str)[:500])
            except LeaseLost as e:
                # The queue already requeued or finished this job elsewhere; the worker itself is healthy.
                log("abandoned", jid, "lease lost:", e.detail.get("error"))
                q.forget(jid)
            except Exception:
                tb = traceback.format_exc()
                log("FAILED", jid, tb)
                try:
                    q.fail(jid, tb[-4000:])
                    q.status("error", resident["h"].family if resident["h"] else None, {"last_error": tb[-1000:]})
                except Exception as e:
                    log("could not report failure (the queue requeues after missed heartbeats):", e)
                return EXIT_JOB_FAILED  # a clean process is safer than a GPU in an unknown state
            finally:
                clear_caches()
            state.update(state="ready", job=None)
            q.status("ready", resident["h"].family)
            idle_since = time.time()
            if a.once:
                return EXIT_OK
    except KeyboardInterrupt:
        log("interrupted")
        return EXIT_OK
    finally:
        stop.set()
        try:
            if state["state"] != "busy":
                q.status("stopped", resident["h"].family if resident["h"] else None)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
