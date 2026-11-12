"""DiLoCo job handler (stdlib only; the torch work runs in worker/diloco/inner.py as a subprocess).

  worker/supervise.py --gpus all -- --handler diloco=diloco.handler:DiLoCo     # one worker per session, all GPUs

Every job carries presigned object-store URLs, so weights never pass through the queue or the laptop:
  params = {"op": "init"|"train"|"reduce"|"export",
            "spec": {...},                       # for inner.py (recipe, cfg, round, replica, steps, ...)
            "get":  {"theta.pt": url, ...},      # downloaded into the work dir first
            "put":  {"delta.pt": url, ...},      # uploaded afterwards; must exist
            "put_optional": {"inner_state_next.pt": url}}   # uploaded if inner.py wrote it
Train jobs run under torchrun with one process per visible GPU (--opt nproc=N overrides; CPU: 1).
"""
import json, os, shutil, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

from client import LeaseLost
from runtime import Handler

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = 4


def gpu_count():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None:
        return len([x for x in v.split(",") if x.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.TimeoutExpired):
        return 0


def _retry(fn, what):
    for attempt in range(RETRY):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (408, 429):  # 403 = expired/bad URL: retrying won't help
                raise RuntimeError(f"{what}: HTTP {e.code} {e.read()[:300]!r}") from None
            err = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            err = e
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"{what} failed after {RETRY} tries: {err}")


def download(url, path):
    def go():
        with urllib.request.urlopen(url, timeout=120) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(path + ".part", path)
        return os.path.getsize(path)
    return _retry(go, f"download {os.path.basename(path)}")


def upload(url, path):
    size = os.path.getsize(path)

    def go():
        with open(path, "rb") as f:
            req = urllib.request.Request(url, data=f, method="PUT",
                                         headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"})
            urllib.request.urlopen(req, timeout=600).close()
        return size
    return _retry(go, f"upload {os.path.basename(path)}")


class DiLoCo(Handler):
    def run(self, job, ctx):
        p = ctx.params
        op = p.get("op")
        if op not in ("init", "train", "reduce", "export"):
            raise ValueError(f"unknown diloco op {op!r}")
        work = tempfile.mkdtemp(prefix=f"diloco-{op}-")
        try:
            return self._run(op, p, work, ctx)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run(self, op, p, work, ctx):
        t0 = time.time()
        down = 0
        for name, url in (p.get("get") or {}).items():
            ctx.progress(phase="download", file=name)
            down += download(url, os.path.join(work, name))
        t_down = time.time() - t0
        spec = {**(p.get("spec") or {}), "op": op, "work": work}
        if self.device == "cpu":
            spec["cpu"] = True
        if op == "export" and p.get("put", {}).get("adapter.tar.gz"):
            spec.setdefault("tar", True)
        with open(os.path.join(work, "spec.json"), "w") as f:
            json.dump(spec, f)

        nproc = int(self.options.get("nproc") or (gpu_count() if self.device != "cpu" else 1) or 1)
        script = os.path.join(HERE, "inner.py")
        if op == "train" and nproc > 1:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
                   script, os.path.join(work, "spec.json")]
        else:
            cmd = [sys.executable, script, os.path.join(work, "spec.json")]
        t1 = time.time()
        rc, tail = self._subprocess(cmd, work, ctx)
        if rc != 0:
            raise RuntimeError(f"diloco {op} exited {rc}:\n" + "".join(tail[-40:]))
        t_run = time.time() - t1

        if op == "export":  # pack the export dir for a single PUT
            shutil.make_archive(os.path.join(work, "adapter"), "gztar", os.path.join(work, "export"))
        t2 = time.time()
        up = 0
        for name, url in (p.get("put") or {}).items():
            path = os.path.join(work, name)
            if not os.path.exists(path):
                raise RuntimeError(f"diloco {op} did not write {name}")
            ctx.progress(phase="upload", file=name)
            up += upload(url, path)
        for name, url in (p.get("put_optional") or {}).items():
            if os.path.exists(os.path.join(work, name)):
                up += upload(url, os.path.join(work, name))
        result = {}
        rp = os.path.join(work, "result.json")
        if os.path.exists(rp):
            result = json.load(open(rp))
        return {**result, "op": op, "nproc": nproc if op == "train" else 1, "download_mb": round(down / 2**20, 2),
                "upload_mb": round(up / 2**20, 2), "download_s": round(t_down, 1), "run_s_inner": round(t_run, 1),
                "upload_s": round(time.time() - t2, 1)}

    def _subprocess(self, cmd, work, ctx):
        """Runs inner.py, streams its output, forwards progress.json to the heartbeat, and kills it if the
        lease is lost (job cancelled by the trainer, or requeued)."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(HERE), os.getcwd(), env.get("PYTHONPATH", "")])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)
        tail = []

        def pump():
            for line in proc.stdout:
                tail.append(line)
                del tail[:-200]
                print("  [inner]", line, end="", flush=True)
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        prog = os.path.join(work, "progress.json")
        try:
            while proc.poll() is None:
                time.sleep(2)
                fields = {"phase": "run"}
                try:
                    fields.update(json.load(open(prog)))
                except (OSError, ValueError):
                    pass
                ctx.progress(**fields)  # raises LeaseLost if the job was cancelled or requeued
        except (LeaseLost, KeyboardInterrupt):
            self._kill(proc)
            raise
        t.join(10)
        return proc.returncode, tail

    @staticmethod
    def _kill(proc):
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
