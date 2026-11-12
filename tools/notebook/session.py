"""Notebook session driver: run this inside the GPU notebook (one cell, see tools/notebook/notebook_cell.py).

Runs phases in order. After each phase it writes session.json and uploads it to the laptop immediately
(data/reports/<session>/), so results survive the notebook dying or hitting its time limit.

  preflight  nvidia-smi, CPU/RAM/disk, Python/torch/CUDA, internet, queue round trip, provider hints
  install    pip install -r <--requirements> (timed); never replaces the provider's CUDA torch build
  workers    worker/supervise.py: one worker per GPU, until --worker-minutes or the queue stays idle
  <name>     custom phases from --cmd name="shell command": run once per GPU in parallel (CUDA_VISIBLE_DEVICES
             set), logs uploaded. Use them for benchmarks, model downloads, smoke tests.

A background sampler records session RAM and per-GPU VRAM/utilisation/temperature/power every 2 s for the
whole run; each phase reports its own peaks. --budget-minutes stops starting new phases near the deadline
and keeps time to upload the reports.

  python tools/notebook/session.py --phases preflight,install,workers --requirements worker/requirements.txt \
      --worker-args "--handler echo=handlers.echo:Echo" --worker-minutes 30
Env: QUEUE_URL, QUEUE_TOKEN (from the provider's secrets).
"""
import argparse, json, os, platform, shlex, socket, subprocess, sys, threading, time, traceback, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "worker"))
from client import Queue  # stdlib only, safe before installs

BUILTIN = ("preflight", "install", "workers")


def sh_rc(cmd, timeout=600):
    """(returncode, output). Judge pip by exit code: it prints resolver warnings starting with "ERROR:" about
    unrelated preinstalled packages and still exits 0."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return -1, f"ERROR {e}"


def sh(cmd, timeout=600):
    rc, out = sh_rc(cmd, timeout)
    return out if rc == 0 else (out if out.startswith("ERROR ") else f"ERROR exit {rc}: {out}")


def last_line(out):
    """Errors from probes: keep only the final line (e.g. "ModuleNotFoundError: No module named 'torch'")."""
    return out.strip().splitlines()[-1] if out.startswith("ERROR") and out.strip() else out


def mem():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        d[k] = int(v.split()[0]) / 2**20
    return {"total_gb": round(d["MemTotal"], 2), "available_gb": round(d["MemAvailable"], 2),
            "used_gb": round(d["MemTotal"] - d["MemAvailable"], 2)}


def gpu_ids():
    out = sh("nvidia-smi -L", timeout=20)
    return [str(i) for i, _ in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]


class Session:
    def __init__(self, a):
        self.a = a
        self.id = a.session or time.strftime("nb-%Y%m%d-%H%M%S")
        self.out = os.path.abspath(a.out)
        os.makedirs(self.out, exist_ok=True)
        self.q = Queue(os.environ["QUEUE_URL"], os.environ["QUEUE_TOKEN"], f"{self.id}-driver")
        self.data = {"session": self.id, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "phases": {}}
        self.samples = []
        self._stop = threading.Event()
        self.deadline = time.monotonic() + a.budget_minutes * 60 if a.budget_minutes else None

    def remaining_s(self):
        return None if self.deadline is None else max(0, int(self.deadline - time.monotonic()))

    def bounded_timeout(self, normal, reserve=60):
        """Keep a phase inside the session budget while preserving time to upload the reports."""
        left = self.remaining_s()
        if left is None:
            return normal
        if left <= reserve:
            raise RuntimeError(f"time budget exhausted ({left}s remain; preserving {reserve}s for report upload)")
        return min(normal, left - reserve)

    def save(self):
        p = os.path.join(self.out, "session.json")
        json.dump(self.data, open(p, "w"), indent=2, default=str)
        self.upload("session.json", p)
        sp = os.path.join(self.out, "samples.jsonl")
        with open(sp, "w") as f:
            for s in list(self.samples):
                f.write(json.dumps(s) + "\n")
        self.upload("samples.jsonl", sp)

    def upload(self, name, path):
        try:
            self.q.report(self.id, name, open(path, "rb").read())
        except Exception as e:
            print(f"[session] upload {name} failed: {e}", flush=True)

    def log(self, *a):
        print(f"[session {time.strftime('%H:%M:%S')}]", *a, flush=True)

    # ------------------------------------------------------------- sampler
    def start_sampler(self):
        def run():
            t0 = time.time()
            last_up = 0
            while not self._stop.wait(2):
                s = {"t": round(time.time() - t0, 1), "ram": mem()}
                q = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,"
                       "clocks.sm --format=csv,noheader,nounits", timeout=10)
                gpus = []
                for line in q.splitlines():
                    try:
                        i, m, u, tmp, pw, clk = [x.strip() for x in line.split(",")]
                        gpus.append({"i": int(i), "mem_mb": float(m), "util": float(u), "temp": float(tmp),
                                     "power_w": float(pw) if pw not in ("[N/A]", "") else None, "sm_mhz": float(clk)})
                    except ValueError:
                        pass
                s["gpu"] = gpus
                self.samples.append(s)
                if time.time() - last_up > 60:
                    last_up = time.time()
                    self.save()
        threading.Thread(target=run, daemon=True).start()

    def peaks(self, since_t=0):
        rows = [s for s in list(self.samples) if s["t"] >= since_t]
        if not rows:
            return {}
        r = {"ram_used_peak_gb": max(s["ram"]["used_gb"] for s in rows),
             "ram_available_min_gb": min(s["ram"]["available_gb"] for s in rows)}
        for i in sorted({g["i"] for s in rows for g in s["gpu"]}):
            g = [x for s in rows for x in s["gpu"] if x["i"] == i]
            r[f"gpu{i}_mem_peak_mb"] = max(x["mem_mb"] for x in g)
            r[f"gpu{i}_util_mean"] = round(sum(x["util"] for x in g) / len(g), 1)
            r[f"gpu{i}_temp_max"] = max(x["temp"] for x in g)
            pw = [x["power_w"] for x in g if x["power_w"] is not None]
            if pw:
                r[f"gpu{i}_power_mean_w"] = round(sum(pw) / len(pw), 1)
            r[f"gpu{i}_sm_mhz_min"] = min(x["sm_mhz"] for x in g)
        return r

    def now_t(self):
        return self.samples[-1]["t"] if self.samples else 0

    # ------------------------------------------------------------- phases
    def run_phase(self, name, fn):
        self.log(f"==== phase {name}")
        t0, s0 = time.time(), self.now_t()
        try:
            r = {"ok": True, **(fn() or {})}
        except Exception as e:
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()[-4000:]}
        r["wall_s"] = round(time.time() - t0, 1)
        if self.remaining_s() is not None:
            r["budget_remaining_s"] = self.remaining_s()
        r["resources"] = self.peaks(s0)
        self.data["phases"][name] = r
        self.save()
        self.log(f"phase {name}: ok={r['ok']} wall={r['wall_s']}s {json.dumps(r.get('summary', ''))[:300]}")
        return r

    def preflight(self):
        r = {"hostname": socket.gethostname(), "python": sys.version.split()[0], "platform": platform.platform(),
             "cpus": os.cpu_count(), "ram": mem(),
             "nvidia_smi": sh("nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,"
                              "pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv"),
             "nvidia_smi_topo": sh("nvidia-smi topo -m"),
             "disk": {p: sh(f"df -h {p} | tail -1") for p in ("/", "/kaggle/working", "/content", "/tmp", os.getcwd())
                      if os.path.exists(p)},
             "torch": last_line(sh(f"{sys.executable} -c \"import torch;print(torch.__version__, torch.version.cuda, "
                                   f"torch.cuda.is_available(), torch.cuda.device_count())\"")),
             "preinstalled": sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE "
                                f"'^(torch|diffusers|transformers|accelerate|peft|numpy|xformers|flash|vllm)'"),
             "ffmpeg": sh("ffmpeg -version | head -1"),
             "env_hints": {k: v for k, v in os.environ.items()
                           if k.startswith(("KAGGLE", "COLAB", "PAPERSPACE", "SM_", "CUDA"))
                           and "TOKEN" not in k and "SECRET" not in k and "KEY" not in k},
             "cgroup_mem_limit": last_line(sh("cat /sys/fs/cgroup/memory.max 2>/dev/null || "
                                    "cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null"))}
        t = time.time()
        try:
            urllib.request.urlopen(self.a.probe_url, timeout=20).read(1)
            r["internet_s"] = round(time.time() - t, 2)
        except Exception as e:
            r["internet"] = f"no: {e}"
        t = time.time()
        r["queue_health"] = self.q._req("GET", "/health")
        r["queue_rtt_s"] = round(time.time() - t, 3)
        r["summary"] = {"python": r["python"], "gpus": len(gpu_ids()), "ram_gb": r["ram"]["total_gb"],
                        "torch": r["torch"], "queue_rtt_s": r["queue_rtt_s"]}
        return r

    def install(self):
        req = self.a.requirements
        if not req:
            return {"summary": "skipped: no --requirements"}
        req = req if os.path.isabs(req) else os.path.join(ROOT, req)
        before = sh(f"{sys.executable} -c 'import torch;print(torch.__version__)'")
        t = time.time()
        rc, out = sh_rc(f"{sys.executable} -m pip install -q -r {shlex.quote(req)}",
                        timeout=self.bounded_timeout(1800, reserve=180))
        r = {"pip_s": round(time.time() - t, 1), "pip_rc": rc, "pip_tail": out[-3000:]}
        after = sh(f"{sys.executable} -c 'import torch;print(torch.__version__)'")
        r["torch_before"], r["torch_after"] = before, after
        if before != after:
            r["warning"] = "pip changed the torch build: keep torch out of the requirements and use the provider's"
        if rc != 0:
            raise RuntimeError("pip install failed: " + out[-1500:])
        r["summary"] = {"pip_s": r["pip_s"], "torch": after}
        return r

    def workers(self):
        cmd = [sys.executable, "-u", os.path.join(ROOT, "worker", "supervise.py"), "--gpus", self.a.gpus,
               "--name", self.id]
        if self.a.worker_minutes:
            cmd += ["--minutes", str(self.a.worker_minutes)]
        cmd += ["--", *shlex.split(self.a.worker_args)]
        if self.a.worker_idle_exit:
            cmd += ["--idle-exit", str(self.a.worker_idle_exit)]
        logp = os.path.join(self.out, "workers.log")
        with open(logp, "w") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=self.bounded_timeout(10**7, reserve=90))
            except subprocess.TimeoutExpired:
                p.send_signal(2)  # SIGINT: supervisor stops workers cleanly
                code = p.wait(60)
        self.upload("workers.log", logp)
        done = sum(1 for line in open(logp) if "] done " in line)
        failed = sum(1 for line in open(logp) if "] FAILED " in line)
        if code == 2:
            raise RuntimeError("workers misconfigured (exit 2), see workers.log")
        return {"exit_code": code, "summary": {"jobs_done": done, "jobs_failed": failed}}

    def custom(self, name, command):
        def fn():
            gpus = gpu_ids() or [""]
            procs = []
            for g in gpus:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "SESSION_ID": self.id, "SESSION_OUT": self.out}
                logp = os.path.join(self.out, f"{name}_{'gpu' + g if g else 'cpu'}.log")
                log = open(logp, "w")
                procs.append((g, logp, log, subprocess.Popen(command, shell=True, cwd=ROOT, env=env,
                                                             stdout=log, stderr=subprocess.STDOUT)))
            codes = {}
            for g, logp, log, p in procs:
                try:
                    codes[g or "cpu"] = p.wait(timeout=self.bounded_timeout(10**7, reserve=90))
                except subprocess.TimeoutExpired:
                    p.kill(); codes[g or "cpu"] = "timeout"
                log.close()
                self.upload(os.path.basename(logp), logp)
            if any(c != 0 for c in codes.values()):
                raise RuntimeError(f"exit codes {codes}")
            return {"summary": {"exit_codes": codes}}
        return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", default="preflight,install,workers", help="comma list, in order")
    ap.add_argument("--cmd", action="append", default=[], help='custom phase: name="shell command" (repeatable)')
    ap.add_argument("--session")
    ap.add_argument("--out", default="session_out")
    ap.add_argument("--requirements", help="pip requirements file for the install phase (relative to the bundle)")
    ap.add_argument("--worker-args", default="--handler echo=handlers.echo:Echo", help="worker/runtime.py args")
    ap.add_argument("--worker-minutes", type=float, default=0, help="stop workers after this long (0 = budget)")
    ap.add_argument("--worker-idle-exit", type=float, default=0, help="each worker exits after this many idle s")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--probe-url", default="https://pypi.org/simple/")
    ap.add_argument("--budget-minutes", type=float, default=None, help="wall-clock budget for the whole session")
    a = ap.parse_args()
    for k in ("QUEUE_URL", "QUEUE_TOKEN"):
        if not os.environ.get(k):
            sys.exit(f"{k} is not set (use the provider's secrets, see docs/notebooks.md)")
    custom = {}
    for c in a.cmd:
        name, sep, command = c.partition("=")
        if not sep or not name or name in BUILTIN:
            ap.error(f"--cmd must look like name=command with a name other than {BUILTIN}: {c!r}")
        custom[name] = command
    phases = [p for p in a.phases.split(",") if p]
    unknown = [p for p in phases if p not in BUILTIN and p not in custom]
    if unknown:
        ap.error(f"unknown phases {unknown}: define them with --cmd name=...")
    s = Session(a)
    s.log(f"session {s.id}, results go to the laptop: data/reports/{s.id}/")
    s.start_sampler()
    fns = {"preflight": s.preflight, "install": s.install, "workers": s.workers,
           **{n: s.custom(n, c) for n, c in custom.items()}}
    for ph in phases:
        if s.remaining_s() is not None and s.remaining_s() <= 60:
            s.data["phases"][ph] = {"ok": False, "skipped": "time budget preserved for report upload",
                                    "budget_remaining_s": s.remaining_s()}
            s.save()
            s.log("stopping: time budget reached")
            break
        r = s.run_phase(ph, fns[ph])
        if not r["ok"] and ph == "install":
            s.log("stopping: install failed, later phases need it")
            break
    s._stop.set()
    s.data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    s.data["resources_total"] = s.peaks(0)
    s.save()
    s.log("finished. On the laptop: python3 tools/notebook/report.py data/reports/" + s.id)


if __name__ == "__main__":
    main()
