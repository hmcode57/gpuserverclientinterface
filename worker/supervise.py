"""Start one worker process per GPU and restart it when it crashes (the notebook's restart loop).

  python3 worker/supervise.py [--gpus auto|0,1|cpu] [--name NAME] [--minutes M] -- <worker/runtime.py args>

  python3 worker/supervise.py --gpus auto -- --handler echo=handlers.echo:Echo

Each process sees exactly one GPU (CUDA_VISIBLE_DEVICES) and is named <name>-gpu<i>, so the queue can show
and fence each GPU separately. With --gpus all there is one process, <name>-all, that sees every GPU (for
handlers that run torchrun themselves, like the DiLoCo trainer). Load is split by the queue: every worker pulls the next job it can run, so a
free GPU never waits for a busy one.

Restart policy, by worker exit code (see worker/runtime.py):
  0  done: not restarted        1  job failed: restarted after a backoff (5 s, doubling, max 60 s)
  2  misconfigured: not restarted, and the supervisor reports it
More than --max-restarts crashes within 10 minutes stops that GPU's worker (a crash loop burns GPU hours).
"""
import argparse, os, signal, socket, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(f"[supervise {time.strftime('%H:%M:%S')}]", *a, flush=True)


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [str(i) for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]


def run_gpu(gpu, name, worker_args, a, stop, results):
    restarts, backoff = [], 5.0
    while not stop.is_set():
        env = dict(os.environ)
        if gpu == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            extra = ["--device", "cpu"]
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu
            extra = []
        cmd = [sys.executable, "-u", os.path.join(HERE, "runtime.py"), "--name", name, *extra, *worker_args]
        log(f"{name}: start")
        p = subprocess.Popen(cmd, env=env)
        while p.poll() is None:
            if stop.wait(1):
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
        code = p.returncode
        results[name] = code
        if code == 0 or stop.is_set():
            log(f"{name}: exited {code}")
            return
        if code == 2:
            log(f"{name}: misconfigured (exit 2), not restarting; check QUEUE_URL, QUEUE_TOKEN and the handler args")
            return
        now = time.time()
        restarts = [t for t in restarts if now - t < 600] + [now]
        if len(restarts) > a.max_restarts:
            log(f"{name}: {len(restarts)} crashes in 10 minutes, giving up on this GPU")
            return
        log(f"{name}: exited {code}, restart {len(restarts)}/{a.max_restarts} in {backoff:.0f}s")
        if stop.wait(backoff):
            return
        backoff = min(backoff * 2, 60)


def main():
    argv = sys.argv[1:]
    worker_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, worker_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="auto", help="auto (one worker per visible GPU, else one CPU worker), 0,1, "
                    "cpu, or all (ONE worker that sees every GPU, for multi-GPU handlers)")
    ap.add_argument("--name", default=socket.gethostname(), help="worker name prefix")
    ap.add_argument("--minutes", type=float, default=0, help="stop all workers after this long (0 = never)")
    ap.add_argument("--max-restarts", type=int, default=5)
    a = ap.parse_args(argv)
    if not worker_args:
        ap.error("give the worker arguments after --, e.g. -- --handler echo=handlers.echo:Echo")
    for k in ("QUEUE_URL", "QUEUE_TOKEN"):
        if not os.environ.get(k):
            sys.exit(f"{k} is not set")
    if a.gpus == "all":  # one worker that sees every GPU (multi-GPU handlers, e.g. torchrun for DiLoCo)
        found = detect_gpus()
        gpus = [",".join(found)] if found else []
    else:
        gpus = detect_gpus() if a.gpus == "auto" else a.gpus.split(",")
    if not gpus:
        log("no GPU found, starting one CPU worker")
        gpus = ["cpu"]
    stop = threading.Event()
    results = {}
    threads = []
    for g in gpus:
        name = f"{a.name}-{'cpu' if g == 'cpu' else 'all' if ',' in g or a.gpus == 'all' else 'gpu' + g}"
        t = threading.Thread(target=run_gpu, args=(g, name, worker_args, a, stop, results), daemon=True)
        t.start()
        threads.append(t)
    deadline = time.time() + a.minutes * 60 if a.minutes else None
    try:
        while any(t.is_alive() for t in threads):
            if deadline and time.time() > deadline:
                log("time limit reached, stopping workers")
                stop.set()
            time.sleep(1)
    except KeyboardInterrupt:
        log("interrupted, stopping workers")
        stop.set()
    for t in threads:
        t.join(40)
    log("exit codes:", results)
    sys.exit(2 if any(c == 2 for c in results.values()) else 0)


if __name__ == "__main__":
    main()
