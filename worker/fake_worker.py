"""Minimal protocol test worker: claims jobs, heartbeats, uploads a dummy result. Talks to the queue directly
(no runtime, no handlers), so it isolates queue/tunnel problems from worker problems.

QUEUE_URL=https://xxx.trycloudflare.com QUEUE_TOKEN=... python3 worker/fake_worker.py --name fake0
--hang      : claim one job then stop heartbeating (tests requeue)
--once      : process at most one job and exit
"""
import argparse, json, os, socket, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Heartbeat, LeaseLost, Queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"fake-{socket.gethostname()}-{os.getpid()}")
    ap.add_argument("--family", default="echo")
    ap.add_argument("--poll", type=float, default=3)
    ap.add_argument("--work-seconds", type=float, default=5)
    ap.add_argument("--heartbeat", type=float, default=30)
    ap.add_argument("--output-bytes", type=int, default=4096)
    ap.add_argument("--hang", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    q = Queue(os.environ["QUEUE_URL"], os.environ["QUEUE_TOKEN"], a.name)
    print(f"[{a.name}] ready family={a.family}", flush=True)
    while True:
        try:
            job = q.claim(a.family, accept=[a.family])
        except Exception as e:
            if getattr(e, "code", None) in (400, 401, 426):
                raise
            print(f"[{a.name}] claim failed, retrying: {e}", flush=True)
            time.sleep(a.poll)
            continue
        if not job:
            if a.once:
                return
            time.sleep(a.poll)
            continue
        jid = job["id"]
        print(f"[{a.name}] claimed {jid} kind={job['kind']}", flush=True)
        if a.hang:
            print(f"[{a.name}] hanging without heartbeat", flush=True)
            return
        try:
            for name in job["files"]:
                if name not in job["outputs"]:
                    q.download(jid, name)
            with Heartbeat(q, jid, a.heartbeat) as hb:
                steps = 4
                for s in range(steps):
                    time.sleep(a.work_seconds / steps)
                    hb.check()
                    hb.progress = {"step": s + 1, "steps": steps}
                head = json.dumps({"job": jid, "worker": a.name}).encode()
                q.upload(jid, "result.bin", head + b"\0" * max(0, a.output_bytes - len(head)))
            q.complete(jid, score=1.0)
            print(f"[{a.name}] done {jid}", flush=True)
        except LeaseLost as e:
            print(f"[{a.name}] abandoned {jid}: lease lost ({e.detail.get('error')})", flush=True)
            q.forget(jid)
        except Exception:
            try:
                q.fail(jid, traceback.format_exc())
            except Exception as e:
                print(f"[{a.name}] could not report failure: {e}", flush=True)
            raise  # exit; the restart loop brings up a clean worker
        if a.once:
            return


if __name__ == "__main__":
    main()
