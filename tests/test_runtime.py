"""Worker runtime and supervisor against a real queue (CPU only, no torch needed).
Run: python3 tests/test_runtime.py

Covers: echo handler end to end (inputs, config, output, metrics), family swap with one resident handler,
a failing job (reported, exit 1), misconfiguration (exit 2), abandoning a job whose lease was lost, and the
supervisor's restart loop (crash -> restart -> retry -> fail after 2 attempts -> next job done).
"""
import json, os, subprocess, sys, tempfile, threading, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QUEUE_HEARTBEAT_TIMEOUT"] = "1.5"
os.environ["QUEUE_QUIET"] = "1"
_CONFIGS = tempfile.mkdtemp()
os.makedirs(os.path.join(_CONFIGS, "demo"))
open(os.path.join(_CONFIGS, "demo", "v1.json"), "w").write('{"greeting": "hello"}')
os.environ["QUEUE_CONFIGS"] = _CONFIGS
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker")]
import queue_server
from client import Queue

TOKEN = "r" * 32
ECHO = "echo=handlers.echo:Echo"


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = queue_server.serve(self.tmp, TOKEN, 0, reap_every=0.3)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.lap = Queue(self.url, TOKEN, "laptop")
        self.env = {**os.environ, "QUEUE_URL": self.url, "QUEUE_TOKEN": TOKEN, "CUDA_VISIBLE_DEVICES": ""}

    def tearDown(self):
        self.httpd.stop_reaper.set(); self.httpd.shutdown(); self.httpd.server_close()

    def worker(self, *args, token=TOKEN, timeout=60):
        env = {**self.env, "QUEUE_TOKEN": token}
        return subprocess.run([sys.executable, os.path.join(ROOT, "worker/runtime.py"), "--device", "cpu",
                               "--heartbeat", "0.5", "--poll", "0.2", "--name", "rt0", *args],
                              env=env, capture_output=True, text=True, timeout=timeout)

    def test_echo_end_to_end(self):
        jid = self.lap.submit("demo", "echo", {"text": "hi", "seconds": 0.5}, files={"in.txt": b"abc"},
                              config="demo@1")
        p = self.worker("--handler", ECHO, "--once")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        j = self.lap.job(jid)
        self.assertEqual(j["state"], "done")
        r = json.loads(self.lap.download(jid, "result.json"))
        self.assertEqual((r["text"], r["inputs"]["in.txt"]["bytes"], r["config"]), ("hi", 3, {"greeting": "hello"}))
        self.assertEqual((j["metrics"]["inputs"], j["metrics"]["worker"]), (1, "rt0"))
        self.assertGreater(j["metrics"]["run_s"], 0.4)
        self.assertEqual(self.lap.workers()[0]["state"], "stopped")

    def test_family_swap(self):
        jid = self.lap.submit("demo", "other", {"seconds": 0})
        p = self.worker("--handler", ECHO, "--handler", "other=handlers.echo:Echo", "--once")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("unloading echo for a other job", p.stdout)
        self.assertEqual(self.lap.job(jid)["state"], "done")

    def test_failing_job_is_reported_and_exits_1(self):
        jid = self.lap.submit("demo", "echo", {"fail": True})
        p = self.worker("--handler", ECHO, "--once")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        j = self.lap.job(jid)
        self.assertEqual((j["state"], j["attempts"]), ("queued", 1))
        self.assertIn("asked to fail", j["last_error"])
        self.assertEqual(self.lap.workers()[0]["state"], "error")

    def test_misconfiguration_exits_2(self):
        self.assertEqual(self.worker("--handler", ECHO, "--once", token="wrong" * 8).returncode, 2)
        self.assertEqual(self.worker("--handler", "echo=handlers.nope:X", "--once").returncode, 2)
        self.assertEqual(self.worker("--handler", "no-equals-sign", "--once").returncode, 2)

    def test_lost_lease_abandons_job(self):
        jid = self.lap.submit("demo", "echo", {"seconds": 6})
        # heartbeat every 3 s but the queue expires leases after 1.5 s: the job is requeued under the worker
        p = subprocess.run([sys.executable, os.path.join(ROOT, "worker/runtime.py"), "--device", "cpu",
                            "--heartbeat", "3", "--name", "slow", "--handler", ECHO, "--once"],
                           env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("lease lost", p.stdout)
        j = self.lap.job(jid)
        self.assertEqual((j["state"], j["attempts"]), ("queued", 1))
        self.assertNotIn("result.json", j["files"])

    def test_supervisor_restarts_after_crash(self):
        bad = self.lap.submit("demo", "echo", {"fail": True})
        good = self.lap.submit("demo", "echo", {"seconds": 0})
        t = time.time()
        p = subprocess.run([sys.executable, os.path.join(ROOT, "worker/supervise.py"), "--gpus", "cpu",
                            "--name", "sup", "--", "--handler", ECHO, "--heartbeat", "0.5", "--poll", "0.2",
                            "--idle-exit", "1"], env=self.env, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.lap.job(bad)["state"], "failed")   # 2 attempts, 2 crashes, 2 restarts
        self.assertEqual(self.lap.job(good)["state"], "done")
        self.assertEqual(p.stdout.count("sup-cpu: start"), 3)
        self.assertLess(time.time() - t, 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
