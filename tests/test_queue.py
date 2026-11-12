"""Queue behaviour over real HTTP. Run: python3 tests/test_queue.py

Covers: fake worker end to end, 401 on wrong token, lost heartbeat requeues then fails, `after` gating,
atomic claim under concurrency, claim order (loaded family, priority, age), validation, restart survival,
worker status and configs, family filtering, metrics, bundle and reports, declared outputs.
The heartbeat timeout is shortened via QUEUE_HEARTBEAT_TIMEOUT so the tests run in seconds.
"""
import os, subprocess, sys, tempfile, threading, time, unittest, urllib.error
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QUEUE_HEARTBEAT_TIMEOUT"] = "2"
os.environ["QUEUE_QUIET"] = "1"
_CONFIGS = tempfile.mkdtemp()
os.makedirs(os.path.join(_CONFIGS, "demo"))
open(os.path.join(_CONFIGS, "demo", "v1.json"), "w").write('{"name": "demo", "version": 1}')
os.environ["QUEUE_CONFIGS"] = _CONFIGS
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker")]
import queue_server
from client import Queue

TOKEN = "t" * 32
SPEC = dict(kind="test", family="echo", params={"n": 1})


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.port = None
        self.start()
        self.lap = Queue(self.url, TOKEN, "laptop")

    def start(self):
        self.httpd = queue_server.serve(self.tmp, TOKEN, self.port or 0, reap_every=0.5)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.stop_reaper.set(); self.httpd.shutdown(); self.httpd.server_close()

    def tearDown(self):
        self.stop()

    def submit(self, **kw):
        return self.lap._req("POST", "/jobs", {**SPEC, **kw})

    def state(self, jid):
        return self.lap._req("GET", f"/jobs/{jid}")

    def test_fake_worker_end_to_end(self):
        jid = self.lap.submit("test", "echo", {"a": 1}, files={"input.bin": b"data"})
        env = {**os.environ, "QUEUE_URL": self.url, "QUEUE_TOKEN": TOKEN}
        subprocess.run([sys.executable, os.path.join(ROOT, "worker/fake_worker.py"), "--once",
                        "--work-seconds", "0.5", "--name", "fake0"], env=env, check=True)
        s = self.state(jid)
        self.assertEqual(s["state"], "done")
        self.assertEqual(s["params"], {"a": 1})
        self.assertEqual(os.path.getsize(os.path.join(self.tmp, "files", jid, "result.bin")), 4096)
        self.assertEqual(s["outputs"], ["result.bin"])  # written with the lease -> recorded as output

    def test_wrong_token_401(self):
        for tok in ("wrong", ""):
            with self.assertRaises(urllib.error.HTTPError) as e:
                Queue(self.url, tok, "x")._req("GET", "/jobs")
            self.assertEqual(e.exception.code, 401)

    def test_lost_heartbeat_requeues_then_fails(self):
        j = self.submit()
        w = Queue(self.url, TOKEN, "w1")
        self.assertEqual(w.claim("echo")["id"], j["id"])
        time.sleep(3.5)
        s = self.state(j["id"])
        self.assertEqual((s["state"], s["attempts"]), ("queued", 1))
        with self.assertRaises(urllib.error.HTTPError) as e:  # the stale worker can no longer touch it
            w.complete(j["id"])
        self.assertEqual(e.exception.code, 409)
        w.claim("echo"); w.fail(j["id"], "boom")
        s = self.state(j["id"])
        self.assertEqual((s["state"], s["last_error"]), ("failed", "boom"))

    def test_heartbeat_keeps_job(self):
        j = self.submit()
        w = Queue(self.url, TOKEN, "w1"); w.claim("echo")
        for _ in range(6):
            time.sleep(0.6); w.heartbeat(j["id"], {"step": 1})
        self.assertEqual(self.state(j["id"])["state"], "running")

    def test_after_gating(self):
        a = self.submit()
        b = self.submit(after=a["id"])
        w = Queue(self.url, TOKEN, "w1"); w2 = Queue(self.url, TOKEN, "w2")
        self.assertEqual(w.claim("echo")["id"], a["id"])
        self.assertIsNone(w2.claim("echo"))
        w.complete(a["id"])
        self.assertEqual(w2.claim("echo")["id"], b["id"])

    def test_atomic_claim(self):
        ids = {self.submit()["id"] for _ in range(20)}
        got, lock = [], threading.Lock()

        def grab(n):
            q = Queue(self.url, TOKEN, f"w{n}")
            while (j := q.claim("echo")):
                with lock: got.append(j["id"])
        ts = [threading.Thread(target=grab, args=(n,)) for n in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(sorted(got), sorted(ids))

    def test_claim_order(self):
        old = self.submit(family="a")
        urgent = self.submit(family="a", priority=5)
        other = self.submit(family="b")
        w = Queue(self.url, TOKEN, "w")
        self.assertEqual(w.claim("b")["id"], other["id"])   # the loaded family first (no model swap)
        self.assertEqual(w.claim("a")["id"], urgent["id"])  # then higher priority
        self.assertEqual(w.claim("a")["id"], old["id"])     # then oldest

    def test_validation(self):
        for bad in ({"kind": ""}, {"kind": "a b"}, {"family": None}, {"family": "x" * 65}, {"params": [1]},
                    {"priority": "high"}, {"outputs": ["../x"]}, {"outputs": "x"},
                    {"config": "demo"}, {"config": "demo@2"}, {"config": "../x@1"}, {"after": "nope"}):
            with self.assertRaises(urllib.error.HTTPError, msg=bad) as e:
                self.submit(**bad)
            self.assertEqual(e.exception.code, 400, bad)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap.upload(self.submit()["id"], "../x", b"")
        self.assertIn(e.exception.code, (400, 404))

    def test_declared_outputs_need_the_lease(self):
        j = self.submit(outputs=["out.json"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap.upload(j["id"], "out.json", b"forged")
        self.assertEqual(e.exception.code, 409)
        self.lap.upload(j["id"], "in.json", b"{}")           # undeclared names are inputs
        w = Queue(self.url, TOKEN, "w"); w.claim("echo")
        w.upload(j["id"], "out.json", b"real"); w.complete(j["id"])
        s = self.state(j["id"])
        self.assertEqual((s["outputs"], sorted(s["files"])), (["out.json"], ["in.json", "out.json"]))

    def test_survives_restart(self):
        j = self.submit(priority=3, params={"k": "v"})
        self.lap.upload(j["id"], "input.bin", b"d")
        self.stop(); self.start()
        s = self.state(j["id"])
        self.assertEqual((s["state"], list(s["files"]), s["priority"], s["params"]),
                         ("queued", ["input.bin"], 3, {"k": "v"}))

    def test_workers_and_configs(self):
        w = Queue(self.url, TOKEN, "g0")
        w.status("ready", "echo", {"gpu": "T4"})
        w.status("busy", "echo", job="x")  # empty info keeps the startup info
        ws = {r["name"]: r for r in self.lap.workers()}
        self.assertEqual((ws["g0"]["state"], ws["g0"]["job"], ws["g0"]["info"]), ("busy", "x", {"gpu": "T4"}))
        time.sleep(2.5)  # past the (test) heartbeat timeout
        self.assertEqual(self.lap.workers()[0]["state"], "gone")
        with self.assertRaises(urllib.error.HTTPError) as e:
            w.status("dancing")
        self.assertEqual(e.exception.code, 400)
        self.assertEqual(w.config("demo@1")["version"], 1)
        self.assertEqual(self.submit(config="demo@1")["config"], "demo@1")
        with self.assertRaises(urllib.error.HTTPError) as e:
            w.config("demo@9")
        self.assertEqual(e.exception.code, 404)

    def test_accept_filter_and_metrics(self):
        llm = self.submit(family="llm")
        e = self.submit()
        w = Queue(self.url, TOKEN, "echoonly")
        self.assertEqual(w.claim("echo", ["echo"])["id"], e["id"])
        self.assertIsNone(w.claim("echo", ["echo"]))  # the llm job is left for an llm-capable worker
        w.complete(e["id"], metrics={"run_s": 1.5})
        self.assertEqual(self.state(e["id"])["metrics"], {"run_s": 1.5})
        self.assertEqual(Queue(self.url, TOKEN, "any").claim("echo")["id"], llm["id"])  # accept=None: any family
        with self.assertRaises(urllib.error.HTTPError) as err:
            w.claim("echo", [])
        self.assertEqual(err.exception.code, 400)

    def test_bundle_and_reports(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap._req("GET", "/bundle")
        self.assertEqual(e.exception.code, 404)
        open(os.path.join(self.tmp, "bundle.tar.gz"), "wb").write(b"tgz")
        self.assertEqual(self.lap._req("GET", "/bundle"), b"tgz")
        w = Queue(self.url, TOKEN, "nb")
        w.report("sess1", "results.json", '{"a": 1}')
        self.assertEqual(open(os.path.join(self.tmp, "reports/sess1/results.json")).read(), '{"a": 1}')
        for bad in ("../x", ".hidden"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                w.report(bad, "r.json", "x")
            self.assertIn(e.exception.code, (400, 404))
        with self.assertRaises(urllib.error.HTTPError) as e:
            Queue(self.url, "wrong", "nb").report("s", "r.json", "x")
        self.assertEqual(e.exception.code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
