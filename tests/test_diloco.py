"""DiLoCo trainer end to end on CPU, over real HTTP. Run: python3 tests/test_diloco.py  (needs torch; the
torch-free parts run without it)

Covers: SigV4 presigning against AWS's published example, the outer step (one replica with lr=1, mu=0 equals
plain training; Nesterov matches torch.optim.SGD), int8 delta error, job cancel, and a full run: queue +
object store + two supervised CPU workers (one training with 2-process DDP under torchrun) + the laptop
driver: init, rounds, a replica that fails and is retried and then dropped, resume after a driver restart,
export, and the held-out loss dropping.
"""
import datetime, http.server, json, os, shutil, subprocess, sys, tempfile, threading, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QUEUE_HEARTBEAT_TIMEOUT"] = "10"
os.environ["QUEUE_QUIET"] = "1"
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker")]
import objstore, queue_server  # noqa: E402
from client import LeaseLost, Queue  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

TOKEN = "d" * 32


class Presign(unittest.TestCase):
    def test_aws_example_signature(self):
        # https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
        u = objstore.presign("GET", "examplebucket.s3.amazonaws.com", "/test.txt", "AKIAIOSFODNN7EXAMPLE",
                             "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1", 86400,
                             now=datetime.datetime(2013, 5, 24, tzinfo=datetime.timezone.utc))
        self.assertTrue(u.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))

    def test_r2_url_shape(self):
        s = objstore.R2Store("acct", "AK", "SK", "bucket")
        u = s.put_url("diloco/run/r00001/delta 0.pt")
        self.assertTrue(u.startswith("https://acct.r2.cloudflarestorage.com/bucket/diloco/run/r00001/delta%200.pt?"))
        self.assertIn("auto%2Fs3%2Faws4_request", u)


class Cancel(unittest.TestCase):
    def test_cancel_running_job_fences_worker(self):
        tmp = tempfile.mkdtemp()
        httpd = queue_server.serve(tmp, TOKEN, 0, worker_token="w" * 32)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            lap, w = Queue(url, TOKEN, "lap", url_source=""), Queue(url, "w" * 32, "w0", url_source="")
            jid = lap.submit("t", "f", {})
            self.assertEqual(w.claim("f")["id"], jid)
            with self.assertRaises(Exception) as e:
                w.cancel(jid)
            self.assertEqual(e.exception.code, 403)  # workers can't cancel
            self.assertEqual(lap.cancel(jid)["state"], "cancelled")
            with self.assertRaises(LeaseLost):
                w.heartbeat(jid)
            self.assertIsNone(w.claim("f"))  # never requeued
        finally:
            httpd.stop_reaper.set(); httpd.shutdown(); httpd.server_close()


@unittest.skipIf(torch is None, "torch not installed")
class Core(unittest.TestCase):
    def setUp(self):
        from diloco import core
        self.core = core
        g = torch.Generator().manual_seed(0)
        self.theta = {"a": torch.randn(4, 6, generator=g), "b": torch.randn(3, generator=g)}

    def test_single_replica_plain_training(self):
        local = {n: t + 0.3 * torch.randn_like(t) for n, t in self.theta.items()}
        d = self.core.read_delta(self.core.make_delta(self.theta, local, bits=32))
        new, _, _ = self.core.outer_step(self.theta, [d], lr=1.0, mu=0.0)
        for n in local:
            self.assertTrue(torch.allclose(new[n], local[n], atol=1e-6))

    def test_nesterov_matches_torch_sgd(self):
        p = {n: torch.nn.Parameter(t.clone()) for n, t in self.theta.items()}
        opt = torch.optim.SGD(p.values(), lr=0.7, momentum=0.9, nesterov=True)
        theta, mom = dict(self.theta), None
        for step in range(3):
            g = {n: torch.randn_like(t) for n, t in self.theta.items()}
            for n in p:
                p[n].grad = g[n].clone()
            opt.step()
            theta, mom, _ = self.core.outer_step(theta, [g], mom, lr=0.7, mu=0.9)
        for n in p:
            self.assertTrue(torch.allclose(theta[n], p[n].detach(), atol=1e-5))

    def test_mean_of_replicas_and_int8_error(self):
        locals_ = [{n: t + 0.1 * torch.randn_like(t) for n, t in self.theta.items()} for _ in range(3)]
        ds = [self.core.read_delta(self.core.make_delta(self.theta, l)) for l in locals_]
        exact = [self.core.read_delta(self.core.make_delta(self.theta, l, bits=32)) for l in locals_]
        for d, e in zip(ds, exact):
            for n in d:
                err = (d[n] - e[n]).abs().max()
                self.assertLessEqual(float(err), float(e[n].abs().max()) / 127 * 0.51 + 1e-7)
        new, _, st = self.core.outer_step(self.theta, exact, lr=1.0, mu=0.0)
        mean_local = {n: sum(l[n] for l in locals_) / 3 for n in self.theta}
        for n in new:
            self.assertTrue(torch.allclose(new[n], mean_local[n], atol=1e-6))
        self.assertEqual(st["replicas"], 3)


class DirStore(http.server.BaseHTTPRequestHandler):
    root = None

    def _path(self):
        p = os.path.normpath(os.path.join(self.root, self.path.split("?")[0].lstrip("/")))
        assert p.startswith(self.root)
        return p

    def do_PUT(self):
        n = int(self.headers["Content-Length"])
        p = self._path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def do_DELETE(self):
        p = self._path()
        if os.path.exists(p):
            os.remove(p)
        self.send_response(204); self.end_headers()

    def do_GET(self):
        p = self._path()
        if not os.path.exists(p):
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        b = open(p, "rb").read()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


@unittest.skipIf(torch is None, "torch not installed")
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = queue_server.serve(os.path.join(self.tmp, "q"), TOKEN, 0, reap_every=0.5)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        root = os.path.join(self.tmp, "store")
        os.makedirs(root)
        DirStore.root = root
        self.store = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DirStore)
        threading.Thread(target=self.store.serve_forever, daemon=True).start()
        self.store_url = f"http://127.0.0.1:{self.store.server_address[1]}/bucket"
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.terminate()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        self.httpd.stop_reaper.set(); self.httpd.shutdown(); self.httpd.server_close()
        self.store.shutdown(); self.store.server_close()

    def worker(self, name, *opts):
        env = {**os.environ, "QUEUE_URL": self.url, "QUEUE_TOKEN": TOKEN, "CUDA_VISIBLE_DEVICES": "",
               "OMP_NUM_THREADS": "1"}
        log = open(os.path.join(self.tmp, f"{name}.log"), "w")
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, "worker/supervise.py"), "--gpus", "cpu",
                              "--name", name, "--", "--device", "cpu", "--heartbeat", "1", "--poll", "0.3",
                              "--handler", "diloco=diloco.handler:DiLoCo", *opts],
                             env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        self.procs.append(p)

    def driver(self, *extra, cfg=None):
        import diloco_train
        cfg = cfg or {"dim": 16, "rank": 4, "micro_batch": 32, "lr": 0.02, "warmup": 0,
                      "fail_at": [1, 1]}  # replica 1 fails in round 2 (round index 1)
        os.environ["QUEUE_TOKEN"] = TOKEN
        argv = ["--run", "toy", "--recipe", "diloco.recipes.toy", "--cfg", json.dumps(cfg), "--replicas", "2",
                "--steps", "15", "--store", self.store_url, "--url", self.url, "--data", os.path.join(self.tmp, "data"),
                "--poll", "0.3", "--min-replicas", "1", "--outer-lr", "1.0", "--outer-momentum", "0.5", *extra]
        diloco_train.main(argv)
        return cfg, json.load(open(os.path.join(self.tmp, "data", "diloco", "toy", "state.json")))

    def test_full_run(self):
        from diloco import core
        from diloco.recipes import toy
        self.worker("sessA", "--opt", "nproc=2")  # 2-process DDP (gloo) under torchrun
        self.worker("sessB")
        cfg, st = self.driver("--rounds", "2", "--no-export")
        self.assertEqual(st["round"], 2)
        self.assertEqual(st["history"][0]["replicas"], [0, 1])
        # replica 1 fails in round 2 on whichever worker runs it: it is resubmitted once and then dropped
        self.assertEqual(st["history"][1]["replicas"], [0])
        # resume: raise --rounds, same run continues from round 2 and exports
        cfg, st = self.driver("--rounds", "5")
        self.assertEqual(st["round"], 5)
        self.assertEqual([h["round"] for h in st["history"]], [1, 2, 3, 4, 5])
        run_dir = os.path.join(DirStore.root, "bucket", "diloco", "toy")
        torch.manual_seed(0)  # the init job's seed: rebuild theta_0 (it was deleted by --keep-rounds)
        theta0 = core.trainable_state(toy.load_model(cfg, "cpu"))
        theta5 = core.load(os.path.join(run_dir, "r00005", "theta.pt"))
        l0, l5 = toy.eval_loss(cfg, theta0), toy.eval_loss(cfg, theta5)
        self.assertLess(l5, l0 * 0.5, f"held-out loss {l0:.4f} -> {l5:.4f}")
        # inner AdamW state was carried between rounds; cleanup kept only what's still needed
        left = sorted(os.path.relpath(os.path.join(d, f), run_dir) for d, _, fs in os.walk(run_dir) for f in fs)
        self.assertEqual(left, ["r00004/inner_state_0.pt", "r00004/inner_state_1.pt", "r00004/momentum.pt",
                                "r00004/theta.pt", "r00005/adapter.tar.gz", "r00005/momentum.pt", "r00005/theta.pt"])
        exported = os.path.join(self.tmp, "data", "diloco", "toy", "adapter", "trainable_state.pt")
        self.assertTrue(os.path.exists(exported))
        self.assertTrue(torch.allclose(core.load(exported)["lora_B"], theta5["lora_B"]))
        print(f"\n  toy held-out loss: {l0:.4f} -> {l5:.4f}")

    def test_straggler_is_cancelled_and_its_worker_freed(self):
        self.worker("sessA")
        self.worker("sessB")
        cfg = {"dim": 16, "rank": 4, "micro_batch": 32, "lr": 0.02, "sleep_at": [1, 0, 300]}
        t0 = time.time()
        _, st = self.driver("--rounds", "1", "--no-export", "--straggler-timeout", "3", cfg=cfg)
        self.assertLess(time.time() - t0, 120)
        self.assertEqual(st["history"][0]["replicas"], [0])
        lap = Queue(self.url, TOKEN, "t", url_source="")
        slow = [j for j in lap.jobs() if j["kind"] == "diloco-train" and j["params"]["spec"]["replica"] == 1]
        self.assertEqual([j["state"] for j in slow], ["cancelled"])
        for _ in range(60):  # the stalled worker notices the lost lease, kills inner.py, and is ready again
            if all(w["state"] == "ready" and not w["job"] for w in lap.workers()):
                break
            time.sleep(0.5)
        self.assertTrue(all(w["state"] == "ready" and not w["job"] for w in lap.workers()), lap.workers())


if __name__ == "__main__":
    unittest.main(verbosity=2)
