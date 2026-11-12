"""Queue hardening regression tests (protocol 2). Run: python3 tests/test_queue_hardening.py

Each test reproduces a failure seen against the pre-lease queue over real HTTP, then checks the fix:
  lost claim response, retried complete, stale worker reusing its name, concurrent same-name uploads,
  oversized/stalled bodies, path traversal, outdated clients, capability hints.
Tunnel-level behaviour (cloudflared) is not covered; dropped sockets on localhost stand in for it.
"""
import json, os, socket, sys, tempfile, threading, time, unittest, urllib.error
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QUEUE_HEARTBEAT_TIMEOUT"] = "1.5"
os.environ["QUEUE_QUIET"] = "1"
os.environ["QUEUE_MAX_FILE_MB"] = "8"
os.environ["QUEUE_READ_TIMEOUT"] = "1.5"
os.environ["QUEUE_CONFIGS"] = tempfile.mkdtemp()
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker")]
import queue_server
from client import Heartbeat, LeaseLost, Queue

TOKEN = "h" * 32
SPEC = dict(kind="test", family="echo", params={"n": 1}, outputs=["result.bin"])


class Hardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = queue_server.serve(self.tmp, TOKEN, 0, reap_every=0.2)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.lap = Queue(self.url, TOKEN, "laptop", retries=0)

    def tearDown(self):
        self.httpd.stop_reaper.set(); self.httpd.shutdown(); self.httpd.server_close()

    def worker(self, name, **kw):
        return Queue(self.url, TOKEN, name, backoff=0.05, **kw)

    def submit(self, **kw):
        return self.lap._req("POST", "/jobs", {**SPEC, **kw})["id"]

    def state(self, jid):
        return self.lap._req("GET", f"/jobs/{jid}")

    def raw(self, head, body=b"", read=True, wait=0.0):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall(head + body)
        if wait:
            time.sleep(wait)
        out = s.recv(4096) if read else b""
        s.close()
        return out

    def raw_claim(self, worker, request_id, read=False):
        body = json.dumps({"worker": worker, "family": "echo", "accept": ["echo"], "request_id": request_id,
                           "protocol": 2}).encode()
        head = (f"POST /claim HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
        return self.raw(head, body, read=read, wait=0 if read else 0.2)

    # ---------------------------------------------------------------- leases
    def test_lost_claim_response_is_recovered_by_retry(self):
        """Before: two dropped claim responses -> job failed with attempts=2, never ran."""
        jid = self.submit()
        self.raw_claim("w1", "req-lost-0001")              # claimed, response never read
        self.assertEqual(self.state(jid)["state"], "running")
        w = self.worker("w1")
        w._claim_req = "req-lost-0001"                      # what the client keeps after a lost answer
        j = w.claim("echo", ["echo"])
        self.assertEqual((j["id"], j["replayed"]), (jid, True))
        w.heartbeat(jid, {"step": 1})
        w.complete(jid, metrics={"ok": 1})
        s = self.state(jid)
        self.assertEqual((s["state"], s["attempts"]), ("done", 0))

    def test_client_reuses_request_id_after_transport_error(self):
        w = self.worker("w1", retries=0)
        jid = self.submit()
        real = Queue._req
        calls = []

        def flaky(self_, method, path, *a, **kw):
            if path == "/claim" and not calls:
                calls.append(1)
                real(self_, method, path, *a, **kw)        # server claims...
                raise ConnectionResetError("answer lost")  # ...but we never see it
            return real(self_, method, path, *a, **kw)
        Queue._req = flaky
        try:
            with self.assertRaises(ConnectionResetError):
                w.claim("echo", ["echo"])
            j = w.claim("echo", ["echo"])
        finally:
            Queue._req = real
        self.assertEqual((j["id"], j["replayed"]), (jid, True))

    def test_lost_complete_response_retry_is_idempotent(self):
        """Before: retried /complete -> 409 'job not running on this worker'."""
        jid = self.submit()
        w = self.worker("w2")
        lease = w.claim("echo", ["echo"])["lease"]
        w.complete(jid, metrics={"total_s": 1})
        w.leases[jid] = lease                               # retry exactly what was sent before
        self.assertEqual(w.complete(jid, metrics={"total_s": 1})["state"], "done")
        w.leases[jid] = lease
        with self.assertRaises(LeaseLost):                  # but a done job cannot be failed afterwards
            w.fail(jid, "late")
        self.assertEqual(self.state(jid)["state"], "done")

    def test_lost_fail_response_retry_counts_once(self):
        jid = self.submit()
        w = self.worker("w3")
        lease = w.claim("echo", ["echo"])["lease"]
        w.fail(jid, "boom")
        w.leases[jid] = lease
        w.fail(jid, "boom")
        s = self.state(jid)
        self.assertEqual((s["state"], s["attempts"]), ("queued", 1))

    def test_stale_process_with_same_name_is_fenced(self):
        """Before: old process (same worker name) completed the requeued run."""
        jid = self.submit()
        old = self.worker("session1-gpu0")
        old.claim("echo", ["echo"])
        time.sleep(2.2)                                     # heartbeat timeout: requeued, attempt 1
        self.assertEqual(self.state(jid)["state"], "queued")
        new = self.worker("session1-gpu0")                  # restarted notebook, same name
        self.assertEqual(new.claim("echo", ["echo"])["id"], jid)
        for call in (lambda: old.heartbeat(jid), lambda: old.upload(jid, "result.bin", b"stale"),
                     lambda: old.complete(jid), lambda: old.fail(jid, "stale")):
            with self.assertRaises(LeaseLost):
                call()
        self.assertEqual(self.state(jid)["state"], "running")
        new.upload(jid, "result.bin", b"fresh")
        new.complete(jid)
        self.assertEqual(open(os.path.join(self.tmp, "files", jid, "result.bin"), "rb").read(), b"fresh")

    def test_heartbeat_thread_reports_lease_loss(self):
        jid = self.submit()
        w = self.worker("w4")
        w.claim("echo", ["echo"])
        time.sleep(2.2)                                     # requeued under us
        with Heartbeat(w, jid, 0.1) as hb:
            self.assertTrue(hb.lost.wait(2))
            with self.assertRaises(LeaseLost):
                hb.check()

    def test_outputs_need_lease_inputs_need_queued(self):
        jid = self.submit()
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap.upload(jid, "result.bin", b"fake output")   # no lease: refused even while queued
        self.assertEqual(e.exception.code, 409)
        w = self.worker("w5")
        w.claim("echo", ["echo"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap.upload(jid, "input.bin", b"swap input mid-run")
        self.assertEqual(e.exception.code, 409)

    def test_lost_heartbeat_still_counts_attempts(self):
        """A lost heartbeat is one attempt; two -> failed."""
        jid = self.submit()
        for n in (1, 2):
            self.worker(f"dead{n}").claim("echo", ["echo"])
            time.sleep(2.2)
        s = self.state(jid)
        self.assertEqual((s["state"], s["attempts"]), ("failed", 2))
        self.assertIn("never sent a heartbeat", s["last_error"])

    # ------------------------------------------------------------ protocol
    def test_old_client_gets_426(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap._req("POST", "/claim", {"worker": "old", "family": "echo"})
        self.assertEqual(e.exception.code, 426)
        with self.assertRaises(urllib.error.HTTPError) as e:   # leaseless calls are rejected too
            self.lap._req("POST", f"/jobs/{self.submit()}/complete", {"worker": "old"})
        self.assertEqual(e.exception.code, 400)

    def test_waiting_for_explains_unclaimable_jobs(self):
        other = self.submit(family="llm")
        self.assertIn("no live worker accepts 'llm'", self.state(other)["waiting_for"])
        self.assertIsNone(self.worker("echo0").claim("echo", ["echo"]))
        self.assertIn("'llm'", self.state(other)["waiting_for"])  # an echo worker is live; llm still missing
        d = self.submit()
        self.assertIsNone(self.state(d)["waiting_for"])           # claimable: echo0 is live
        a = self.submit()
        b = self.submit(after=a)
        self.assertIn("dependency", self.state(b)["waiting_for"])
        w = self.worker("echo1")
        while True:
            j = w.claim("echo", ["echo"])
            if j["id"] == a:
                break
            w.complete(j["id"])
        w.fail(a, "x"); w.claim("echo", ["echo"]); w.fail(a, "x")
        self.assertEqual(self.state(a)["state"], "failed")
        self.assertIn("can never run", self.state(b)["waiting_for"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.submit(after=a)
        self.assertEqual(e.exception.code, 400)

    # --------------------------------------------------------- files/limits
    def test_concurrent_same_name_uploads(self):
        """Before: shared '.part' path -> handler crashed (FileNotFoundError) 27/30 times."""
        for _ in range(15):
            jid = self.submit()
            a, b = b"A" * 2_000_000, b"B" * 2_000_000
            errs = []

            def put(d):
                try:
                    Queue(self.url, TOKEN, "laptop", retries=0).upload(jid, "input.bin", d)
                except Exception as e:
                    errs.append(e)
            ts = [threading.Thread(target=put, args=(d,)) for d in (a, b)]
            [t.start() for t in ts]; [t.join() for t in ts]
            self.assertEqual(errs, [])
            self.assertIn(open(os.path.join(self.tmp, "files", jid, "input.bin"), "rb").read(), (a, b))
            self.assertEqual([f for f in os.listdir(os.path.join(self.tmp, "files", jid)) if f.endswith(".part")], [])

    def test_upload_limit_413_without_reading(self):
        """Before: 64 MiB accepted. Limit here is 8 MiB."""
        jid = self.submit()
        head = (f"PUT /jobs/{jid}/files/input.bin HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
                f"Content-Length: {64 * 2**20}\r\n\r\n").encode()
        t = time.time()
        out = self.raw(head)                               # body never sent: must still answer
        self.assertTrue(out.startswith(b"HTTP/1.1 413"), out[:80])
        self.assertLess(time.time() - t, 4)
        with self.assertRaises(ValueError):                # client refuses before sending
            self.lap.upload(jid, "input.bin", b"x" * (9 * 2**20))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "files", jid, "input.bin")))

    def test_json_body_limit(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.lap._req("POST", "/jobs", {**SPEC, "params": {"x": "x" * (2 * 2**20)}})
        self.assertEqual(e.exception.code, 413)

    def test_stalled_body_times_out_and_frees_thread(self):
        """Before: bogus Content-Length pinned a handler thread with no reply."""
        jid = self.submit()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=6)
        s.sendall((f"PUT /jobs/{jid}/files/input.bin HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
                   f"Content-Length: 1000000\r\n\r\npartial").encode())
        t = time.time()
        rest = s.recv(4096)                               # server gives up after QUEUE_READ_TIMEOUT
        self.assertLess(time.time() - t, 5)
        self.assertEqual(rest, b"")
        s.close()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "files", jid, "input.bin")))
        self.assertEqual([f for f in os.listdir(os.path.join(self.tmp, "files", jid)) if f.endswith(".part")], [])
        self.assertEqual(self.state(jid)["files"], {})
        self.assertEqual(self.lap._req("GET", "/health")["ok"], True)   # server still serving

    def test_unauthenticated_body_never_read(self):
        out = self.raw(b"PUT /jobs/x/files/y HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer nope\r\n"
                       b"Content-Length: 99999999999\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 401"), out[:80])

    def test_path_traversal_blocked(self):
        """Before: GET /jobs/../files/queue.db returned the SQLite database."""
        for path in ("/jobs/../files/queue.db", "/jobs/..%2F..%2Fqueue.db/files/x", "/jobs/./files/queue.db",
                     "/jobs/x/files/..", "/jobs/x/files/.hidden"):
            head = f"GET {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n\r\n".encode()
            out = self.raw(head)
            self.assertNotIn(b"SQLite format", out, path)
            self.assertRegex(out[:12].decode(), r"HTTP/1.1 (400|404)", path)

    def test_download_streams_large_file(self):
        jid = self.submit()
        data = os.urandom(6 * 2**20)
        self.lap.upload(jid, "input.bin", data)
        self.assertEqual(self.lap.download(jid, "input.bin"), data)

    def test_interrupted_upload_leaves_no_temp_file_after_restart(self):
        jid = self.submit()
        os.makedirs(os.path.join(self.tmp, "files", jid), exist_ok=True)
        open(os.path.join(self.tmp, "files", jid, ".input.bin.abc.part"), "wb").write(b"half")
        self.tearDown(); self.setUp_same_dir()
        self.assertEqual(os.listdir(os.path.join(self.tmp, "files", jid)), [])

    def setUp_same_dir(self):
        self.httpd = queue_server.serve(self.tmp, TOKEN, 0, reap_every=0.2)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.lap = Queue(self.url, TOKEN, "laptop", retries=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
