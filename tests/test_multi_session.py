"""Several notebooks (sessions, possibly on other people's accounts) against one queue.
Run: python3 tests/test_multi_session.py

Covers: the restricted worker token (can run jobs end to end, cannot list, read, submit or touch jobs it
holds no lease for), URL discovery after a tunnel move (plain file and GitHub-gist-style JSON sources), and
unique default session ids.
"""
import http.server, json, os, subprocess, sys, tempfile, threading, types, unittest, urllib.error
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QUEUE_HEARTBEAT_TIMEOUT"] = "2"
os.environ["QUEUE_QUIET"] = "1"
os.environ["QUEUE_URL_RESOLVE_EVERY"] = "0"  # no lookup cache/rate limit in tests
_CONFIGS = tempfile.mkdtemp()
os.makedirs(os.path.join(_CONFIGS, "demo"))
open(os.path.join(_CONFIGS, "demo", "v1.json"), "w").write('{"greeting": "hello"}')
os.environ["QUEUE_CONFIGS"] = _CONFIGS
sys.path[:0] = [os.path.join(ROOT, "laptop"), os.path.join(ROOT, "worker"), os.path.join(ROOT, "tools", "notebook")]
import queue_server
from client import Queue, resolve_url

TOKEN = "p" * 32
WTOKEN = "w" * 32


def dead_url():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    return f"http://127.0.0.1:{port}"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = queue_server.serve(self.tmp, TOKEN, 0, reap_every=0.5, worker_token=WTOKEN)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.lap = Queue(self.url, TOKEN, "laptop", url_source="")

    def tearDown(self):
        self.httpd.stop_reaper.set(); self.httpd.shutdown(); self.httpd.server_close()

    def code(self, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except urllib.error.HTTPError as e:
            return e.code
        return 200


class WorkerToken(Base):
    def test_worker_token_runs_a_job_but_cannot_read_others(self):
        jid = self.lap.submit("t", "echo", {"x": 1}, files={"in.txt": b"secret input"})
        other = self.lap.submit("t", "other", {}, files={"in.txt": b"not yours"})
        w = Queue(self.url, WTOKEN, "kaggleB-gpu0", url_source="")
        self.assertEqual(w._req("GET", "/health")["ok"], True)
        job = w.claim("echo", accept=["echo"])
        self.assertEqual(job["id"], jid)
        self.assertEqual(w.download(jid, "in.txt"), b"secret input")
        w.upload(jid, "out.txt", b"result")
        w.heartbeat(jid, {"p": 1})
        w.status("busy", "echo", job=jid)
        old_lease = w.lease(jid)
        w.complete(jid, metrics={"ok": 1})
        self.assertEqual(self.lap.job(jid)["state"], "done")
        self.assertEqual(self.lap.download(jid, "out.txt"), b"result")
        # forbidden: listing, reading, submitting, worker list, files of a job it holds no lease for
        self.assertEqual(self.code(w.jobs), 403)
        self.assertEqual(self.code(w.job, jid), 403)
        self.assertEqual(self.code(w.workers), 403)
        self.assertEqual(self.code(w._req, "POST", "/jobs", {"kind": "t", "family": "echo"}), 403)
        self.assertIn(self.code(w.download, other, "in.txt"), (400, 409))       # no lease
        self.assertEqual(self.code(w._req, "GET", f"/jobs/{jid}/files/in.txt",
                                   headers={"X-Lease": old_lease}), 409)       # lease ended (job done)
        self.assertEqual(self.code(w._req, "PUT", f"/jobs/{other}/files/x", raw=b"1"), 400)  # input upload
        # producer token unchanged, wrong token still 401
        self.assertEqual(len(self.lap.jobs()), 2)
        self.assertEqual(self.code(Queue(self.url, "x" * 32, "w", url_source="").jobs), 401)

    def test_runtime_with_worker_token(self):
        jid = self.lap.submit("greet", "echo", {"name": "B"}, config="demo@1", files={"input.txt": b"abc"})
        env = {**os.environ, "QUEUE_URL": self.url, "QUEUE_TOKEN": WTOKEN, "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([sys.executable, os.path.join(ROOT, "worker/runtime.py"), "--device", "cpu",
                            "--heartbeat", "0.5", "--poll", "0.2", "--name", "kaggleB-gpu0", "--once",
                            "--handler", "echo=handlers.echo:Echo"],
                           env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(self.lap.job(jid)["state"], "done", r.stdout + r.stderr)

    def test_same_tokens_refused(self):
        with self.assertRaises(ValueError):
            queue_server.serve(tempfile.mkdtemp(), TOKEN, 0, worker_token=TOKEN)


class UrlDiscovery(Base):
    def test_client_follows_url_file_after_tunnel_move(self):
        src = os.path.join(self.tmp, "url_source")
        open(src, "w").write(self.url + "\n")
        q = Queue(dead_url(), TOKEN, "kaggleA-gpu0", backoff=0.01, url_source=src)
        self.assertEqual(len(q.jobs()), 0)  # first attempt fails, follows the file, retry succeeds
        self.assertEqual(q.base, self.url)

    def test_heartbeat_style_call_switches_for_next_call(self):
        src = os.path.join(self.tmp, "url_source")
        open(src, "w").write(self.url)
        q = Queue(dead_url(), TOKEN, "a", url_source=src)
        with self.assertRaises(Exception):
            q._req("GET", "/health")  # retries=0: this call fails...
        self.assertTrue(q._req("GET", "/health")["ok"])  # ...but the next one uses the new URL

    def test_gist_api_json_source(self):
        live = self.url
        body = json.dumps({"files": {"README.md": {"content": "x"},
                                     "queue_url": {"content": live + "\n"}}}).encode()

        class G(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        g = http.server.HTTPServer(("127.0.0.1", 0), G)
        threading.Thread(target=g.serve_forever, daemon=True).start()
        try:
            src = f"http://127.0.0.1:{g.server_address[1]}/gists/abc"
            self.assertEqual(resolve_url(src), live)
            q = Queue("", TOKEN, "a", url_source=src)  # no URL at all: resolved at start
            self.assertEqual(q.base, live)
        finally:
            g.shutdown(); g.server_close()

    def test_unreadable_source_keeps_url(self):
        q = Queue(self.url, TOKEN, "a", url_source=os.path.join(self.tmp, "missing"))
        self.assertFalse(q.follow_url())
        self.assertEqual(q.base, self.url)


class SessionIds(unittest.TestCase):
    def test_default_session_ids_differ_within_one_second(self):
        os.environ.setdefault("QUEUE_URL", "http://127.0.0.1:9")
        os.environ.setdefault("QUEUE_TOKEN", TOKEN)
        import session
        a = types.SimpleNamespace(session=None, out=tempfile.mkdtemp(), budget_minutes=None)
        ids = {session.Session(a).id for _ in range(20)}
        self.assertEqual(len(ids), 20)
        a.session = "kaggleA"
        self.assertEqual(session.Session(a).id, "kaggleA")


if __name__ == "__main__":
    unittest.main(verbosity=1)
