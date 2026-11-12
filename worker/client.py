"""Queue client shared by workers, the laptop CLI and notebook tools. Stdlib only (runs on any notebook image).

Queue protocol 2: /claim returns a lease; this client remembers it per job and sends it with heartbeat,
output upload, complete and fail. Calls that are safe to repeat (claim with the same request_id, complete
and fail with the same lease, file up/downloads, heartbeats, status) are retried on transient network
errors, which are common over a quick tunnel. Job submission (POST /jobs) is never retried automatically.
"""
import http.client, io, json, secrets, socket, threading, time, urllib.error, urllib.request

PROTOCOL = 2
RETRY_HTTP = {429, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530}  # proxy/tunnel errors


class LeaseLost(urllib.error.HTTPError):
    """409 with code=lease_lost: the job was requeued or finished elsewhere; stop working on it."""


def transient(e):
    """True for failures worth retrying with the same idempotency key: the request never arrived, its
    answer was lost, or the tunnel/proxy answered instead of the queue."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code in RETRY_HTTP
    return isinstance(e, (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError,
                          socket.timeout))


class Queue:
    def __init__(self, base_url, token, worker, retries=4, backoff=1.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.worker = worker
        self.retries, self.backoff = retries, backoff
        self.leases = {}           # job id -> lease of the run this process holds
        self._claim_req = None     # request_id of a claim whose answer we never got (reused on next claim)
        self._max_file = None      # server's upload limit, read once from /health

    def _req(self, method, path, body=None, raw=None, headers=None, timeout=60, retries=0):
        h = {"Authorization": f"Bearer {self.token}", "User-Agent": f"gpuq-client/{PROTOCOL}"}
        data = None
        if raw is not None:
            data = raw
            h["Content-Type"] = "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    payload = r.read()
                    if r.status == 204 or not payload:
                        return None
                    ctype = r.headers.get("Content-Type", "")
                    return json.loads(payload) if "json" in ctype else payload
            except urllib.error.HTTPError as e:
                err = self._http_error(e)
                if attempt < retries and transient(err):
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise err from None
            except Exception as e:
                if attempt < retries and transient(e):
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise

    @staticmethod
    def _http_error(e):
        """Reads the JSON error body once, keeps it on `.detail`, and turns lost leases into LeaseLost."""
        try:
            body = e.read()
        except Exception:
            body = b""
        try:
            detail = json.loads(body) if body else {}
        except ValueError:
            detail = {"error": body[:200].decode(errors="replace")}
        cls = LeaseLost if e.code == 409 and detail.get("code") == "lease_lost" else urllib.error.HTTPError
        err = cls(e.url, e.code, detail.get("error") or e.msg, e.hdrs, io.BytesIO(body))
        err.detail = detail
        return err

    # ------------------------------------------------------------------ jobs
    def submit(self, kind, family, params=None, files=None, config=None, priority=0, after=None, outputs=None):
        """Creates a job, then uploads its input files ({name: bytes}). Returns the job id.
        Creating is not retried automatically (a lost answer would otherwise create the job twice)."""
        job = self._req("POST", "/jobs", {"kind": kind, "family": family, "params": params or {},
                                          "config": config, "priority": priority, "after": after,
                                          "outputs": outputs or []})
        for name, data in (files or {}).items():
            self.upload(job["id"], name, data)
        return job["id"]

    def job(self, job_id):
        return self._req("GET", f"/jobs/{job_id}", retries=self.retries)

    def jobs(self):
        return self._req("GET", "/jobs", retries=self.retries)

    def workers(self):
        return self._req("GET", "/workers", retries=self.retries)

    def claim(self, family, accept=None):
        """Claims the next job (or None). If the previous claim's answer was lost, this retries it with the
        same request_id, so the queue hands back that job instead of leaving it orphaned on us."""
        rid = self._claim_req or secrets.token_hex(12)
        self._claim_req = rid
        job = self._req("POST", "/claim", {"worker": self.worker, "family": family, "accept": accept,
                                           "request_id": rid, "protocol": PROTOCOL}, retries=self.retries)
        self._claim_req = None
        if job:
            self.leases[job["id"]] = job["lease"]
        return job

    def lease(self, job_id):
        return self.leases.get(job_id)

    def download(self, job_id, name):
        return self._req("GET", f"/jobs/{job_id}/files/{name}", timeout=300, retries=self.retries)

    def upload(self, job_id, name, data):
        """Raises ValueError locally if `data` exceeds the queue's limit (instead of sending it and having the
        connection cut, which would look like a network error and be retried)."""
        if self._max_file is None:
            try:
                self._max_file = int((self._req("GET", "/health", retries=self.retries) or {})
                                     .get("max_file_bytes") or 0)
            except Exception:
                self._max_file = 0  # unknown: let the server decide
        if self._max_file and len(data) > self._max_file:
            raise ValueError(f"{name} is {len(data)} bytes, queue limit is {self._max_file} "
                             "(raise QUEUE_MAX_FILE_MB on the laptop)")
        h = {"X-Worker": self.worker}
        if job_id in self.leases:
            h["X-Lease"] = self.leases[job_id]
        return self._req("PUT", f"/jobs/{job_id}/files/{name}", raw=data, headers=h, timeout=600,
                         retries=self.retries)

    def heartbeat(self, job_id, progress=None):
        # not retried: the next beat is the retry
        return self._req("POST", f"/jobs/{job_id}/heartbeat",
                         {"worker": self.worker, "lease": self.leases.get(job_id), "progress": progress})

    def complete(self, job_id, score=None, metrics=None):
        r = self._req("POST", f"/jobs/{job_id}/complete",
                      {"worker": self.worker, "lease": self.leases.get(job_id), "score": score,
                       "metrics": metrics}, retries=self.retries)
        self.leases.pop(job_id, None)
        return r

    def fail(self, job_id, error):
        r = self._req("POST", f"/jobs/{job_id}/fail",
                      {"worker": self.worker, "lease": self.leases.get(job_id), "error": error},
                      retries=self.retries)
        self.leases.pop(job_id, None)
        return r

    def forget(self, job_id):
        self.leases.pop(job_id, None)

    # --------------------------------------------------------------- workers
    def status(self, state, family=None, info=None, job=None):
        return self._req("POST", "/workers", {"worker": self.worker, "state": state, "family": family,
                                              "info": info, "job": job, "protocol": PROTOCOL},
                         retries=self.retries)

    def config(self, ref):
        """An immutable versioned config, e.g. 'example@1' -> configs/example/v1.json on the laptop."""
        return self._req("GET", f"/configs/{ref}", retries=self.retries)

    def report(self, session, name, data):
        """Upload a session report file to the laptop (data/reports/<session>/<name>)."""
        if isinstance(data, str):
            data = data.encode()
        return self._req("PUT", f"/reports/{session}/{name}", raw=data, timeout=600, retries=self.retries)


class Heartbeat:
    """Background thread: heartbeat every `interval` s with the latest progress. Never blocks the caller.
    If the queue says the lease is gone (job requeued after missed heartbeats, or finished elsewhere),
    `lost` is set and `check()` raises LeaseLost so the caller can stop wasting GPU time on it."""

    def __init__(self, q, job_id, interval=30.0):
        self.q, self.job_id, self.interval = q, job_id, interval
        self.progress = None
        self.lost = threading.Event()
        self.lost_error = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.q.heartbeat(self.job_id, self.progress)
            except LeaseLost as e:
                self.lost_error = e
                self.lost.set()
                print(f"[heartbeat] lease lost for {self.job_id}: {e.detail.get('error')}", flush=True)
                return
            except Exception as e:  # network blips must not kill the render
                print(f"[heartbeat] {e}", flush=True)
            self._stop.wait(self.interval)

    def check(self):
        if self.lost.is_set():
            raise self.lost_error

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
