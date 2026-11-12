"""Queue client shared by workers, the laptop CLI and notebook tools. Stdlib only (runs on any notebook image).

Queue protocol 2: /claim returns a lease; this client remembers it per job and sends it with heartbeat,
output upload, complete and fail. Calls that are safe to repeat (claim with the same request_id, complete
and fail with the same lease, file up/downloads, heartbeats, status) are retried on transient network
errors, which are common over a quick tunnel. Job submission (POST /jobs) is never retried automatically.

URL discovery: quick-tunnel URLs change on every laptop restart. If QUEUE_URL_SOURCE is set (or `url_source`
is passed), a client that hits transport errors re-reads the current URL from it and switches over, so
running notebooks follow the laptop without being restarted. See resolve_url() for the accepted sources.
"""
import hashlib, http.client, io, json, os, re, secrets, socket, tempfile, threading, time, urllib.error, \
    urllib.request

PROTOCOL = 2
RETRY_HTTP = {429, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530}  # proxy/tunnel errors


URL_RE = re.compile(r"https?://[A-Za-z0-9.-]+(?::\d+)?(?:/[^\s\"']*)?")
RESOLVE_EVERY = float(os.environ.get("QUEUE_URL_RESOLVE_EVERY", "60"))  # min s between lookups per notebook


def resolve_url(source, timeout=20):
    """Current queue URL from `source`, or None if it can't be read. `source` is one of:
      - a local file path (e.g. run/url on the laptop, or a mounted dataset file)
      - an https URL whose body is the queue URL as plain text (a raw file, a paste, any static host)
      - a GitHub gist API URL, https://api.github.com/gists/<id> (always current; raw gist URLs are cached
        by GitHub's CDN for minutes). laptop/start.sh updates such a gist when QUEUE_URL_GIST is set.
    Lookups are cached for RESOLVE_EVERY seconds in a temp file shared by all processes on the machine, so
    several workers on one notebook make one request, which keeps GitHub's anonymous rate limit happy."""
    if not source:
        return None
    cache = os.path.join(tempfile.gettempdir(), "gpuq-url-" + hashlib.sha1(source.encode()).hexdigest()[:12])
    try:
        st = os.stat(cache)
        # only trust a cache file we wrote: another local user could otherwise redirect us (and our token)
        if time.time() - st.st_mtime < RESOLVE_EVERY and st.st_uid == getattr(os, "getuid", lambda: st.st_uid)():
            with open(cache) as f:
                return f.read().strip() or None
    except OSError:
        pass
    try:
        if "://" not in source:
            with open(source) as f:
                text = f.read()
        else:
            req = urllib.request.Request(source, headers={"User-Agent": "gpuq-client",
                                                          "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read(1 << 20).decode(errors="replace")
            if text.lstrip().startswith("{"):  # gist API: file "queue_url", else the first file
                files = json.loads(text).get("files") or {}
                f = files.get("queue_url") or next(iter(files.values()), {})
                text = f.get("content") or ""
        m = URL_RE.search(text)
        url = m.group(0).rstrip("/") if m else None
        if url and not (url.startswith("https://") or re.match(r"http://(127\.0\.0\.1|localhost)[:/]", url + "/")):
            url = None  # the token is sent to this URL: plain http only for the local machine
    except Exception:
        url = None
    try:  # cache misses too, so a broken source isn't hammered
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache))
        with os.fdopen(fd, "w") as f:
            f.write(url or "")
        os.replace(tmp, cache)
    except OSError:
        pass
    return url


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
    def __init__(self, base_url, token, worker, retries=4, backoff=1.0, url_source=None):
        self.url_source = url_source if url_source is not None else os.environ.get("QUEUE_URL_SOURCE")
        self.base = (base_url or resolve_url(self.url_source) or "").rstrip("/")
        self._resolved_at = 0.0
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
                if transient(err):
                    self.follow_url()
                if attempt < retries and transient(err):
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise err from None
            except Exception as e:
                if transient(e):
                    self.follow_url()
                if attempt < retries and transient(e):
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise

    def follow_url(self):
        """After a transport error: if a URL source is configured, switch to the URL it now names (the laptop
        restarted its tunnel). Returns True if the URL changed. Rate-limited; cheap to call often."""
        if not self.url_source or time.monotonic() - self._resolved_at < min(RESOLVE_EVERY, 30):
            return False
        self._resolved_at = time.monotonic()
        url = resolve_url(self.url_source)
        if url and url != self.base:
            print(f"[client] queue moved: {self.base} -> {url}", flush=True)
            self.base = url
            return True
        return False

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
        # the lease is required when this client uses the worker token, harmless otherwise
        h = {"X-Lease": self.leases[job_id]} if job_id in self.leases else None
        return self._req("GET", f"/jobs/{job_id}/files/{name}", headers=h, timeout=300, retries=self.retries)

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

    def cancel(self, job_id):
        """Producer token only: stop a queued or running job for good (its worker gets lease_lost)."""
        return self._req("POST", f"/jobs/{job_id}/cancel", {}, retries=self.retries)

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
