"""Durable job queue for remote GPU workers. Stdlib only (Python 3.9+), so nothing to install or pin.

Run: QUEUE_TOKEN=... [QUEUE_WORKER_TOKEN=...] python3 laptop/queue_server.py [--port 8787] [--data ./data]
Binds 127.0.0.1 only; expose with `cloudflared tunnel --url http://127.0.0.1:8787` (laptop/start.sh does this).

Jobs are generic: `kind` (what to do), `family` (which loaded model/runtime can do it), free-form `params` JSON,
optional `config` (an immutable, versioned settings file name@version), `priority`, `after` (dependency) and
`outputs` (file names only the worker holding the job's lease may write). Input files are uploaded while the
job is queued. The queue never interprets params; workers do.

Leases (queue protocol 2): every claim returns a lease token. Heartbeat, output upload, complete and fail
must carry it, so a worker whose job was requeued (stale process, healed network partition, reused worker
name) can no longer touch the new run. This fences stale processes; it is not an authorization boundary
(every holder of the producer token can still read jobs).
Tokens: QUEUE_TOKEN (producer) can do everything. The optional QUEUE_WORKER_TOKEN is for notebooks you don't
fully trust (e.g. other people's accounts): it can claim jobs, read the inputs of the job it holds a lease
for, upload outputs, heartbeat/complete/fail, report status and session reports, and fetch the bundle and
configs, but it cannot list or read jobs, submit, or see other jobs' files.
Retry safety: a claim retried with the same request_id returns the same job and lease; complete/fail
retried with the same lease return the stored result. A lost heartbeat still counts as an attempt
; the client recovers a lost claim response by retrying with the same request_id.
Limits: request bodies are capped (QUEUE_MAX_FILE_MB for file uploads, 1 MiB for JSON) and refused before
they are read; idle or stalled sockets time out after QUEUE_READ_TIMEOUT seconds.
"""
import argparse, hmac, json, os, re, secrets, shutil, socket, sqlite3, sys, tempfile, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 170 s timeout + 10 s reaper interval: a dead job is back in `queued` within 180 s worst case.
HEARTBEAT_TIMEOUT = float(os.environ.get("QUEUE_HEARTBEAT_TIMEOUT", 170))
MAX_ATTEMPTS = int(os.environ.get("QUEUE_MAX_ATTEMPTS", 2))
MIN_PROTOCOL = 2                 # workers older than this get 426 on /claim
MAX_FILE_BYTES = int(float(os.environ.get("QUEUE_MAX_FILE_MB", 512)) * 2**20)
MAX_JSON_BYTES = 2**20
READ_TIMEOUT = float(os.environ.get("QUEUE_READ_TIMEOUT", 60))
CONFIG_DIR = Path(os.environ.get("QUEUE_CONFIGS", Path(__file__).resolve().parent.parent / "configs"))
MAX_REPORT_BYTES = 200 * 1024 * 1024
LIVE_STATES = ("starting", "ready", "busy")
FILE_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # job ids: no dots, so never '.' or '..' in a file path
REQ_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")    # job kinds and model families
CONFIG_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}@[0-9]{1,6}$")
CHUNK = 2**20
LEASE_KEEP_S = 7 * 86400         # ended leases are kept this long so late retries get a clear answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, family TEXT NOT NULL, config TEXT,
  params TEXT NOT NULL DEFAULT '{}', priority INTEGER NOT NULL DEFAULT 0, after TEXT,
  outputs TEXT NOT NULL DEFAULT '[]', state TEXT NOT NULL DEFAULT 'queued',
  attempts INTEGER NOT NULL DEFAULT 0, worker TEXT, heartbeat REAL, lease TEXT,
  progress TEXT, score REAL, metrics TEXT, last_error TEXT, files TEXT NOT NULL DEFAULT '{}',
  created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS workers (
  name TEXT PRIMARY KEY, state TEXT NOT NULL, family TEXT, info TEXT NOT NULL DEFAULT '{}',
  job TEXT, seen REAL NOT NULL, accept TEXT, protocol INTEGER);
CREATE TABLE IF NOT EXISTS leases (
  lease TEXT PRIMARY KEY, job TEXT NOT NULL, worker TEXT NOT NULL, claim_req TEXT,
  acked INTEGER NOT NULL DEFAULT 0, outcome TEXT, created REAL NOT NULL, ended REAL);
CREATE INDEX IF NOT EXISTS leases_req ON leases (worker, claim_req);
"""
# columns added after the first release go here ({table: {column: declaration}}); older databases are
# migrated in place at startup
MIGRATIONS = {}


class Conflict(Exception):
    """409. `code` lets clients tell a lost lease apart from other conflicts."""

    def __init__(self, msg, code="conflict", **extra):
        super().__init__(msg)
        self.code, self.extra = code, extra


class UpgradeRequired(Exception):
    pass


class TooLarge(Exception):
    pass


class LengthRequired(Exception):
    pass


def config_path(ref):
    """'name@3' -> configs/name/v3.json. Configs are immutable: a change is a new version file, so a retried
    or repeated job always runs with exactly the settings it was submitted with."""
    name, ver = ref.split("@")
    return CONFIG_DIR / name / f"v{ver}.json"


def new_id():
    return f"{time.time_ns():020d}-{secrets.token_hex(3)}"


def atomic_write(path, data):
    """Unique temp file in the target dir, fsync, rename. Concurrent writers never share a temp path."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Store:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        for stray in (self.dir / "files").glob("*/.*.part"):  # uploads interrupted by a crash
            stray.unlink(missing_ok=True)
        self.db = sqlite3.connect(self.dir / "queue.db", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.lock = threading.RLock()  # one shared connection: every access goes through this lock

    # ------------------------------------------------------------------ reads
    def _raw(self, job_id):
        with self.lock:
            r = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def row(self, job_id):
        d = self._raw(job_id)
        if not d:
            return None
        d.pop("lease", None)  # handed out only by /claim
        d["files"] = json.loads(d["files"])
        d["params"] = json.loads(d["params"])
        d["outputs"] = json.loads(d["outputs"])
        d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else None
        d["progress"] = json.loads(d["progress"]) if d.get("progress") else None
        return d

    def annotated(self, rows):
        """Adds `waiting_for` to queued jobs: why nobody is running them yet (None = claimable now)."""
        live = self._live_families()
        with self.lock:
            deps = {r["id"]: r["state"] for r in self.db.execute(
                "SELECT id, state FROM jobs WHERE id IN (SELECT after FROM jobs WHERE after IS NOT NULL)")}
        for r in rows:
            r["waiting_for"] = None
            if r["state"] != "queued":
                continue
            dep = deps.get(r["after"]) if r["after"] else "done"
            if dep == "failed":
                r["waiting_for"] = f"blocked: dependency {r['after']} failed, so this job can never run"
            elif dep != "done":
                r["waiting_for"] = f"dependency {r['after']} is {dep}"
            elif r["family"] not in live and "*" not in live:
                r["waiting_for"] = f"no live worker accepts '{r['family']}' jobs"
        return rows

    def job(self, job_id):
        r = self.row(job_id)
        if not r:
            raise KeyError(job_id)
        return self.annotated([r])[0]

    def list(self):
        with self.lock:
            ids = [r["id"] for r in self.db.execute("SELECT id FROM jobs ORDER BY id")]
        return self.annotated([self.row(i) for i in ids])

    # ---------------------------------------------------------------- workers
    def worker_status(self, name, state, fam, info, job, protocol=None):
        with self.lock:
            self.db.execute(
                "INSERT INTO workers (name,state,family,info,job,seen,protocol) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET state=excluded.state, family=excluded.family, "
                "info=CASE WHEN excluded.info='{}' THEN workers.info ELSE excluded.info END, "
                "job=excluded.job, seen=excluded.seen, protocol=COALESCE(excluded.protocol, workers.protocol)",
                (name, state, fam, json.dumps(info or {}), job, time.time(), protocol))

    def _touch_worker(self, name, fam, accept, protocol):
        """Every /claim poll counts as a sign of life and records what the worker can run."""
        self.db.execute(
            "INSERT INTO workers (name,state,family,info,job,seen,accept,protocol) "
            "VALUES (?,'ready',?,'{}',NULL,?,?,?) ON CONFLICT(name) DO UPDATE SET seen=excluded.seen, "
            "accept=excluded.accept, protocol=excluded.protocol, family=COALESCE(excluded.family, workers.family)",
            (name, fam, time.time(), json.dumps(accept) if accept is not None else None, protocol))

    def workers(self):
        with self.lock:
            rows = [dict(r) for r in self.db.execute("SELECT * FROM workers ORDER BY name")]
        now = time.time()
        for r in rows:
            r["info"] = json.loads(r["info"])
            r["accept"] = json.loads(r["accept"]) if r.get("accept") else None
            r["age_s"] = round(now - r["seen"], 1)
            if r["age_s"] > HEARTBEAT_TIMEOUT:
                r["state"] = "gone"
        return rows

    def _live_families(self):
        """Families some live worker accepts; '*' = a live worker accepts any family."""
        fams = set()
        for w in self.workers():
            if w["state"] in LIVE_STATES and (w.get("protocol") or 0) >= MIN_PROTOCOL:
                fams.update(w["accept"] if w["accept"] is not None else ["*"])
        return fams

    # ------------------------------------------------------------------- jobs
    def create(self, spec):
        for k in ("kind", "family"):
            if not isinstance(spec.get(k), str) or not NAME_RE.match(spec[k]):
                raise ValueError(f"{k} is required: 1-64 characters of A-Z a-z 0-9 . _ -")
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
        outputs = spec.get("outputs", [])
        if not isinstance(outputs, list) or not all(isinstance(o, str) and FILE_RE.match(o) and o[0] != "."
                                                    for o in outputs):
            raise ValueError("outputs must be a list of file names")
        try:
            priority = int(spec.get("priority", 0))
        except (TypeError, ValueError):
            raise ValueError("priority must be an integer") from None
        config = spec.get("config") or None
        if config is not None:
            if not isinstance(config, str) or not CONFIG_RE.match(config):
                raise ValueError("config must look like name@version")
            if not config_path(config).is_file():
                raise ValueError(f"config {config} not found in {CONFIG_DIR}")
        after = spec.get("after") or None
        if after:
            dep = self._raw(after) if isinstance(after, str) and ID_RE.match(after) else None
            if not dep:
                raise ValueError("after refers to unknown job")
            if dep["state"] == "failed":
                raise ValueError(f"after refers to failed job {after}; this job could never run")
        now = time.time()
        jid = new_id()
        with self.lock:
            self.db.execute(
                "INSERT INTO jobs (id,kind,family,config,params,priority,after,outputs,created,updated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (jid, spec["kind"], spec["family"], config, json.dumps(params), priority, after,
                 json.dumps(sorted(set(outputs))), now, now))
        return self.job(jid)

    def claim(self, worker, loaded, accept=None, request_id=None, protocol=None):
        """loaded: the family this worker has resident now (preferred, so models are not swapped needlessly).
        accept: families this worker can run (None = any). Jobs of other families stay queued for others.
        request_id: retrying a claim with the same id returns the same job and lease instead of a new one."""
        try:
            protocol = int(protocol)
        except (TypeError, ValueError):
            protocol = 0
        if protocol < MIN_PROTOCOL:
            raise UpgradeRequired(f"queue protocol {MIN_PROTOCOL} required (worker sent {protocol or 'none'}); "
                                  "update worker/client.py on this session")
        if request_id is not None and not (isinstance(request_id, str) and REQ_RE.match(request_id)):
            raise ValueError("bad request_id")
        if accept is not None and not (isinstance(accept, list) and accept
                                       and all(isinstance(f, str) and NAME_RE.match(f) for f in accept)):
            raise ValueError("accept must be a non-empty list of family names (or null for any)")
        fam_filter = f"AND j.family IN ({','.join('?' * len(accept))})" if accept is not None else ""
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self._touch_worker(worker, loaded, accept, protocol)
                if request_id:
                    prev = self.db.execute(
                        "SELECT l.lease, l.job FROM leases l JOIN jobs j ON j.id = l.job "
                        "WHERE l.worker=? AND l.claim_req=? AND l.outcome IS NULL "
                        "AND j.state='running' AND j.lease = l.lease", (worker, request_id)).fetchone()
                    if prev:
                        self.db.execute("COMMIT")
                        return {**self.row(prev["job"]), "lease": prev["lease"], "replayed": True}
                # order: the family already loaded on this worker, then priority, then oldest
                r = self.db.execute(
                    f"""SELECT j.id FROM jobs j LEFT JOIN jobs d ON d.id = j.after
                       WHERE j.state='queued' AND (j.after IS NULL OR d.state='done') {fam_filter}
                       ORDER BY (CASE WHEN j.family = ? THEN 0 ELSE 1 END), j.priority DESC, j.id LIMIT 1""",
                    (*(accept or []), loaded or "")).fetchone()
                if not r:
                    self.db.execute("COMMIT")
                    return None
                now, lease = time.time(), secrets.token_urlsafe(18)
                self.db.execute(
                    "UPDATE jobs SET state='running', worker=?, heartbeat=?, progress=NULL, lease=?, updated=? "
                    "WHERE id=?", (worker, now, lease, now, r["id"]))
                self.db.execute("INSERT INTO leases (lease,job,worker,claim_req,created) VALUES (?,?,?,?,?)",
                                (lease, r["id"], worker, request_id, now))
                self.db.execute("UPDATE workers SET state='busy', job=? WHERE name=?", (r["id"], worker))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        return {**self.row(r["id"]), "lease": lease, "replayed": False}

    def _lease(self, job_id, lease):
        """Returns (job, lease_row, status); status is 'active' or the lease's recorded outcome."""
        if not lease or not isinstance(lease, str):
            raise ValueError("lease required: update worker/client.py (queue protocol 2)")
        j = self._raw(job_id)
        if not j:
            raise KeyError(job_id)
        lr = self.db.execute("SELECT * FROM leases WHERE lease=? AND job=?", (lease, job_id)).fetchone()
        if not lr:
            raise Conflict("unknown lease for this job", "lease_lost", outcome=None, state=j["state"])
        if lr["outcome"] is None and j["state"] == "running" and j["lease"] == lease:
            return j, lr, "active"
        return j, lr, lr["outcome"] or "ended"

    def check_lease(self, job_id, lease):
        """Raises unless `lease` is the active lease of `job_id` (worker-token downloads need one)."""
        with self.lock:
            j, _, status = self._lease(job_id, lease)
            if status != "active":
                raise self._lost(j, status)

    @staticmethod
    def _lost(j, status):
        return Conflict(f"lease {status}: job is now {j['state']}", "lease_lost", outcome=status, state=j["state"])

    def heartbeat(self, job_id, worker, lease, progress):
        with self.lock:
            j, _, status = self._lease(job_id, lease)
            if status != "active":
                raise self._lost(j, status)
            now = time.time()
            self.db.execute("UPDATE jobs SET heartbeat=?, progress=?, updated=? WHERE id=?",
                            (now, json.dumps(progress) if progress is not None else None, now, job_id))
            self.db.execute("UPDATE leases SET acked=1 WHERE lease=?", (lease,))

    def complete(self, job_id, worker, lease, score, metrics=None):
        with self.lock:
            j, _, status = self._lease(job_id, lease)
            if status == "done":
                return self.row(job_id)  # retried after a lost response
            if status != "active":
                raise self._lost(j, status)
            now = time.time()
            self.db.execute("UPDATE jobs SET state='done', score=?, metrics=?, heartbeat=NULL, updated=? "
                            "WHERE id=?", (score, json.dumps(metrics) if metrics else None, now, job_id))
            self.db.execute("UPDATE leases SET outcome='done', acked=1, ended=? WHERE lease=?", (now, lease))
            self.db.execute("UPDATE workers SET state='ready', job=NULL WHERE name=? AND job=?",
                            (j["worker"], job_id))
        return self.row(job_id)

    def _end_run(self, j, error, outcome):
        """Ends the current run as one failed attempt: requeue, or `failed` after MAX_ATTEMPTS."""
        attempts = j["attempts"] + 1
        state = "failed" if attempts >= MAX_ATTEMPTS else "queued"
        now = time.time()
        self.db.execute(
            "UPDATE jobs SET state=?, attempts=?, last_error=?, worker=NULL, heartbeat=NULL, lease=NULL, updated=? "
            "WHERE id=?", (state, attempts, error, now, j["id"]))
        if j.get("lease"):
            self.db.execute("UPDATE leases SET outcome=?, ended=? WHERE lease=?", (outcome, now, j["lease"]))
        if j.get("worker"):
            self.db.execute("UPDATE workers SET job=NULL WHERE name=? AND job=?", (j["worker"], j["id"]))

    def fail(self, job_id, worker, lease, error):
        with self.lock:
            j, _, status = self._lease(job_id, lease)
            if status == "failed":
                return self.row(job_id)  # retried after a lost response
            if status != "active":
                raise self._lost(j, status)
            self._end_run(j, error[:8000], outcome="failed")
        return self.row(job_id)

    def reap(self, timeout=None):
        timeout = HEARTBEAT_TIMEOUT if timeout is None else timeout
        cutoff = time.time() - timeout
        with self.lock:
            stale = [dict(r) for r in self.db.execute(
                "SELECT * FROM jobs WHERE state='running' AND heartbeat < ?", (cutoff,))]
            for j in stale:
                lr = self.db.execute("SELECT acked FROM leases WHERE lease=?", (j["lease"],)).fetchone() \
                    if j["lease"] else None
                if lr is not None and not lr["acked"]:
                    msg = (f"lost heartbeat (> {timeout:.0f}s): {j['worker']} claimed the job but never sent a "
                           "heartbeat (worker died right after claiming, or the claim response was lost and "
                           "never retried)")
                else:
                    msg = f"lost heartbeat (> {timeout:.0f}s)"
                self._end_run(j, msg, outcome="expired")
            self.db.execute("DELETE FROM leases WHERE ended IS NOT NULL AND ended < ?", (time.time() - LEASE_KEEP_S,))
        return [j["id"] for j in stale]

    # ------------------------------------------------------------------ files
    def file_path(self, job_id, name):
        if not ID_RE.match(job_id):
            raise KeyError(job_id)
        if not FILE_RE.match(name) or name.startswith("."):
            raise ValueError("bad file name")
        return self.dir / "files" / job_id / name

    def check_upload(self, job_id, name, lease):
        """Inputs: anyone with the token while the job is queued. Outputs and uploads to a running job:
        only the holder of the run's lease."""
        self.file_path(job_id, name)
        with self.lock:
            j = self._raw(job_id)
            if not j:
                raise KeyError(job_id)
            if lease:
                j, _, status = self._lease(job_id, lease)
                if status != "active":
                    raise self._lost(j, status)
                return j
            if j["state"] != "queued":
                raise Conflict(f"job is {j['state']}: only the worker holding its lease can upload", "not_queued")
            if name in json.loads(j["outputs"]):
                raise Conflict(f"{name} is a worker output; upload it with the run's lease", "output_file")
            return j

    def put_file(self, job_id, name, stream, length, worker=None, lease=None):
        """Streams `length` bytes to a unique temp file, fsyncs it, then renames it into place. Call
        check_upload first; ownership is checked again before the rename (a lease can expire mid-upload)."""
        p = self.file_path(job_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=p.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                left = length
                while left:
                    chunk = stream.read(min(CHUNK, left))
                    if not chunk:
                        raise ConnectionError(f"client sent {length - left} of {length} bytes")
                    f.write(chunk)
                    left -= len(chunk)
                f.flush()
                os.fsync(f.fileno())
            with self.lock:
                j = self.check_upload(job_id, name, lease)
                os.replace(tmp, p)
                files = json.loads(j["files"])
                files[name] = f"/jobs/{job_id}/files/{name}"
                outputs = json.loads(j["outputs"])
                if lease and name not in outputs:  # anything the lease holder writes is an output
                    outputs = sorted(outputs + [name])
                self.db.execute("UPDATE jobs SET files=?, outputs=?, updated=? WHERE id=?",
                                (json.dumps(files), json.dumps(outputs), time.time(), job_id))
                if lease:
                    self.db.execute("UPDATE leases SET acked=1 WHERE lease=?", (lease,))
            try:
                dfd = os.open(p.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def worker_may(method, parts):
    """Routes the worker token may use. Everything else (listing or reading jobs, submitting, the worker
    list) needs the producer token. Job file access is further limited to the lease holder in dispatch()."""
    if (method, parts) in (("GET", ["health"]), ("GET", ["bundle"]), ("POST", ["workers"]), ("POST", ["claim"])):
        return True
    if method == "GET" and len(parts) == 2 and parts[0] == "configs":
        return True
    if method == "PUT" and len(parts) == 3 and parts[0] == "reports":
        return True
    if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "files" and method in ("GET", "PUT"):
        return True
    return method == "POST" and len(parts) == 3 and parts[0] == "jobs" and parts[2] in ("heartbeat", "complete", "fail")


def make_handler(store, token, worker_token=None):
    """`token` is the full-access (producer) token. `worker_token`, if set, is a second token that can only
    do what a notebook worker needs: see WORKER_ROUTES and the lease checks in dispatch()."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = READ_TIMEOUT  # socket timeout: stalled uploads and idle keep-alive sockets free their thread

        def log_message(self, fmt, *a):
            if os.environ.get("QUEUE_QUIET") != "1":
                sys.stderr.write("[queue] " + fmt % a + "\n")

        def send(self, code, obj=None, raw=None, ctype="application/json", close=False):
            body = raw if raw is not None else (b"" if obj is None else json.dumps(obj).encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path, ctype="application/octet-stream"):
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                shutil.copyfileobj(f, self.wfile, CHUNK)

        def declared_length(self):
            if self.headers.get("Transfer-Encoding"):
                raise LengthRequired("chunked bodies are not supported; send Content-Length")
            v = self.headers.get("Content-Length")
            if v is None:
                return 0
            n = int(v)
            if n < 0:
                raise ValueError("bad Content-Length")
            return n

        def body(self, limit=MAX_JSON_BYTES):
            n = self.declared_length()
            if n > limit:
                raise TooLarge(f"body is {n} bytes, limit {limit}")
            self.body_read = True
            data = self.rfile.read(n) if n else b""
            if len(data) < n:
                raise ConnectionError(f"client sent {len(data)} of {n} bytes")
            return data

        def drain(self, n):
            """Discards a refused body so the client can read our answer (it sends the whole body before it
            reads a response). Only for authenticated requests within the size limit."""
            self.body_read = True
            while n:
                chunk = self.rfile.read(min(CHUNK, n))
                if not chunk:
                    raise ConnectionError("client closed while sending the body")
                n -= len(chunk)

        def linger(self, budget_s=2.0, max_bytes=64 * CHUNK):
            """After answering without reading the body (401, 411, 413), discard what the client is still
            sending for a moment. Closing with unread data makes the kernel reset the connection, and the
            client would then see a connection error instead of our answer."""
            self.close_connection = True
            try:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.connection.settimeout(0.2)
                end, got = time.monotonic() + budget_s, 0
                while time.monotonic() < end and got < max_bytes:
                    try:
                        b = self.connection.recv(CHUNK)
                    except (TimeoutError, socket.timeout):
                        continue
                    if not b:
                        break
                    got += len(b)
            except OSError:
                pass

        def json_body(self):
            b = self.body()
            d = json.loads(b) if b else {}
            if not isinstance(d, dict):
                raise ValueError("JSON object expected")
            return d

        def authed(self):
            """Returns 'producer', 'worker' or None. Both comparisons always run (constant time)."""
            got = self.headers.get("Authorization", "").encode()
            full = hmac.compare_digest(got, f"Bearer {token}".encode())
            work = bool(worker_token) and hmac.compare_digest(got, f"Bearer {worker_token or ''}".encode())
            return "producer" if full else "worker" if work else None

        def route(self, method):
            self.body_read = False
            try:
                self.role = self.authed()
                if not self.role:  # never read an unauthenticated body
                    self.send(401, {"error": "unauthorized"}, close=True)
                    if self.headers.get("Content-Length", "0") not in ("", "0"):
                        self.linger(budget_s=1.0, max_bytes=4 * CHUNK)
                    return
                parts = [p for p in self.path.split("?")[0].split("/") if p]
                if self.role == "worker" and not worker_may(method, parts):
                    self.send(403, {"error": "the worker token cannot do this; use the producer token",
                                    "code": "forbidden"}, close=True)
                    if self.headers.get("Content-Length", "0") not in ("", "0"):
                        self.linger(budget_s=1.0, max_bytes=4 * CHUNK)
                    return
                return self.dispatch(method, parts)
            except KeyError:
                self.safe_send(404, {"error": "not found"})
            except Conflict as e:
                self.safe_send(409, {"error": str(e), "code": e.code, **e.extra})
            except UpgradeRequired as e:
                self.safe_send(426, {"error": str(e), "min_protocol": MIN_PROTOCOL})
            except TooLarge as e:
                self.safe_send(413, {"error": str(e)}, close=True)
                self.linger()
            except LengthRequired as e:
                self.safe_send(411, {"error": str(e)}, close=True)
                self.linger()
            except (ValueError, json.JSONDecodeError) as e:
                self.safe_send(400, {"error": str(e)})
            except (TimeoutError, ConnectionError) as e:
                sys.stderr.write(f"[queue] {method} {self.path}: client connection: {e}\n")
                self.close_connection = True
            except Exception:
                sys.stderr.write(f"[queue] {method} {self.path} failed:\n{traceback.format_exc()}")
                self.safe_send(500, {"error": "internal error (see queue log)"}, close=True)
            finally:
                try:
                    unread = not self.body_read and self.declared_length() > 0
                except (ValueError, LengthRequired):
                    unread = True
                if unread:  # body left on the socket: the connection cannot be reused
                    self.close_connection = True

        def safe_send(self, *a, **kw):
            try:
                self.send(*a, **kw)
            except OSError:
                self.close_connection = True

        def dispatch(self, m, p):
            if m == "GET" and p == ["health"]:
                return self.send(200, {"ok": True, "protocol": MIN_PROTOCOL, "max_file_bytes": MAX_FILE_BYTES})
            if m == "GET" and p == ["jobs"]:
                return self.send(200, store.list())
            if m == "GET" and p == ["bundle"]:
                # code + inputs for notebook sessions (built by laptop/make_bundle.py)
                f = store.dir / "bundle.tar.gz"
                if not f.is_file():
                    raise KeyError("bundle")
                return self.send_file(f, ctype="application/gzip")
            if m == "PUT" and len(p) == 3 and p[0] == "reports":
                # session reports from notebooks: data/reports/<session>/<file>
                if not FILE_RE.match(p[1]) or not FILE_RE.match(p[2]) or p[1][0] == "." or p[2][0] == ".":
                    raise ValueError("bad report path")
                data = self.body(limit=MAX_REPORT_BYTES)
                d = store.dir / "reports" / p[1]
                d.mkdir(parents=True, exist_ok=True)
                atomic_write(d / p[2], data)
                return self.send(200, {"ok": True, "bytes": len(data)})
            if m == "GET" and p == ["workers"]:
                return self.send(200, store.workers())
            if m == "POST" and p == ["workers"]:
                b = self.json_body()
                if not b.get("worker") or b.get("state") not in ("starting", "ready", "busy", "error", "stopped"):
                    raise ValueError("worker and state (starting|ready|busy|error|stopped) required")
                store.worker_status(b["worker"], b["state"], b.get("family"), b.get("info"), b.get("job"),
                                    b.get("protocol"))
                return self.send(200, {"ok": True})
            if m == "GET" and len(p) == 2 and p[0] == "configs":
                if not CONFIG_RE.match(p[1]):
                    raise ValueError("config must look like name@version")
                f = config_path(p[1])
                if not f.is_file():
                    raise KeyError(p[1])
                return self.send(200, json.loads(f.read_text()))
            if m == "POST" and p == ["jobs"]:
                return self.send(201, store.create(self.json_body()))
            if m == "POST" and p == ["claim"]:
                b = self.json_body()
                if not b.get("worker"):
                    raise ValueError("worker required")
                j = store.claim(b["worker"], b.get("family"), b.get("accept"), b.get("request_id"),
                                b.get("protocol"))
                return self.send(200, j) if j else self.send(204)
            if len(p) >= 2 and p[0] == "jobs":
                jid = p[1]
                if not ID_RE.match(jid):
                    raise KeyError(jid)
                if m == "GET" and len(p) == 2:
                    return self.send(200, store.job(jid))
                if len(p) == 4 and p[2] == "files":
                    if m == "GET":
                        if self.role == "worker":  # workers read inputs only of the job they hold
                            store.check_lease(jid, self.headers.get("X-Lease"))
                        f = store.file_path(jid, p[3])
                        if not f.is_file():
                            raise KeyError(p[3])
                        return self.send_file(f)
                    if m == "PUT":
                        n = self.declared_length()
                        if self.headers.get("Content-Length") is None:
                            raise LengthRequired("Content-Length required")
                        if n > MAX_FILE_BYTES:
                            raise TooLarge(f"file is {n} bytes, limit {MAX_FILE_BYTES} (QUEUE_MAX_FILE_MB)")
                        lease = self.headers.get("X-Lease")
                        try:
                            if self.role == "worker" and not lease:
                                raise ValueError("the worker token can only upload with the run's lease (X-Lease)")
                            store.check_upload(jid, p[3], lease)
                        except (Conflict, KeyError, ValueError):
                            self.drain(n)  # refused before reading: discard so the client gets the answer
                            raise
                        self.body_read = True  # put_file consumes it (or the connection is closed on error)
                        store.put_file(jid, p[3], self.rfile, n, self.headers.get("X-Worker"), lease)
                        return self.send(200, store.row(jid))
                if m == "POST" and len(p) == 3:
                    b = self.json_body()
                    w, lease = b.get("worker", ""), b.get("lease")
                    if p[2] == "heartbeat":
                        store.heartbeat(jid, w, lease, b.get("progress"))
                        return self.send(200, {"ok": True})
                    if p[2] == "complete":
                        return self.send(200, store.complete(jid, w, lease, b.get("score"), b.get("metrics")))
                    if p[2] == "fail":
                        return self.send(200, store.fail(jid, w, lease, str(b.get("error", "unknown"))))
            raise KeyError("route")

        def do_GET(self): self.route("GET")
        def do_POST(self): self.route("POST")
        def do_PUT(self): self.route("PUT")

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # clients vanishing mid-request is routine over a tunnel; log one line instead of a traceback
        e = sys.exc_info()[1]
        if isinstance(e, (ConnectionError, TimeoutError)):
            sys.stderr.write(f"[queue] client {client_address[0]} dropped: {type(e).__name__}\n")
            return
        super().handle_error(request, client_address)


def serve(data_dir, token, port=8787, reap_every=10.0, worker_token=None):
    if worker_token and worker_token == token:
        raise ValueError("the worker token must differ from the producer token")
    store = Store(data_dir)
    httpd = Server(("127.0.0.1", port), make_handler(store, token, worker_token))
    stop = threading.Event()

    def reaper():
        while not stop.wait(reap_every):
            for jid in store.reap():
                sys.stderr.write(f"[queue] requeued/failed stale job {jid}\n")

    threading.Thread(target=reaper, daemon=True).start()
    httpd.stop_reaper = stop
    httpd.store = store
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    a = ap.parse_args()
    token = os.environ.get("QUEUE_TOKEN")
    if not token or len(token) < 16:
        sys.exit("set QUEUE_TOKEN (>=16 chars), e.g. export QUEUE_TOKEN=$(openssl rand -hex 32)")
    worker_token = os.environ.get("QUEUE_WORKER_TOKEN") or None
    if worker_token and (len(worker_token) < 16 or worker_token == token):
        sys.exit("QUEUE_WORKER_TOKEN must be at least 16 chars and differ from QUEUE_TOKEN")
    httpd = serve(a.data, token, a.port, worker_token=worker_token)
    print(f"queue on http://127.0.0.1:{a.port} data={a.data} protocol={MIN_PROTOCOL} "
          f"worker_token={'on' if worker_token else 'off'} "
          f"max_file={MAX_FILE_BYTES >> 20}MiB read_timeout={READ_TIMEOUT:.0f}s", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
