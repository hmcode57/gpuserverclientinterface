"""Object storage for large job data (model weights, deltas) that should not go through the laptop.
Stdlib only. The laptop keeps the credentials and hands workers short-lived presigned URLs, one per object
and method, so a notebook can read or write exactly the objects of its job and nothing else.

  R2Store      Cloudflare R2 (or any S3-compatible store) via AWS Signature V4 presigned URLs.
               Env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
               (optional R2_ENDPOINT for other S3-compatible services, e.g. https://s3.eu-central-1.amazonaws.com
               with R2_REGION=eu-central-1).
  HttpStore    a plain base URL (tests, or a local server you run yourself): no signing.

Both expose get_url(key), put_url(key) and, for the laptop's own use, get(key), exists(key), delete(key).
"""
import datetime, hashlib, hmac, os, urllib.error, urllib.parse, urllib.request

UNRESERVED = "-_.~"


def _q(s, safe=""):
    return urllib.parse.quote(s, safe=UNRESERVED + safe)


def presign(method, host, uri, access_key, secret_key, region, expires=3600, now=None, service="s3",
            scheme="https"):
    """AWS SigV4 query-string presigned URL (UNSIGNED-PAYLOAD). `uri` is the path, e.g. /bucket/key."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/{region}/{service}/aws4_request"
    query = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access_key}/{scope}",
             "X-Amz-Date": amz_date, "X-Amz-Expires": str(int(expires)), "X-Amz-SignedHeaders": "host"}
    canonical_query = "&".join(f"{_q(k)}={_q(v)}" for k, v in sorted(query.items()))
    canonical_uri = _q(uri, safe="/")
    canonical = "\n".join([method, canonical_uri, canonical_query, f"host:{host}", "", "host", "UNSIGNED-PAYLOAD"])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + secret_key).encode()
    for part in (day, region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{scheme}://{host}{canonical_uri}?{canonical_query}&X-Amz-Signature={sig}"


class Store:
    expires = 12 * 3600

    def get_url(self, key):
        raise NotImplementedError

    def put_url(self, key):
        raise NotImplementedError

    def delete_url(self, key):
        raise NotImplementedError

    def delete(self, key):
        """Best effort; returns False instead of raising (a leftover object only costs storage)."""
        try:
            urllib.request.urlopen(urllib.request.Request(self.delete_url(key), method="DELETE"), timeout=60).close()
            return True
        except (urllib.error.URLError, OSError):
            return False

    def get(self, key, timeout=600):
        with urllib.request.urlopen(self.get_url(key), timeout=timeout) as r:
            return r.read()

    def exists(self, key):
        try:
            req = urllib.request.Request(self.get_url(key), headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60):
                return True
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):  # S3 answers 403 for missing keys when the presign can't list
                return False
            raise


class R2Store(Store):
    def __init__(self, account_id=None, access_key=None, secret_key=None, bucket=None, endpoint=None,
                 region=None, expires=None):
        env = os.environ.get
        self.access_key = access_key or env("R2_ACCESS_KEY_ID")
        self.secret_key = secret_key or env("R2_SECRET_ACCESS_KEY")
        self.bucket = bucket or env("R2_BUCKET")
        account = account_id or env("R2_ACCOUNT_ID")
        endpoint = endpoint or env("R2_ENDPOINT") or (f"https://{account}.r2.cloudflarestorage.com" if account else None)
        missing = [n for n, v in (("R2_ACCESS_KEY_ID", self.access_key), ("R2_SECRET_ACCESS_KEY", self.secret_key),
                                  ("R2_BUCKET", self.bucket), ("R2_ACCOUNT_ID or R2_ENDPOINT", endpoint)) if not v]
        if missing:
            raise ValueError("object store not configured, set " + ", ".join(missing))
        u = urllib.parse.urlsplit(endpoint)
        self.scheme, self.host = u.scheme, u.netloc
        self.region = region or env("R2_REGION") or "auto"
        if expires:
            self.expires = expires

    def _url(self, method, key):
        return presign(method, self.host, f"/{self.bucket}/{key.lstrip('/')}", self.access_key, self.secret_key,
                       self.region, self.expires, scheme=self.scheme)

    def get_url(self, key):
        return self._url("GET", key)

    def put_url(self, key):
        return self._url("PUT", key)

    def delete_url(self, key):
        return self._url("DELETE", key)


class HttpStore(Store):
    """Unsigned base URL: GET/PUT <base>/<key>. For tests and self-hosted stores on a trusted network."""

    def __init__(self, base):
        self.base = base.rstrip("/")

    def get_url(self, key):
        return f"{self.base}/{_q(key.lstrip('/'), safe='/')}"

    put_url = delete_url = get_url


def from_spec(spec):
    """'r2' (credentials from the environment) or 'http://host:port/prefix' (HttpStore)."""
    if spec in (None, "", "r2"):
        return R2Store()
    if spec.startswith(("http://", "https://")):
        return HttpStore(spec)
    raise ValueError(f"unknown store {spec!r}: use r2 or an http(s) base URL")
