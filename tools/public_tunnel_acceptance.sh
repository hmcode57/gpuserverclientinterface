#!/usr/bin/env bash
# Opt-in acceptance test for the protocol-2 queue over a Cloudflare quick tunnel.
# Requires: bash, python3, curl and cloudflared.
# It creates a temporary queue database and never prints QUEUE_TOKEN.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCRATCH=
QUEUE_PID=
TUNNEL_PID=
WORKER_PID=

fail() {
    printf '%s\n' "public-tunnel acceptance: $*" >&2
    exit 1
}

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -n "${WORKER_PID:-}" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    if [ -n "${TUNNEL_PID:-}" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
    if [ -n "${QUEUE_PID:-}" ] && kill -0 "$QUEUE_PID" 2>/dev/null; then
        kill "$QUEUE_PID" 2>/dev/null || true
        wait "$QUEUE_PID" 2>/dev/null || true
    fi
    if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ]; then
        rm -rf -- "$SCRATCH"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM HUP

require() {
    command -v "$1" >/dev/null 2>&1 || fail "requires '$1' in PATH"
}
for command in python3 curl cloudflared grep; do
    require "$command"
done

[ -f "$ROOT/laptop/queue_server.py" ] || fail "run from within the repository"

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/gpuq-public-tunnel.XXXXXX") || fail "could not create scratch directory"
PORT=$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
) || fail "could not select an unused local port"
TOKEN=$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
) || fail "could not generate queue token"
LOCAL_URL="http://127.0.0.1:$PORT"
SERVER_LOG="$SCRATCH/queue.log"
TUNNEL_LOG="$SCRATCH/cloudflared.log"

QUEUE_TOKEN="$TOKEN" \
QUEUE_HEARTBEAT_TIMEOUT=2 \
python3 "$ROOT/laptop/queue_server.py" --port "$PORT" --data "$SCRATCH/data" >"$SERVER_LOG" 2>&1 &
QUEUE_PID=$!

wait_for_local_queue() {
    attempts=0
    while [ "$attempts" -lt 100 ]; do
        if curl --silent --fail --max-time 2 \
            -H "Authorization: Bearer $TOKEN" "$LOCAL_URL/health" >"$SCRATCH/local-health.json" 2>/dev/null; then
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 0.1
    done
    fail "queue did not become ready; see $SERVER_LOG"
}
wait_for_local_queue

cloudflared tunnel --no-autoupdate --url "$LOCAL_URL" >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

PUBLIC_URL=
find_tunnel_url() {
    grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -n 1 || true
}
attempts=0
while [ "$attempts" -lt 150 ]; do
    PUBLIC_URL=$(find_tunnel_url)
    [ -n "$PUBLIC_URL" ] && break
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        fail "cloudflared exited before publishing a quick-tunnel URL; see $TUNNEL_LOG"
    fi
    attempts=$((attempts + 1))
    sleep 0.2
done
[ -n "$PUBLIC_URL" ] || fail "cloudflared did not publish a trycloudflare.com URL within 30 seconds; see $TUNNEL_LOG"

health_code=
attempts=0
while [ "$attempts" -lt 450 ]; do
    health_code=$(curl --silent --show-error --output "$SCRATCH/public-health.json" --write-out '%{http_code}' \
        --max-time 2 -H "Authorization: Bearer $TOKEN" "$PUBLIC_URL/health" 2>"$SCRATCH/public-health.err" || true)
    if [ "$health_code" = 200 ]; then
        break
    fi
    attempts=$((attempts + 1))
    sleep 0.2
done
[ "$health_code" = 200 ] || {
    printf 'last public /health error: ' >&2
    cat "$SCRATCH/public-health.err" >&2 || true
    fail "public /health was not reachable within 90 seconds of tunnel URL publication"
}
python3 - "$SCRATCH/public-health.json" <<'PY' || fail "public /health did not return protocol 2 JSON"
import json, sys
health = json.load(open(sys.argv[1]))
assert health.get("ok") is True and health.get("protocol") == 2
PY

wrong_code=$(curl --silent --show-error --output "$SCRATCH/wrong-token.json" --write-out '%{http_code}' \
    --max-time 20 -H 'Authorization: Bearer definitely-wrong-token' "$PUBLIC_URL/jobs") \
    || fail "wrong-token public /jobs request failed"
[ "$wrong_code" = 401 ] || fail "public /jobs with a wrong bearer token returned HTTP $wrong_code, expected 401"

submit() {
    QUEUE_TOKEN="$TOKEN" python3 "$ROOT/laptop/submit.py" --url "$PUBLIC_URL" "$@"
}
NORMAL_ID=$(submit --kind tunnel-test --family echo --params '{"case": "normal"}')
DEPENDENT_ID=$(submit --kind tunnel-test --family echo --params '{"case": "dependent"}' --after "$NORMAL_ID")
UNSERVED_ID=$(submit --kind tunnel-test --family nobody --params '{"case": "no worker for this family"}')

start_worker() {
    QUEUE_URL="$PUBLIC_URL" QUEUE_TOKEN="$TOKEN" \
        python3 "$ROOT/worker/fake_worker.py" --name public-tunnel-worker --family echo \
        --poll 0.1 --work-seconds 60 --heartbeat 60 "$@" >>"$SCRATCH/worker.log" 2>&1 &
    WORKER_PID=$!
}

# Kill a live worker only after its log proves it has claimed the first job.
start_worker
attempts=0
while [ "$attempts" -lt 200 ]; do
    if grep -F "claimed $NORMAL_ID" "$SCRATCH/worker.log" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        fail "first worker exited before claiming normal job; see $SCRATCH/worker.log"
    fi
    attempts=$((attempts + 1))
    sleep 0.1
done
grep -F "claimed $NORMAL_ID" "$SCRATCH/worker.log" >/dev/null 2>&1 \
    || fail "worker did not claim normal job within 20 seconds"
kill "$WORKER_PID" 2>/dev/null || fail "could not stop first worker after claim"
wait "$WORKER_PID" 2>/dev/null || true
WORKER_PID=

# The server reaper runs every 10 seconds. Polling observes real lease expiry, not a fixed sleep.
job_json() {
    curl --silent --show-error --fail --max-time 15 -H "Authorization: Bearer $TOKEN" "$PUBLIC_URL/jobs/$1"
}
wait_for_state() {
    wanted=$1
    job_id=$2
    attempts=0
    while [ "$attempts" -lt 350 ]; do
        if job_json "$job_id" >"$SCRATCH/job-$job_id.json"; then
            state=$(python3 - "$SCRATCH/job-$job_id.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["state"])
PY
)
            [ "$state" = "$wanted" ] && return 0
        fi
        attempts=$((attempts + 1))
        sleep 0.1
    done
    fail "job $job_id did not reach state '$wanted'"
}
wait_for_state queued "$NORMAL_ID"

# Restart exactly the same worker identity twice to finish the retried normal job and its dependency.
start_worker --once --work-seconds 0.2 --heartbeat 0.1
wait "$WORKER_PID" || fail "first restart worker failed; see $SCRATCH/worker.log"
WORKER_PID=
start_worker --once --work-seconds 0.2 --heartbeat 0.1
wait "$WORKER_PID" || fail "second restart worker failed; see $SCRATCH/worker.log"
WORKER_PID=

wait_for_state done "$NORMAL_ID"
wait_for_state done "$DEPENDENT_ID"
job_json "$NORMAL_ID" >"$SCRATCH/normal.json"
job_json "$DEPENDENT_ID" >"$SCRATCH/dependent.json"
job_json "$UNSERVED_ID" >"$SCRATCH/unserved.json"
curl --silent --show-error --fail --max-time 30 -H "Authorization: Bearer $TOKEN" \
    "$PUBLIC_URL/jobs/$NORMAL_ID/files/result.bin" >"$SCRATCH/normal-result.bin"
curl --silent --show-error --fail --max-time 30 -H "Authorization: Bearer $TOKEN" \
    "$PUBLIC_URL/jobs/$DEPENDENT_ID/files/result.bin" >"$SCRATCH/dependent-result.bin"
[ -s "$SCRATCH/normal-result.bin" ] && [ -s "$SCRATCH/dependent-result.bin" ] \
    || fail "completed result.bin output was empty"
python3 - "$SCRATCH/normal.json" "$SCRATCH/dependent.json" "$SCRATCH/unserved.json" <<'PY' \
    || fail "job result assertions failed"
import json, sys
normal, dependent, unserved = (json.load(open(path)) for path in sys.argv[1:])
assert normal["state"] == "done" and "result.bin" in normal["files"]
assert dependent["state"] == "done" and "result.bin" in dependent["files"]
assert unserved["state"] == "queued"
reason = unserved.get("waiting_for") or ""
assert reason and "no live worker accepts 'nobody'" in reason, reason
PY

if grep -F 'Traceback (most recent call last):' "$SERVER_LOG" >/dev/null 2>&1; then
    fail "server log contains a traceback; see $SERVER_LOG"
fi

printf '%s\n' "public-tunnel acceptance passed"
