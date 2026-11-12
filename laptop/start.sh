#!/usr/bin/env bash
# Start queue + Cloudflare tunnel on the laptop. Prints the URL to paste into the notebook cell.
#
# Optional environment:
#   TUNNEL_NAME + TUNNEL_HOSTNAME  use a named tunnel with a fixed address instead of a quick tunnel, e.g.
#                                  TUNNEL_NAME=gpuq TUNNEL_HOSTNAME=queue.example.com (one-time setup in
#                                  docs/multi-session.md). The URL then never changes.
#   QUEUE_URL_GIST                 id of a GitHub gist; the current URL is written to its file "queue_url"
#                                  (needs the gh CLI, logged in). Notebooks with
#                                  QUEUE_URL_SOURCE=https://api.github.com/gists/<id> follow it after restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
for command in python3 openssl cloudflared curl; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "missing required command: $command" >&2
    exit 1
  }
done
if [ -n "${TUNNEL_NAME:-}" ] || [ -n "${TUNNEL_HOSTNAME:-}" ]; then
  [ -n "${TUNNEL_NAME:-}" ] && [ -n "${TUNNEL_HOSTNAME:-}" ] || {
    echo "set both TUNNEL_NAME and TUNNEL_HOSTNAME for a named tunnel" >&2
    exit 1
  }
fi
if [ -n "${QUEUE_URL_GIST:-}" ]; then
  command -v gh >/dev/null 2>&1 || { echo "QUEUE_URL_GIST needs the gh CLI" >&2; exit 1; }
fi
mkdir -p run
[ -f run/token ] || { openssl rand -hex 32 > run/token; chmod 600 run/token; }
# restricted token for notebooks you don't fully trust (other people's accounts): it can run jobs but not
# list, read or submit them
[ -f run/worker_token ] || { openssl rand -hex 32 > run/worker_token; chmod 600 run/worker_token; }
export QUEUE_TOKEN="$(cat run/token)"
export QUEUE_WORKER_TOKEN="$(cat run/worker_token)"
pkill -f laptop/queue_server.py || true
pkill -f "cloudflared tunnel --no-autoupdate .*http://127.0.0.1:8787" || true
nohup python3 laptop/queue_server.py > run/queue.log 2>&1 &
if [ -n "${TUNNEL_NAME:-}" ]; then
  nohup cloudflared tunnel --no-autoupdate run --url http://127.0.0.1:8787 "$TUNNEL_NAME" > run/tunnel.log 2>&1 &
else
  nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8787 > run/tunnel.log 2>&1 &
fi
for _ in $(seq 50); do
  curl -fsS --max-time 1 -H "Authorization: Bearer $QUEUE_TOKEN" http://127.0.0.1:8787/health >/dev/null 2>&1 && break
  sleep 0.1
done
curl -fsS --max-time 2 -H "Authorization: Bearer $QUEUE_TOKEN" http://127.0.0.1:8787/health >/dev/null || {
  echo "queue did not start, see run/queue.log" >&2
  exit 1
}
if [ -n "${TUNNEL_HOSTNAME:-}" ]; then
  U="https://$TUNNEL_HOSTNAME"
else
  for _ in $(seq 60); do
    U=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' run/tunnel.log | head -1 || true)
    [ -n "$U" ] && break; sleep 1
  done
  [ -n "${U:-}" ] || {
    echo "cloudflared did not publish a tunnel URL, see run/tunnel.log" >&2
    exit 1
  }
fi
for _ in $(seq 90); do
  curl -fsS --max-time 2 -H "Authorization: Bearer $QUEUE_TOKEN" "$U/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS --max-time 5 -H "Authorization: Bearer $QUEUE_TOKEN" "$U/health" >/dev/null || {
  echo "tunnel URL $U is not publicly reachable, see run/tunnel.log" >&2
  exit 1
}
echo "$U" > run/url
if [ -n "${QUEUE_URL_GIST:-}" ]; then
  printf '{"files":{"queue_url":{"content":"%s\\n"}}}' "$U" \
    | gh api -X PATCH "gists/$QUEUE_URL_GIST" --input - >/dev/null || {
    echo "could not update gist $QUEUE_URL_GIST (the queue is running; notebooks still need the new URL)" >&2
    exit 1
  }
  echo "https://api.github.com/gists/$QUEUE_URL_GIST" > run/url_source
  echo "QUEUE_URL_SOURCE=$(cat run/url_source)"
else
  rm -f run/url_source
fi
echo "QUEUE_URL=$U"
echo "QUEUE_TOKEN is in run/token, the restricted worker token in run/worker_token"
echo "(put them in the provider's secrets as QUEUE_TOKEN, never in notebook text)"
