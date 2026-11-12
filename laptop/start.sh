#!/usr/bin/env bash
# Start queue + quick tunnel on the laptop. Prints the URL to paste into the notebook cell.
set -euo pipefail
cd "$(dirname "$0")/.."
for command in python3 openssl cloudflared curl; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "missing required command: $command" >&2
    exit 1
  }
done
mkdir -p run
[ -f run/token ] || { openssl rand -hex 32 > run/token; chmod 600 run/token; }
export QUEUE_TOKEN="$(cat run/token)"
pkill -f laptop/queue_server.py || true
pkill -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8787" || true
nohup python3 laptop/queue_server.py > run/queue.log 2>&1 &
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8787 > run/tunnel.log 2>&1 &
for _ in $(seq 50); do
  curl -fsS --max-time 1 -H "Authorization: Bearer $QUEUE_TOKEN" http://127.0.0.1:8787/health >/dev/null 2>&1 && break
  sleep 0.1
done
curl -fsS --max-time 2 -H "Authorization: Bearer $QUEUE_TOKEN" http://127.0.0.1:8787/health >/dev/null || {
  echo "queue did not start, see run/queue.log" >&2
  exit 1
}
for _ in $(seq 60); do
  U=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' run/tunnel.log | head -1 || true)
  [ -n "$U" ] && break; sleep 1
done
[ -n "${U:-}" ] || {
  echo "cloudflared did not publish a tunnel URL, see run/tunnel.log" >&2
  exit 1
}
for _ in $(seq 90); do
  curl -fsS --max-time 2 -H "Authorization: Bearer $QUEUE_TOKEN" "$U/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS --max-time 5 -H "Authorization: Bearer $QUEUE_TOKEN" "$U/health" >/dev/null || {
  echo "tunnel URL was published but is not publicly reachable, see run/tunnel.log" >&2
  exit 1
}
echo "$U" > run/url
echo "QUEUE_URL=$U"
echo "QUEUE_TOKEN is in run/token (put it in the provider's secrets, never in notebook text)"
