#!/usr/bin/env bash
# Fetch TRELLIS.2 at a pinned commit and apply the patches in worker/trellis2_t4/patches/. Run as a session phase:
#   --phases preflight,trellis2,workers --cmd "trellis2=bash worker/trellis2_t4/setup.sh"
# Custom phases start once per GPU; a lock makes the first copy do the work and the others wait for it.
# Writes $TRELLIS2_HOME/ready.json; worker/trellis2_t4/pipeline_512.py imports TRELLIS.2 from $TRELLIS2_HOME/src.
# Compiling the CUDA extensions (cumesh, o_voxel, nvdiffrast, flex_gemm) is not part of this script yet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -d /kaggle/working ]; then TRELLIS2_HOME=${TRELLIS2_HOME:-/kaggle/working/trellis2}; else TRELLIS2_HOME=${TRELLIS2_HOME:-$PWD/trellis2}; fi
TRELLIS2_REPO=${TRELLIS2_REPO:-https://github.com/microsoft/TRELLIS.2.git}
TRELLIS2_COMMIT=${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}   # main, 2026-06-05
SRC=$TRELLIS2_HOME/src

mkdir -p "$TRELLIS2_HOME"
exec 9>"$TRELLIS2_HOME/.setup.lock"
flock 9                                                   # other GPUs' copies wait here
if [ -f "$TRELLIS2_HOME/ready.json" ]; then echo "trellis2 already set up: $(cat "$TRELLIS2_HOME/ready.json")"; exit 0; fi

rm -rf "$SRC"
git clone -q --filter=blob:none --no-checkout "$TRELLIS2_REPO" "$SRC"
git -C "$SRC" checkout -q "$TRELLIS2_COMMIT"
patches=()
for p in "$HERE"/patches/*.patch; do
  git -C "$SRC" apply --check "$p"                         # fails loudly if upstream moved under the patch
  git -C "$SRC" apply "$p"
  patches+=("$(basename "$p")")
done
printf '{"commit": "%s", "patches": [%s], "src": "%s"}\n' "$TRELLIS2_COMMIT" \
  "$(printf '"%s",' "${patches[@]}" | sed 's/,$//')" "$SRC" > "$TRELLIS2_HOME/ready.json"
echo "trellis2 ready: $(cat "$TRELLIS2_HOME/ready.json")"
