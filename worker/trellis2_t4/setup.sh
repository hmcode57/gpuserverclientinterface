#!/usr/bin/env bash
# Prepare TRELLIS.2 (512 pipeline) on a notebook GPU. Run as a session phase:
#   --phases preflight,trellis2,workers --cmd "trellis2=bash worker/trellis2_t4/setup.sh"
# Custom phases start once per GPU; a lock makes the first copy do the work and the others wait for it.
#
#   1. TRELLIS.2 source at a pinned commit, with worker/trellis2_t4/patches/*.patch applied
#   2. DINOv3 (gated) from an attached private dataset, copied to /tmp in the background while
#   3. the open TRELLIS.2 weights (512 files only, about 9.8 GB) download from Hugging Face to /tmp
#
# Writes $TRELLIS2_HOME/ready.json (source, weights, DINOv3 paths), which pipeline_512.from_ready() reads.
# Compiling the CUDA extensions (cumesh, o_voxel, nvdiffrast, flex_gemm) is not part of this script yet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -d /kaggle/working ]; then TRELLIS2_HOME=${TRELLIS2_HOME:-/kaggle/working/trellis2}; else TRELLIS2_HOME=${TRELLIS2_HOME:-$PWD/trellis2}; fi
TRELLIS2_REPO=${TRELLIS2_REPO:-https://github.com/microsoft/TRELLIS.2.git}
TRELLIS2_COMMIT=${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}   # main, 2026-06-05
INPUT=${TRELLIS2_INPUT:-/kaggle/input}                 # where attached datasets are mounted
WEIGHTS=${TRELLIS2_WEIGHTS:-/tmp/trellis2-weights}      # /tmp: larger than /kaggle/working, not saved
DINO=${TRELLIS2_DINO:-/tmp/dinov3-vitl16-hf}            # where the DINOv3 copy goes
DINO_SRC=${TRELLIS2_DINO_SRC:-}                         # dataset folder; found under $INPUT when empty
SRC=$TRELLIS2_HOME/src

mkdir -p "$TRELLIS2_HOME"
exec 9>"$TRELLIS2_HOME/.setup.lock"
flock 9                                                   # other GPUs' copies wait here
if [ -f "$TRELLIS2_HOME/ready.json" ]; then echo "trellis2 already set up: $(cat "$TRELLIS2_HOME/ready.json")"; exit 0; fi
t0=$(date +%s)
log() { echo "[trellis2-setup +$(( $(date +%s) - t0 ))s] $*"; }

# ---- DINOv3: start the copy first, it overlaps everything below ----
if [ -z "$DINO_SRC" ]; then
  cfg=$(grep -rl --include=config.json '"model_type": "dinov3_vit"' "$INPUT" 2>/dev/null | head -1 || true)
  [ -n "$cfg" ] || { echo "no DINOv3 under $INPUT: attach the private dataset (e.g. dinov3-vitl16-hf) or set TRELLIS2_DINO_SRC" >&2; exit 1; }
  DINO_SRC=$(dirname "$cfg")
fi
copy_dino() {
  if [ -f "$DINO/model.safetensors" ] && cmp -s "$DINO_SRC/config.json" "$DINO/config.json" \
     && [ "$(stat -c %s "$DINO_SRC/model.safetensors")" = "$(stat -c %s "$DINO/model.safetensors")" ]; then return 0; fi
  rm -rf "$DINO.part" && mkdir -p "$DINO.part"
  cp "$DINO_SRC"/config.json "$DINO_SRC"/model.safetensors "$DINO.part"/
  cp "$DINO_SRC"/preprocessor_config.json "$DINO.part"/ 2>/dev/null || true
  rm -rf "$DINO" && mv "$DINO.part" "$DINO"
}
log "DINOv3: copying $DINO_SRC -> $DINO in the background"
copy_dino & dino_pid=$!

# ---- source ----
rm -rf "$SRC"
git clone -q --filter=blob:none --no-checkout "$TRELLIS2_REPO" "$SRC"
git -C "$SRC" checkout -q "$TRELLIS2_COMMIT"
for p in "$HERE"/patches/*.patch; do
  git -C "$SRC" apply --check "$p"                         # fails loudly if upstream moved under the patch
  git -C "$SRC" apply "$p"
done
log "source at $TRELLIS2_COMMIT, patches applied"

# ---- open weights from Hugging Face ----
log "weights: downloading the 512 files to $WEIGHTS"
local_dirs=$(python3 "$HERE/pipeline_512.py" download "$WEIGHTS" | tail -1)
log "weights done: $(du -sh "$WEIGHTS" | cut -f1)"

wait "$dino_pid" || { echo "DINOv3 copy failed" >&2; exit 1; }
log "DINOv3 done: $(du -sh "$DINO" | cut -f1)"

python3 - "$TRELLIS2_HOME/ready.json" "$TRELLIS2_COMMIT" "$SRC" "$DINO" "$local_dirs" "$HERE"/patches/*.patch <<'EOF'
import json, os, sys
out, commit, src, dino, local_dirs, *patches = sys.argv[1:]
json.dump({"commit": commit, "patches": [os.path.basename(p) for p in patches], "src": src,
           "dino": dino, "weights": json.loads(local_dirs)}, open(out, "w"), indent=2)
EOF
log "ready: $(tr -d '\n ' < "$TRELLIS2_HOME/ready.json")"
