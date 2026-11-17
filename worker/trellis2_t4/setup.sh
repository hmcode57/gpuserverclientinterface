#!/usr/bin/env bash
# Prepare TRELLIS.2 (512 pipeline) on a notebook GPU. Run as a session phase:
#   --phases preflight,install,trellis2,workers --cmd "trellis2=bash worker/trellis2_t4/setup.sh"
#   --requirements worker/trellis2_t4/requirements.txt          (the install phase: pure-Python deps)
# Custom phases start once per GPU; a lock makes the first copy do the work and the others wait for it.
#
#   1. TRELLIS.2 source at a pinned commit, with worker/trellis2_t4/patches/*.patch applied
#   2. in the background, at the same time:
#        DINOv3 (gated) copied from an attached private dataset to /tmp
#        the open TRELLIS.2 weights (512 files only, about 9.8 GB) downloaded from Hugging Face to /tmp
#   3. the CUDA extensions (cumesh, flex_gemm, nvdiffrast, o_voxel): installed from an attached wheels dataset
#      that matches this Python/torch, or compiled for the T4 (sm_75) and saved to $TRELLIS2_HOME/wheels so the
#      next session can skip the build (docs/trellis2-t4-run.md, "Save the build")
#
# Writes $TRELLIS2_HOME/ready.json (source, weights, DINOv3 paths), which pipeline_512.from_ready() reads.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -d /kaggle/working ]; then TRELLIS2_HOME=${TRELLIS2_HOME:-/kaggle/working/trellis2}; else TRELLIS2_HOME=${TRELLIS2_HOME:-$PWD/trellis2}; fi
TRELLIS2_REPO=${TRELLIS2_REPO:-https://github.com/microsoft/TRELLIS.2.git}
TRELLIS2_COMMIT=${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}   # main, 2026-06-05
INPUT=${TRELLIS2_INPUT:-/kaggle/input}                 # where attached datasets are mounted
WEIGHTS=${TRELLIS2_WEIGHTS:-/tmp/trellis2-weights}      # /tmp: larger than /kaggle/working, not saved
DINO=${TRELLIS2_DINO:-/tmp/dinov3-vitl16-hf}            # where the DINOv3 copy goes
DINO_SRC=${TRELLIS2_DINO_SRC:-}                         # dataset folder; found under $INPUT when empty
SKIP_EXT=${TRELLIS2_SKIP_EXT:-0}                        # 1 = don't build/install extensions (tests, CPU checks)
CUDA_ARCH=${TRELLIS2_CUDA_ARCH:-7.5}                    # T4
EXT_BUILD=${TRELLIS2_EXT_BUILD:-/tmp/trellis2-ext}      # extension sources while building
# Extension sources. Upstream's setup.sh installs whatever is newest; these are pinned.
CUMESH_REPO=https://github.com/JeffreyXiang/CuMesh.git;     CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
FLEXGEMM_REPO=https://github.com/JeffreyXiang/FlexGEMM.git; FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
NVDR_REPO=https://github.com/NVlabs/nvdiffrast.git;         NVDR_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae   # v0.4.0
PATCHES=${TRELLIS2_PATCHES:-$HERE/patches}             # tests point this at a fake repo's patches
SRC=$TRELLIS2_HOME/src
WHEELS=$TRELLIS2_HOME/wheels

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
  cp "$DINO_SRC"/preprocessor_config.json "$DINO_SRC"/LICENSE.md "$DINO.part"/ 2>/dev/null || true
  rm -rf "$DINO" && mv "$DINO.part" "$DINO"
}
log "DINOv3: copying $DINO_SRC -> $DINO in the background"
copy_dino & dino_pid=$!

# ---- open weights from Hugging Face, also in the background ----
log "weights: downloading the 512 files to $WEIGHTS in the background"
python3 "$HERE/pipeline_512.py" download "$WEIGHTS" > "$TRELLIS2_HOME/weights.json" 2> "$TRELLIS2_HOME/weights.log" &
weights_pid=$!

# ---- source ----
rm -rf "$SRC"
git clone -q --filter=blob:none --no-checkout "$TRELLIS2_REPO" "$SRC"
git -C "$SRC" checkout -q "$TRELLIS2_COMMIT"
for p in "$PATCHES"/*.patch; do
  git -C "$SRC" apply --check "$p"                         # fails loudly if upstream moved under the patch
  git -C "$SRC" apply "$p"
done
log "source at $TRELLIS2_COMMIT, patches applied"

# ---- CUDA extensions ----
ext_key() {   # what a compiled wheel depends on: Python ABI, torch + CUDA, GPU arch, sources
  python3 -c "import sys, torch; print(f'cp{sys.version_info[0]}{sys.version_info[1]} torch={torch.__version__} cuda={torch.version.cuda} arch=$CUDA_ARCH')"
  echo "cumesh=$CUMESH_COMMIT flexgemm=$FLEXGEMM_COMMIT nvdiffrast=$NVDR_COMMIT trellis2=$TRELLIS2_COMMIT"
}
fetch() {     # fetch <repo> <commit> <dir>: clone at an exact commit, with submodules
  rm -rf "$3"; git clone -q --filter=blob:none "$1" "$3"; git -C "$3" checkout -q "$2"
  git -C "$3" submodule -q update --init --recursive
}
if [ "$SKIP_EXT" = 1 ]; then
  log "extensions: skipped (TRELLIS2_SKIP_EXT=1)"
else
  key=$(ext_key)
  found=""
  for m in $(find "$INPUT" -name trellis2-t4-wheels.txt 2>/dev/null); do
    if [ "$(cat "$m")" = "$key" ]; then found=$(dirname "$m"); break; fi
    log "extensions: ignoring $(dirname "$m"), built for a different setup: $(head -1 "$m")"
  done
  if [ -n "$found" ]; then
    log "extensions: installing the wheels from $found"
  else
    log "extensions: building for sm_${CUDA_ARCH/./} ($(echo "$key" | head -1)); this takes a while"
    command -v nvcc >/dev/null || { echo "nvcc not found: the build needs a GPU image with the CUDA toolkit" >&2; exit 1; }
    export TORCH_CUDA_ARCH_LIST=$CUDA_ARCH MAX_JOBS=${MAX_JOBS:-4}
    rm -rf "$WHEELS" "$EXT_BUILD" && mkdir -p "$WHEELS" "$EXT_BUILD"
    fetch "$CUMESH_REPO" "$CUMESH_COMMIT" "$EXT_BUILD/CuMesh"
    fetch "$FLEXGEMM_REPO" "$FLEXGEMM_COMMIT" "$EXT_BUILD/FlexGEMM"
    fetch "$NVDR_REPO" "$NVDR_COMMIT" "$EXT_BUILD/nvdiffrast"
    git -C "$SRC" submodule -q update --init --depth 1 o-voxel/third_party/eigen   # header-only, o-voxel needs it
    cp -r "$SRC/o-voxel" "$EXT_BUILD/o-voxel"
    for e in CuMesh FlexGEMM nvdiffrast o-voxel; do
      b0=$(date +%s)
      python3 -m pip wheel -q --no-build-isolation --no-deps -w "$WHEELS" "$EXT_BUILD/$e" > "$TRELLIS2_HOME/build_$e.log" 2>&1 \
        || { tail -30 "$TRELLIS2_HOME/build_$e.log" >&2; echo "building $e failed, full log: $TRELLIS2_HOME/build_$e.log" >&2; exit 1; }
      log "extensions: built $e in $(( $(date +%s) - b0 ))s"
    done
    echo "$key" > "$WHEELS/trellis2-t4-wheels.txt"
    found=$WHEELS
  fi
  ls "$found"/*.whl >/dev/null 2>&1 || { echo "no .whl files in $found (did Kaggle unpack them on upload?)" >&2; exit 1; }
  python3 -m pip install -q --no-deps --no-index "$found"/*.whl
  python3 -c "import cumesh, flex_gemm, nvdiffrast.torch, o_voxel" \
    || { echo "the extensions don't import after installing" >&2; exit 1; }
  log "extensions: cumesh, flex_gemm, nvdiffrast, o_voxel import"
fi

wait "$weights_pid" || { tail -20 "$TRELLIS2_HOME/weights.log" >&2; echo "weights download failed, log: $TRELLIS2_HOME/weights.log" >&2; exit 1; }
local_dirs=$(tail -1 "$TRELLIS2_HOME/weights.json")
log "weights done: $(du -sh "$WEIGHTS" | cut -f1)"
wait "$dino_pid" || { echo "DINOv3 copy failed" >&2; exit 1; }
log "DINOv3 done: $(du -sh "$DINO" | cut -f1)"

python3 - "$TRELLIS2_HOME/ready.json" "$TRELLIS2_COMMIT" "$SRC" "$DINO" "$local_dirs" "$SKIP_EXT" "$PATCHES"/*.patch <<'EOF'
import json, os, sys
out, commit, src, dino, local_dirs, skip_ext, *patches = sys.argv[1:]
json.dump({"commit": commit, "patches": [os.path.basename(p) for p in patches], "src": src,
           "dino": dino, "weights": json.loads(local_dirs), "extensions": skip_ext != "1"}, open(out, "w"), indent=2)
EOF
log "ready: $(tr -d '\n ' < "$TRELLIS2_HOME/ready.json")"
