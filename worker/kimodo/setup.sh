#!/usr/bin/env bash
# Prepare kimodo text encoding on a notebook GPU: build kimodo.cpp with CUDA (sm_75 for the T4) and fetch the
# Q8_0 text encoder. Run as a session phase:
#   --phases preflight,kimodo,workers --cmd "kimodo=bash worker/kimodo/setup.sh"
# Custom phases start once per GPU; a lock makes the first copy do the work and the others wait for it.
# Writes $KIMODO_HOME/ready.json, which handlers/kimodo_encode.py reads.
#
# Shortcuts (checked first, both optional):
#   a Kaggle dataset holding Llama-3-Kimodo-Q8_0.gguf + tokenizer.gguf   -> no 8 GB download
#   a Kaggle dataset holding a kimodo-cuda-bin.tar.gz made by this script -> no CUDA build (~10 min)
set -euo pipefail

if [ -d /kaggle/working ]; then KIMODO_HOME=${KIMODO_HOME:-/kaggle/working/kimodo}; else KIMODO_HOME=${KIMODO_HOME:-$PWD/kimodo}; fi
WEIGHTS_DIR=${KIMODO_WEIGHTS_DIR:-/tmp/kimodo-weights}   # /tmp: larger than /kaggle/working, not saved
KIMODO_REPO=${KIMODO_REPO:-https://github.com/hmcode57/kimodo.cpp.git}   # fork of localai-org/kimodo.cpp
KIMODO_COMMIT=${KIMODO_COMMIT:-e6f6a2d}                  # branch kaggle-cuda: CUDA backend, kmd-encode-serve, BF16 norm fix
CUDA_ARCH=${KIMODO_CUDA_ARCH:-75}                        # T4
HF=https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML/resolve/main
Q8=Llama-3-Kimodo-Q8_0.gguf
Q8_SHA=b26d0e74b115b33a7f2df5aca04426548365e99217cd590986f8abc4172e4c5c
TOK_SHA=81614aca62a98846c02b72cc2e5378e5bdce8b4d507b88f96eb1faa90dae607e

mkdir -p "$KIMODO_HOME" "$WEIGHTS_DIR"
exec 9>"$KIMODO_HOME/.setup.lock"
flock 9                                                   # other GPUs' copies wait here
if [ -f "$KIMODO_HOME/ready.json" ]; then echo "kimodo already set up: $(cat "$KIMODO_HOME/ready.json")"; exit 0; fi
t0=$(date +%s)
log() { echo "[kimodo-setup +$(( $(date +%s) - t0 ))s] $*"; }

# ---- weights (in the background, overlapping the build) ----
fetch_weights() {
  local found
  found=$(find /kaggle/input -name "$Q8" 2>/dev/null | head -1 || true)
  if [ -n "$found" ] && [ -f "$(dirname "$found")/tokenizer.gguf" ]; then
    ln -sf "$found" "$WEIGHTS_DIR/$Q8"; ln -sf "$(dirname "$found")/tokenizer.gguf" "$WEIGHTS_DIR/tokenizer.gguf"
    log "weights from dataset $(dirname "$found")"; return 0
  fi
  for f in tokenizer.gguf "$Q8"; do
    [ -s "$WEIGHTS_DIR/$f" ] && continue
    log "downloading $f"
    curl -fL --retry 5 --retry-delay 5 -C - -o "$WEIGHTS_DIR/$f.part" "$HF/$f"
    mv "$WEIGHTS_DIR/$f.part" "$WEIGHTS_DIR/$f"
  done
  echo "$TOK_SHA  $WEIGHTS_DIR/tokenizer.gguf" | sha256sum -c --quiet
  echo "$Q8_SHA  $WEIGHTS_DIR/$Q8" | sha256sum -c --quiet
  log "weights downloaded and verified"
}
fetch_weights > "$KIMODO_HOME/weights.log" 2>&1 &
WPID=$!

# ---- binary ----
BIN_DIR="$KIMODO_HOME/bin"
prebuilt=$(find /kaggle/input -name kimodo-cuda-bin.tar.gz 2>/dev/null | head -1 || true)
if [ -x "$BIN_DIR/kmd-encode-serve" ]; then
  log "binary already built in $BIN_DIR"
elif [ -n "$prebuilt" ]; then
  log "binary from $prebuilt"; mkdir -p "$BIN_DIR"; tar -xzf "$prebuilt" -C "$BIN_DIR"
else
  export PATH="/usr/local/cuda/bin:$PATH"
  command -v nvcc >/dev/null || { echo "nvcc not found: this image has no CUDA toolkit" >&2; exit 1; }
  # kimodo needs C++23 (<expected>): GCC 12 or newer
  CXX_BIN=""
  for c in g++ g++-14 g++-13 g++-12; do
    command -v $c >/dev/null || continue
    if echo '#include <expected>
int main(){std::expected<int,int> e{1};return *e-1;}' | $c -std=c++23 -x c++ - -o /tmp/kimodo-cxx-check 2>/dev/null; then
      CXX_BIN=$(command -v $c); break; fi
  done
  if [ -z "$CXX_BIN" ]; then
    log "installing g++-12 (the default compiler lacks C++23 <expected>)"
    (apt-get update -qq && apt-get install -y -qq g++-12) > "$KIMODO_HOME/apt.log" 2>&1
    CXX_BIN=$(command -v g++-12)
  fi
  CC_BIN=${CXX_BIN/g++/gcc}
  python3 - <<'EOF' || pip install -q "cmake>=3.25" ninja
import shutil, subprocess, sys
v = subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout.split()[2] if shutil.which("cmake") else "0"
sys.exit(0 if tuple(map(int, v.split(".")[:2])) >= (3, 25) and shutil.which("ninja") else 1)
EOF
  hash -r
  SRC="$KIMODO_HOME/src"
  if [ ! -d "$SRC/.git" ]; then
    log "cloning kimodo.cpp at $KIMODO_COMMIT"
    git clone -q "$KIMODO_REPO" "$SRC"
    git -C "$SRC" checkout -q "$KIMODO_COMMIT"
    git -C "$SRC" submodule update -q --init --recursive
  fi
  # GGML_CUDA_NO_VMM: ggml's VMM pool links libcuda.so directly (CUDA::cuda_driver), which CMake doesn't find on
  # the CUDA 13 Kaggle image ("target was not found"). The legacy pool only holds per-prompt scratch buffers; the
  # ~9 GB resident encoder is unaffected.
  log "building with $CXX_BIN, nvcc $(nvcc --version | grep -o 'release [0-9.]*'), sm_$CUDA_ARCH"
  cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DKIMODO_BUILD_TESTS=OFF \
    -DKIMODO_ENABLE_VULKAN=OFF -DKIMODO_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_C_COMPILER="$CC_BIN" -DCMAKE_CXX_COMPILER="$CXX_BIN" -DCMAKE_CUDA_HOST_COMPILER="$CXX_BIN" \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DGGML_CUDA_NO_VMM=ON > "$KIMODO_HOME/cmake.log" 2>&1 \
    || { tail -40 "$KIMODO_HOME/cmake.log"; exit 1; }
  cmake --build "$SRC/build" --target kmd-encode-serve -j"$(nproc)" > "$KIMODO_HOME/build.log" 2>&1 \
    || { tail -60 "$KIMODO_HOME/build.log"; exit 1; }
  mkdir -p "$BIN_DIR"
  cp "$SRC/build/kmd-encode-serve" "$BIN_DIR/"
  find "$SRC/build" -name 'libggml*.so*' -exec cp -P {} "$BIN_DIR/" \;
  tar -czf "$KIMODO_HOME/kimodo-cuda-bin.tar.gz" -C "$BIN_DIR" .
  log "built; to skip the build next time, upload $KIMODO_HOME/kimodo-cuda-bin.tar.gz as a Kaggle dataset"
fi

log "waiting for weights"
if ! wait $WPID; then cat "$KIMODO_HOME/weights.log"; exit 1; fi
cat "$KIMODO_HOME/weights.log"
cat > "$KIMODO_HOME/ready.json" <<EOF
{"bin": "$BIN_DIR/kmd-encode-serve", "lib_dir": "$BIN_DIR", "text_model": "$WEIGHTS_DIR/$Q8"}
EOF
log "ready: $(cat "$KIMODO_HOME/ready.json")"
