"""kimodo text encoder (LLM2Vec, Llama-3 Q8_0) as a queue handler: prompts in, 4096-float embeddings out.

One worker per GPU keeps one encoder resident in VRAM (about 9 GB of a T4's 15 GB) inside a kmd-encode-serve
process, so jobs can be submitted at any time during the session and are answered without reloading.

  python3 laptop/kimodo_prompts.py "a person walks forward and waves" "someone jumps twice"
  python3 laptop/submit.py --kind encode --family kimodo --params '{"prompts": ["a person waves"]}'

Worker: --handler kimodo=handlers.kimodo_encode:KimodoEncode   (after worker/kimodo/setup.sh)
Options (--opt): home=<KIMODO_HOME> (default /kaggle/working/kimodo, reads its ready.json),
                 bin=..., text_model=..., lib_dir=... (override ready.json), allow_cpu=1 (testing only)

params:  prompts  list of strings (or a prompts.txt input file, one prompt per line)
outputs: embeddings.f32  row-major float32 [len(prompts), 4096], little-endian
                         (numpy: np.fromfile(f, "<f4").reshape(-1, 4096)); feeds kimodo_generate_embedding
         result.json     prompts in row order, per-prompt encode times, backend
A prompt the tokenizer rejects (more than 511 tokens) fails the whole job with its index.
"""
import json, os, struct, subprocess, threading, time

from runtime import Handler

DIM = 4096


class KimodoEncode(Handler):
    def load(self):
        o = self.options
        home = o.get("home") or os.environ.get("KIMODO_HOME") or "/kaggle/working/kimodo"
        ready = {}
        if os.path.exists(os.path.join(home, "ready.json")):
            ready = json.load(open(os.path.join(home, "ready.json")))
        self.bin = o.get("bin") or ready.get("bin")
        self.model = o.get("text_model") or ready.get("text_model")
        if not self.bin or not self.model:
            raise RuntimeError(f"kimodo not set up: no {home}/ready.json (run worker/kimodo/setup.sh first) "
                               "and no --opt bin=... text_model=...")
        lib_dir = o.get("lib_dir") or ready.get("lib_dir") or os.path.dirname(self.bin)
        env = {**os.environ,
               "KIMODO_BACKEND": "cpu" if self.device == "cpu" else "cuda",
               "KIMODO_TEXT_LAYER_CHUNK": "32",          # whole encoder resident: ~50 ms per prompt, not ~1 s
               "LD_LIBRARY_PATH": lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
        env.pop("KIMODO_TEXT_RESIDENT_LIMIT_MIB", None)
        t0 = time.time()
        self.proc = subprocess.Popen([self.bin, self.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env)
        self.stderr_tail = []
        first = ""
        for raw in self.proc.stderr:  # ggml prints device info first; EOF means the process died while loading
            line = raw.decode(errors="replace").strip()
            if line.startswith("ready backend="):
                first = line
                break
            self.stderr_tail = (self.stderr_tail + [line])[-20:]
            print("kmd-encode-serve:", line, flush=True)
        if not first:
            raise RuntimeError(f"kmd-encode-serve failed to start (exit {self.proc.wait(10)}): "
                               + " | ".join(self.stderr_tail))
        threading.Thread(target=self._drain, daemon=True).start()
        self.backend = first.split("=", 1)[1]
        if not self.backend.startswith("CUDA") and o.get("allow_cpu") != "1":
            self.close()
            raise RuntimeError(f"kimodo encoder came up on {self.backend}, not CUDA "
                               "(build without CUDA, or no GPU visible); --opt allow_cpu=1 to accept")
        self.load_s = round(time.time() - t0, 2)
        print(f"kimodo encoder ready on {self.backend} in {self.load_s}s", flush=True)

    def _drain(self):
        for line in self.proc.stderr:
            self.stderr_tail = (self.stderr_tail + [line.decode(errors="replace").strip()])[-20:]

    def info(self):
        return {"backend": getattr(self, "backend", None), "text_model": os.path.basename(self.model or ""),
                "load_s": getattr(self, "load_s", None)}

    def encode(self, prompt):
        data = prompt.encode()
        if not data:
            raise ValueError("empty prompt")
        try:
            self.proc.stdin.write(struct.pack("<I", len(data)) + data)
            self.proc.stdin.flush()
            head = self.proc.stdout.read(4)
        except (BrokenPipeError, OSError):
            head = b""
        if len(head) != 4:  # process died: let the job fail, the worker restarts and reloads cleanly
            raise RuntimeError(f"kmd-encode-serve exited ({self.proc.poll()}): {' | '.join(self.stderr_tail)}")
        if struct.unpack("<I", head)[0] != 0:
            n, = struct.unpack("<I", self.proc.stdout.read(4))
            raise ValueError(self.proc.stdout.read(n).decode(errors="replace"))
        out = self.proc.stdout.read(DIM * 4)
        if len(out) != DIM * 4:
            raise RuntimeError("short read from kmd-encode-serve")
        return out

    def warmup(self):
        t0 = time.time()
        self.encode("a person walks forward")
        return {"warmup_s": round(time.time() - t0, 3)}

    def run(self, job, ctx):
        prompts = ctx.params.get("prompts")
        if prompts is None and ctx.params.get("prompt") is not None:
            prompts = [ctx.params["prompt"]]
        if prompts is None and "prompts.txt" in job["files"]:
            prompts = [l for l in ctx.input("prompts.txt").decode().splitlines() if l.strip()]
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("params.prompts must be a non-empty list of strings (or upload prompts.txt)")
        rows, times = [], []
        for i, p in enumerate(prompts):
            t0 = time.time()
            try:
                rows.append(self.encode(p))
            except ValueError as e:
                raise ValueError(f"prompt {i} ({p[:60]!r}): {e}") from None
            times.append(round(time.time() - t0, 4))
            if i % 16 == 15:
                ctx.progress(done=i + 1, total=len(prompts))
        ctx.output("embeddings.f32", b"".join(rows))
        ctx.output("result.json", json.dumps({"prompts": prompts, "shape": [len(prompts), DIM], "dtype": "<f4",
                                              "encode_s": times, "backend": self.backend}, indent=2))
        return {"prompts": len(prompts), "encode_s_total": round(sum(times), 3),
                "encode_ms_mean": round(1000 * sum(times) / len(times), 1)}

    def close(self):
        p = getattr(self, "proc", None)
        if p and p.poll() is None:
            try:
                p.stdin.write(struct.pack("<I", 0)); p.stdin.flush()
                p.wait(10)
            except Exception:
                p.kill()
