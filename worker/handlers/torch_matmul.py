"""Example GPU handler: a matrix-multiply benchmark. Shows the pattern for a real model handler (load once,
run many jobs) and measures what a provider's GPU actually delivers.

  python3 laptop/submit.py --kind bench --family matmul --params '{"size": 8192, "iters": 50, "dtype": "float16"}'

Worker: --handler matmul=handlers.torch_matmul:Matmul   (needs torch; uses the notebook's CUDA build)

params: size (default 4096), iters (default 20), dtype float16|bfloat16|float32 (default float16).
Note: T4 (sm_75) has no fast bfloat16; use float16 there.
"""
import time

from runtime import Handler


class Matmul(Handler):
    def load(self):
        import torch
        self.torch = torch

    def info(self):
        t = self.torch
        if self.device.startswith("cuda"):
            p = t.cuda.get_device_properties(0)
            return {"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                    "compute_capability": f"{p.major}.{p.minor}", "torch": t.__version__, "cuda": t.version.cuda}
        return {"gpu": None, "torch": t.__version__}

    def warmup(self):
        self._bench(512, 2, "float32")
        return {}

    def _bench(self, size, iters, dtype):
        t = self.torch
        dt = getattr(t, dtype)
        a = t.randn(size, size, device=self.device, dtype=dt)
        b = t.randn(size, size, device=self.device, dtype=dt)
        sync = t.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        a @ b
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        sync()
        s = time.perf_counter() - t0
        return s, bool(t.isfinite(c).all())

    def run(self, job, ctx):
        p = ctx.params
        size, iters, dtype = int(p.get("size", 4096)), int(p.get("iters", 20)), p.get("dtype", "float16")
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("dtype must be float16, bfloat16 or float32")
        ctx.progress(phase="matmul", size=size, iters=iters)
        s, finite = self._bench(size, iters, dtype)
        tflops = 2 * size ** 3 * iters / s / 1e12
        return {"size": size, "iters": iters, "dtype": dtype, "seconds": round(s, 4),
                "tflops": round(tflops, 2), "finite": finite, "score": round(tflops, 2)}
