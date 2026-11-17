"""TRELLIS.2 image-to-3D as a queue handler: one image in, a textured GLB out (512 pipeline, fp16 flow models).

  python3 laptop/submit.py --kind generate --family trellis2 --config trellis2-stylized@1 \
      --params '{"seed": 3}' --file image.png=character.png

Worker: --handler trellis2=handlers.trellis2:Trellis2   (after worker/trellis2_t4/setup.sh)
Options (--opt): home=<TRELLIS2_HOME> (default /kaggle/working/trellis2, reads its ready.json)
                 dtype=fp16|fp32|bf16 (flow models, default fp16), rembg=none|<hf id> (default none),
                 warmup=0 (skip the warm-up run)

input:   image.png  PNG with a transparent background (RGBA); with --opt rembg=ZhengPeng7/BiRefNet any image
params:  (config values are defaults, params override them)
         seed               int, default 42
         texture_size       GLB texture size in px, default 1024
         decimation_target  target face count of the GLB, default 50000
         remesh, remesh_band, remesh_project   passed to o_voxel.postprocess.to_glb
         sparse_structure_steps, shape_steps, tex_steps   sampler steps (upstream default 12 each)
outputs: model.glb    the textured mesh (WebP textures)
         result.json  settings used, stage timings, face/vertex counts, whether fp32 was needed

If the fp16 run produces a non-finite or empty mesh, the job is run again with the flow models in fp32 and
result.json says so ("fp32_fallback": true). If the GLB export runs out of GPU memory, it is retried once with
half the texture size.
"""
import io, json, os, tempfile, time

# Must be set before TRELLIS.2 (or torch) is imported: sparse attention reads them once at import.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from runtime import Handler

DEFAULTS = {"seed": 42, "texture_size": 1024, "decimation_target": 50000,
            "remesh": True, "remesh_band": 1, "remesh_project": 0}
STEPS = {"sparse_structure_steps": "sparse_structure_sampler_params", "shape_steps": "shape_slat_sampler_params",
         "tex_steps": "tex_slat_sampler_params"}


def settings(config, params):
    """Job settings: DEFAULTS, then the job's config, then its params. Unknown keys are an error, so a typo
    doesn't silently fall back to a default."""
    s = dict(DEFAULTS)
    for source in (config or {}, params or {}):
        source = {k: v for k, v in source.items() if not k.startswith("_")}     # "_note" etc. in configs
        unknown = set(source) - set(DEFAULTS) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}; known: {sorted(set(DEFAULTS) | set(STEPS))}")
        s.update(source)
    return s


def mesh_ok(mesh):
    import torch
    return mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0 and bool(torch.isfinite(mesh.vertices).all()) \
        and bool(torch.isfinite(mesh.attrs).all())


class Trellis2(Handler):
    def load(self):
        import torch
        o = self.options
        home = o.get("home") or os.environ.get("TRELLIS2_HOME") or "/kaggle/working/trellis2"
        if not os.path.exists(os.path.join(home, "ready.json")):
            raise RuntimeError(f"trellis2 not set up: no {home}/ready.json (run worker/trellis2_t4/setup.sh first)")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        # FlexGEMM tunes its Triton kernels per shape on first use and saves the results; one file per GPU
        # process so two workers don't write the same file. No tuned configs ship for the T4.
        os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.path.join(home, f"flex_gemm_autotune_gpu{gpu}.json"))
        from trellis2_t4 import pipeline_512
        self.pipeline_512 = pipeline_512
        t0 = time.time()
        self.dtype = o.get("dtype") or os.environ.get("TRELLIS_DTYPE") or "fp16"
        self.pipe = pipeline_512.from_ready(home, rembg=o.get("rembg") or pipeline_512.NO_REMBG, dtype=self.dtype)
        self.pipe.cuda()
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            import flex_gemm.kernels.triton.spconv.config as fg
            fg.allow_tf32 = False          # no TF32 below compute capability 8.0 (matters only for fp32 fallbacks)
        import o_voxel
        self.o_voxel = o_voxel
        self.load_s = round(time.time() - t0, 1)
        print(f"trellis2 ready in {self.load_s}s: flow models {self.dtype}", flush=True)

    def info(self):
        import torch
        return {"dtype": getattr(self, "dtype", None), "load_s": getattr(self, "load_s", None),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}

    def warmup(self):
        """A short run on a synthetic image so Triton compiles and FlexGEMM tunes before the first real job."""
        if self.options.get("warmup") == "0":
            return {}
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((96, 96, 416, 416), fill=(200, 80, 40, 255))
        t0 = time.time()
        s = settings(None, {"sparse_structure_steps": 2, "shape_steps": 2, "tex_steps": 2, "texture_size": 512,
                            "decimation_target": 20000})
        mesh, _ = self.generate(img, s)
        self.export(mesh, s)
        return {"warmup_s": round(time.time() - t0, 1)}

    def generate(self, image, s):
        import torch
        kw = {STEPS[k]: {"steps": int(s[k])} for k in STEPS if k in s}
        t = {}
        t0 = time.time()
        with torch.no_grad():
            mesh = self.pipe.run(image, seed=int(s["seed"]), pipeline_type="512", **kw)[0]
        t["generate_s"] = round(time.time() - t0, 1)
        return mesh, t

    def export(self, mesh, s):
        """GLB bytes; on CUDA OOM retried once with half the texture size. Returns (bytes, texture_size)."""
        import torch
        size = int(s["texture_size"])
        for attempt in (0, 1):
            try:
                glb = self.o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=int(s["decimation_target"]), texture_size=size, remesh=bool(s["remesh"]),
                    remesh_band=s["remesh_band"], remesh_project=s["remesh_project"])
                with tempfile.TemporaryDirectory() as d:          # same call as upstream's example.py
                    glb.export(os.path.join(d, "model.glb"), extension_webp=True)
                    with open(os.path.join(d, "model.glb"), "rb") as f:
                        return f.read(), size, glb
            except torch.cuda.OutOfMemoryError:
                if attempt:
                    raise
                torch.cuda.empty_cache()
                size //= 2
                print(f"GLB export ran out of GPU memory, retrying with texture_size={size}", flush=True)

    def run(self, job, ctx):
        from PIL import Image
        if "image.png" not in job["files"]:
            raise ValueError("upload the input as image.png (--file image.png=<your file>)")
        s = settings(ctx.config(), ctx.params)
        image = Image.open(io.BytesIO(ctx.input("image.png")))
        image.load()
        ctx.progress(stage="generate", dtype=self.dtype)
        mesh, timings = self.generate(image, s)
        fallback = False
        if not mesh_ok(mesh) and self.dtype != "fp32":
            ctx.progress(stage="generate", dtype="fp32", note="fp16 result was not finite, rerunning in fp32")
            self.pipeline_512.set_flow_dtype(self.pipe, "fp32")
            try:
                mesh, t32 = self.generate(image, s)
            finally:
                self.pipeline_512.set_flow_dtype(self.pipe, self.dtype)
            timings["generate_fp32_s"] = t32["generate_s"]
            fallback = True
        if not mesh_ok(mesh):
            raise RuntimeError("TRELLIS.2 produced an empty or non-finite mesh"
                               + (" even in fp32" if fallback else ""))
        ctx.progress(stage="export")
        t0 = time.time()
        data, texture_size, glb = self.export(mesh, s)
        timings["export_s"] = round(time.time() - t0, 1)
        ctx.output("model.glb", data)
        result = {"settings": s, "texture_size_used": texture_size, "fp32_fallback": fallback,
                  "flow_dtype": self.dtype, "timings": timings,
                  "raw_mesh": {"vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0])},
                  "glb": {"bytes": len(data), "faces": int(len(glb.faces))}}
        ctx.output("result.json", json.dumps(result, indent=2))
        return {"fp32_fallback": fallback, "glb_mb": round(len(data) / 2**20, 2), **timings}

    def close(self):
        self.pipe = None
