"""Load TRELLIS.2 with only the models its 512 pipeline uses.

Upstream's Trellis2ImageTo3DPipeline.from_pretrained loads every model named in pipeline.json, including the two
1024 flow models (1.3B parameters each) that pipeline_type='512' never touches, and always builds the background
remover (briaai/RMBG-2.0, CC BY-NC). This module writes a reduced pipeline.json and loads from that, so no
TRELLIS.2 source file has to change for it.

    from trellis2_t4.pipeline_512 import from_ready, load
    pipe = from_ready()                                       # what worker/trellis2_t4/setup.sh prepared
    pipe = load("/kaggle/working/trellis2/config")            # weights straight from Hugging Face
    pipe = load(cfg_dir, dino="/tmp/dinov3-vitl16-hf",
                local_dirs={"microsoft/TRELLIS.2-4B": "/tmp/trellis2-weights/microsoft/TRELLIS.2-4B"})

    python3 pipeline_512.py download /tmp/trellis2-weights    # the 512 files only; prints the local_dirs JSON
    python3 pipeline_512.py config                            # the reduced pipeline.json

load() needs TRELLIS.2 importable (setup.sh puts it in $TRELLIS2_HOME/src; from_ready() adds that to sys.path).
"""
import argparse, copy, json, os, sys

REPO = "microsoft/TRELLIS.2-4B"
MODELS_512 = ("sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
              "shape_slat_decoder", "tex_slat_flow_model_512", "tex_slat_decoder")
FLOW_MODELS = ("sparse_structure_flow_model", "shape_slat_flow_model_512", "tex_slat_flow_model_512")
DTYPES = ("fp16", "fp32", "bf16")   # flow-model precision; the checkpoints are bf16, the T4 has no bf16 tensor cores
NO_REMBG = "none"


class NoBackgroundRemoval:
    """Stands in for the background remover. Upstream calls it only for images without transparency."""
    def to(self, *a, **k): return self
    def cuda(self): return self
    def cpu(self): return self

    def __call__(self, image):
        raise ValueError("input needs a transparent background (RGBA with some alpha < 255): "
                         "background removal is switched off (rembg='none')")


def config_512(official, rembg=NO_REMBG, dino=None, local_dirs=None):
    """The official pipeline.json reduced to the 512 path.

    rembg: 'none' (inputs must have transparency) or a Hugging Face id for upstream's BiRefNet loader,
           e.g. 'ZhengPeng7/BiRefNet' (MIT) instead of the default briaai/RMBG-2.0 (CC BY-NC).
    dino: replaces the DINOv3 model id or path (e.g. a local copy if the official one is gated).
    local_dirs: {"owner/repo": "/local/dir"}; model paths under that repo are read from the directory instead.
    """
    cfg = copy.deepcopy(official)
    args = cfg["args"]
    missing = [k for k in MODELS_512 if k not in args["models"]]
    if missing:
        raise KeyError(f"pipeline.json has no {missing}; upstream changed its model names")
    models = {}
    for k in MODELS_512:
        v = args["models"][k]
        if v.startswith("ckpts/"):
            repo, rest = REPO, v                                          # "ckpts/x": a file of REPO
        else:
            owner, name, rest = v.split("/", 2)                           # "owner/repo/ckpts/x"
            repo = f"{owner}/{name}"
        base = (local_dirs or {}).get(repo)
        models[k] = os.path.join(base, rest) if base else f"{repo}/{rest}"
    args["models"] = models
    args["default_pipeline_type"] = "512"
    args["low_vram"] = True
    if rembg == NO_REMBG:
        args["rembg_model"] = {"name": "NoBackgroundRemoval", "args": {}}
    else:
        args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": rembg}}
    if dino:
        args["image_cond_model"]["args"]["model_name"] = dino
    return cfg


def official_config(local_dirs=None):
    base = (local_dirs or {}).get(REPO)
    if base:
        path = os.path.join(base, "pipeline.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, "pipeline.json")
    with open(path) as f:
        return json.load(f)


def weight_files(official):
    """{repo: [file, ...]}: every file the 512 pipeline reads from Hugging Face (pipeline.json and each model's
    .json and .safetensors). Derived from config_512, so it can't drift from what load() opens."""
    files = {REPO: ["pipeline.json"]}
    for path in config_512(official)["args"]["models"].values():
        owner, name, rest = path.split("/", 2)
        files.setdefault(f"{owner}/{name}", []).extend([rest + ".json", rest + ".safetensors"])
    return files


def download(dest, max_workers=8):
    """Download the 512 files into dest/<owner>/<repo>/ (resumes, skips complete files) and return local_dirs."""
    from huggingface_hub import hf_hub_download, snapshot_download
    official_path = hf_hub_download(REPO, "pipeline.json", local_dir=os.path.join(dest, REPO))
    with open(official_path) as f:
        official = json.load(f)
    local_dirs = {}
    for repo, names in weight_files(official).items():
        local_dirs[repo] = os.path.join(dest, repo)
        snapshot_download(repo, allow_patterns=names, local_dir=local_dirs[repo], max_workers=max_workers)
        missing = [n for n in names if not os.path.isfile(os.path.join(local_dirs[repo], n))]
        if missing:
            raise FileNotFoundError(f"{repo}: download incomplete, missing {missing}")
    return local_dirs


def set_flow_dtype(pipe, dtype):
    """Run the three flow models in fp16, fp32 or bf16 with upstream's own convert_to(), which converts the
    transformer blocks and leaves the embedders, output layers and LayerNorm32 layers in fp32. Raises if a weight
    doesn't fit the new type (bf16 reaches 3e38, fp16 only 65504)."""
    import torch
    from trellis2.modules.utils import str_to_dtype
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r}: use one of {DTYPES}")
    for name in FLOW_MODELS:
        model = pipe.models[name]
        model.convert_to(str_to_dtype(dtype))
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            raise ValueError(f"{name} has weights that overflow {dtype}: {bad[:5]}")
    pipe.flow_dtype = dtype


def load(cfg_dir, rembg=NO_REMBG, dino=None, local_dirs=None, official=None, dtype=None):
    """Write the reduced pipeline.json to cfg_dir (writable) and load the pipeline from it, on the CPU.
    Upstream's low_vram mode then moves one model at a time to the GPU during run().
    dino: DINOv3 model directory or id, default $TRELLIS2_DINO or the official (gated) id.
    dtype: flow-model precision, default $TRELLIS_DTYPE or fp16."""
    dino = dino or os.environ.get("TRELLIS2_DINO")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg as rembg_mod
    rembg_mod.NoBackgroundRemoval = NoBackgroundRemoval
    cfg = config_512(official or official_config(local_dirs), rembg, dino, local_dirs)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "pipeline.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # Model paths are absolute or "owner/repo/..." ids; upstream tries f"{cfg_dir}/{path}" first and falls back
    # to the path itself, the same way it already loads sparse_structure_decoder from TRELLIS-image-large.
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(cfg_dir)
    if sorted(pipe.models) != sorted(MODELS_512):
        raise RuntimeError(f"loaded {sorted(pipe.models)}, expected {sorted(MODELS_512)}")
    set_flow_dtype(pipe, dtype or os.environ.get("TRELLIS_DTYPE", "fp16"))
    return pipe


def from_ready(home=None, **kw):
    """load() with what worker/trellis2_t4/setup.sh recorded in $TRELLIS2_HOME/ready.json: the patched source,
    the downloaded weights and the DINOv3 copy. Keyword arguments (rembg, dtype, ...) are passed on."""
    home = home or os.environ.get("TRELLIS2_HOME", "/kaggle/working/trellis2")
    with open(os.path.join(home, "ready.json")) as f:
        ready = json.load(f)
    if ready["src"] not in sys.path:
        sys.path.insert(0, ready["src"])
    kw.setdefault("dino", ready.get("dino"))
    kw.setdefault("local_dirs", ready.get("weights"))
    return load(os.path.join(home, "config"), **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config", help="print the reduced pipeline.json")
    sub.add_parser("download", help="download the 512 files, print local_dirs as JSON").add_argument("dest")
    a = ap.parse_args()
    if a.cmd == "config":
        print(json.dumps(config_512(official_config()), indent=2))
    else:
        print(json.dumps(download(a.dest)))
