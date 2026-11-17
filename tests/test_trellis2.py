"""TRELLIS.2 patches, the 512-only loader and fp16 flow models, on CPU. Run:
    TRELLIS2_SRC=/path/to/patched/TRELLIS.2 python3 tests/test_trellis2.py
TRELLIS2_SRC is a checkout made by worker/trellis2_t4/setup.sh (it is $TRELLIS2_HOME/src). Needs torch,
torchvision, numpy, pillow, easydict, safetensors, transformers and huggingface_hub; the CUDA-only modules
(cumesh, flex_gemm) are replaced by empty stand-ins, since nothing here runs them. The pipeline.json test
downloads one small file from Hugging Face. Those tests skip when TRELLIS2_SRC or torch is missing.

Without torch: the weight list, download(), from_ready() and setup.sh end to end (with a stand-in
huggingface_hub that writes small files, a fake DINOv3 dataset, and a real clone of TRELLIS.2 from GitHub).
"""
import json, os, shutil, subprocess, sys, tempfile, types, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("TRELLIS2_SRC")
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"      # read once when trellis2.modules.sparse is imported
os.environ["ATTN_BACKEND"] = "sdpa"
sys.path[:0] = [os.path.join(ROOT, "worker")] + ([SRC] if SRC else [])

try:
    import torch
except ImportError:
    torch = None

if SRC and torch is not None:
    for name in ("cumesh", "flex_gemm", "flex_gemm.ops", "flex_gemm.ops.grid_sample"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["flex_gemm.ops.grid_sample"].grid_sample_3d = None
    from trellis2.modules.sparse import config as sparse_config, VarLenTensor
    from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention as ssdpa

from trellis2_t4 import pipeline_512  # noqa: E402  (stdlib only at import time)

needs = unittest.skipUnless(SRC and torch is not None, "set TRELLIS2_SRC and install torch")


def naive(q, k, v, q_len, kv_len):
    """Plain softmax attention per sequence in float64. q: [TQ, H, C], k: [TK, H, C], v: [TK, H, Cv]."""
    outs, qo, ko = [], 0, 0
    for ql, kl in zip(q_len, kv_len):
        qi, ki, vi = (t.double().transpose(0, 1) for t in (q[qo:qo + ql], k[ko:ko + kl], v[ko:ko + kl]))
        w = torch.softmax(qi @ ki.transpose(1, 2) / qi.shape[-1] ** 0.5, dim=-1)
        outs.append((w @ vi).transpose(0, 1))
        qo, ko = qo + ql, ko + kl
    return torch.cat(outs)


@needs
class SparseSdpa(unittest.TestCase):
    H, C = 4, 16

    def setUp(self):
        torch.manual_seed(0)

    def varlen(self, lens, *inner):
        feats = torch.randn(sum(lens), *inner)
        return VarLenTensor(feats, VarLenTensor.layout_from_seqlen(lens))

    def close(self, out, ref):
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-4)

    def test_backend_selected_from_env(self):
        self.assertEqual(sparse_config.ATTN, "sdpa")

    def test_self_attention_packed_qkv(self):
        lens = [7, 1, 23]
        qkv = self.varlen(lens, 3, self.H, self.C)
        q, k, v = qkv.feats.unbind(dim=1)
        self.close(ssdpa(qkv).feats, naive(q, k, v, lens, lens))

    def test_cross_attention_sparse_q_dense_kv(self):            # how the flow models attend to the image
        lens, L = [5, 12], 9
        q = self.varlen(lens, self.H, self.C)
        kv = torch.randn(2, L, 2, self.H, self.C)
        k, v = kv.reshape(2 * L, 2, self.H, self.C).unbind(dim=1)
        self.close(ssdpa(q, kv).feats, naive(q.feats, k, v, lens, [L, L]))

    def test_separate_qkv_different_lengths_and_value_width(self):
        q_lens, kv_lens = [3, 8], [10, 2]
        q = self.varlen(q_lens, self.H, self.C)
        k = self.varlen(kv_lens, self.H, self.C)
        v = self.varlen(kv_lens, self.H, 24)
        self.close(ssdpa(q, k, v).feats, naive(q.feats, k.feats, v.feats, q_lens, kv_lens))

    def test_dense_q_sparse_kv(self):
        L, kv_lens = 6, [4, 11]
        q = torch.randn(2, L, self.H, self.C)
        kv = self.varlen(kv_lens, 2, self.H, self.C)
        k, v = kv.feats.unbind(dim=1)
        out = ssdpa(q, kv)
        self.close(out.reshape(2 * L, self.H, self.C), naive(q.reshape(2 * L, self.H, self.C), k, v, [L, L], kv_lens))


@needs
class Loader512(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.official = pipeline_512.official_config()
        except Exception as e:                                    # offline
            raise unittest.SkipTest(f"official pipeline.json unavailable: {e}")

    def test_only_512_models(self):
        cfg = pipeline_512.config_512(self.official)
        args = cfg["args"]
        self.assertEqual(sorted(args["models"]), sorted(pipeline_512.MODELS_512))
        self.assertTrue(all("1024" not in v for v in args["models"].values()))
        self.assertEqual(args["default_pipeline_type"], "512")
        self.assertIs(args["low_vram"], True)
        self.assertEqual(args["rembg_model"]["name"], "NoBackgroundRemoval")
        self.assertIn("1024_cascade", json.dumps(self.official))  # the official file itself is unchanged

    def test_model_paths(self):
        hf = pipeline_512.config_512(self.official)["args"]["models"]
        self.assertTrue(hf["shape_slat_flow_model_512"].startswith("microsoft/TRELLIS.2-4B/ckpts/"))
        self.assertTrue(hf["sparse_structure_decoder"].startswith("microsoft/TRELLIS-image-large/ckpts/"))
        local = pipeline_512.config_512(self.official, local_dirs={"microsoft/TRELLIS.2-4B": "/w/t2"})["args"]["models"]
        self.assertTrue(local["tex_slat_decoder"].startswith("/w/t2/ckpts/"))
        self.assertEqual(local["sparse_structure_decoder"], hf["sparse_structure_decoder"])

    def test_options(self):
        args = pipeline_512.config_512(self.official, rembg="ZhengPeng7/BiRefNet", dino="/w/dino")["args"]
        self.assertEqual(args["rembg_model"], {"name": "BiRefNet", "args": {"model_name": "ZhengPeng7/BiRefNet"}})
        self.assertEqual(args["image_cond_model"]["args"]["model_name"], "/w/dino")

    def test_load_reads_only_512_checkpoints(self):
        import trellis2.models as models
        from trellis2.modules import image_feature_extractor as ife
        from PIL import Image
        loaded = []

        def fake_model(path, **kw):
            loaded.append(path)
            if path.startswith("/"):
                raise FileNotFoundError(path)                     # upstream's first try, f"{cfg_dir}/{path}"
            if "ss_flow" in path:
                return tiny_flow("SparseStructureFlowModel")
            if "slat_flow" in path:
                return tiny_flow("SLatFlowModel")
            return torch.nn.Linear(1, 1)

        class FakeDino:
            def __init__(self, model_name, **kw): self.model_name = model_name
        saved = models.from_pretrained, ife.DinoV3FeatureExtractor
        models.from_pretrained, ife.DinoV3FeatureExtractor = fake_model, FakeDino
        try:
            with tempfile.TemporaryDirectory() as d:
                pipe = pipeline_512.load(d, official=self.official)
        finally:
            models.from_pretrained, ife.DinoV3FeatureExtractor = saved
        expected = set(pipeline_512.config_512(self.official)["args"]["models"].values())
        self.assertEqual({p for p in loaded if not p.startswith("/")}, expected)
        self.assertEqual(sorted(pipe.models), sorted(pipeline_512.MODELS_512))
        self.assertEqual(pipe.default_pipeline_type, "512")
        with self.assertRaisesRegex(ValueError, "transparent background"):
            pipe.preprocess_image(Image.new("RGB", (32, 32)))
        self.assertEqual(pipe.flow_dtype, "fp16")
        for name in pipeline_512.FLOW_MODELS:
            m = pipe.models[name]
            self.assertIs(m.dtype, torch.float16)
            self.assertIs(m.blocks[0].self_attn.to_qkv.weight.dtype, torch.float16)
            self.assertIs(m.input_layer.weight.dtype, torch.float32)   # upstream keeps the edges in fp32


def tiny_flow(cls_name, dtype="bfloat16"):
    """The real flow model classes with the checkpoints' options (rope, shared modulation, qk RMS norm, bf16),
    shrunk to one block and 64 channels, random weights including the zero-initialised output layer."""
    import trellis2.models as models
    torch.manual_seed(0)
    common = dict(model_channels=64, cond_channels=32, num_blocks=1, num_heads=2, mlp_ratio=4, pe_mode="rope",
                  share_mod=True, initialization="scaled", qk_rms_norm=True, qk_rms_norm_cross=True, dtype=dtype)
    if cls_name == "SparseStructureFlowModel":
        m = models.SparseStructureFlowModel(resolution=4, in_channels=8, out_channels=8, **common)
    else:
        m = models.SLatFlowModel(resolution=8, in_channels=8, out_channels=8, **common)
    with torch.no_grad():
        for p in m.out_layer.parameters():
            p.normal_(0, 0.1)
    return m.eval()


@needs
class FlowPrecision(unittest.TestCase):
    """fp16 against fp32 on the same weights. CPU only checks that the fp16 path runs and stays close; the
    T4 numbers come from the parity tool in milestone 2."""

    def pair(self, cls_name):
        import copy
        ref = tiny_flow(cls_name, dtype="float32")
        half = copy.deepcopy(ref)
        half.convert_to(torch.float16)
        return ref, half

    def assert_close_rel(self, a, b, tol=2e-2):
        self.assertTrue(torch.isfinite(b).all())
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        self.assertLess(rel, tol, f"relative error {rel:.3g}")

    def test_sparse_structure_flow_fp16(self):
        ref, half = self.pair("SparseStructureFlowModel")
        x, t, cond = torch.randn(1, 8, 4, 4, 4), torch.tensor([500.0]), torch.randn(1, 5, 32)
        with torch.no_grad():
            self.assert_close_rel(ref(x, t, cond), half(x, t, cond))

    def test_slat_flow_fp16_uses_sparse_sdpa(self):
        from trellis2.modules.sparse import SparseTensor
        ref, half = self.pair("SLatFlowModel")
        coords = torch.unique(torch.randint(0, 8, (40, 3)), dim=0)
        coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int()
        x = SparseTensor(torch.randn(len(coords), 8), coords)
        t, cond = torch.tensor([500.0]), torch.randn(1, 5, 32)
        with torch.no_grad():
            self.assert_close_rel(ref(x, t, cond).feats, half(x, t, cond).feats)

    def test_overflow_is_reported(self):
        pipe = types.SimpleNamespace(models={n: tiny_flow("SLatFlowModel") for n in pipeline_512.FLOW_MODELS})
        with torch.no_grad():
            pipe.models["tex_slat_flow_model_512"].blocks[0].mlp.mlp[0].weight[0, 0] = 1e6
        with self.assertRaisesRegex(ValueError, "tex_slat_flow_model_512 has weights that overflow fp16"):
            pipeline_512.set_flow_dtype(pipe, "fp16")


# The models section of microsoft/TRELLIS.2-4B's pipeline.json (2026-11-17), enough for the file list.
OFFICIAL_MODELS = {
    "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
    "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
    "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
    "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}
OFFICIAL = {"args": {"models": OFFICIAL_MODELS, "image_cond_model": {"args": {"model_name": "facebook/dinov3"}},
                     "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}}}}

# A stand-in huggingface_hub: "downloads" write a few bytes; pipeline.json gets OFFICIAL. Used in-process and,
# through PYTHONPATH, by setup.sh's python3.
FAKE_HUB = '''
import json, os
CALLS = os.environ.get("FAKE_HUB_LOG")
def _write(path, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(%r) if name == "pipeline.json" else "x")
    if CALLS:
        open(CALLS, "a").write(name + "\\n")
def hf_hub_download(repo, name, local_dir=None):
    path = os.path.join(local_dir, name); _write(path, name); return path
def snapshot_download(repo, allow_patterns=(), local_dir=None, max_workers=8):
    for name in allow_patterns:
        _write(os.path.join(local_dir, name), name)
    return local_dir
''' % (OFFICIAL,)


def fake_hub_dir(tmp):
    d = os.path.join(tmp, "fakehub", "huggingface_hub")
    os.makedirs(d)
    open(os.path.join(d, "__init__.py"), "w").write(FAKE_HUB)
    return os.path.dirname(d)


class Weights(unittest.TestCase):
    def test_file_list_is_the_512_files(self):
        files = pipeline_512.weight_files(OFFICIAL)
        self.assertEqual(sorted(files), ["microsoft/TRELLIS-image-large", "microsoft/TRELLIS.2-4B"])
        self.assertEqual(files["microsoft/TRELLIS-image-large"],
                         ["ckpts/ss_dec_conv3d_16l8_fp16.json", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"])
        main = files["microsoft/TRELLIS.2-4B"]
        self.assertEqual(len(main), 1 + 2 * 5)                        # pipeline.json + 5 models x (.json, .safetensors)
        self.assertIn("pipeline.json", main)
        self.assertFalse([f for f in main if "1024" in f])

    def test_download_then_config_points_at_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.path.insert(0, fake_hub_dir(tmp))
            saved = sys.modules.pop("huggingface_hub", None)
            try:
                local = pipeline_512.download(os.path.join(tmp, "w"))
            finally:
                sys.path.pop(0)
                sys.modules.pop("huggingface_hub", None)
                if saved:
                    sys.modules["huggingface_hub"] = saved
            official = pipeline_512.official_config(local)
            for path in pipeline_512.config_512(official, local_dirs=local)["args"]["models"].values():
                for ext in (".json", ".safetensors"):
                    self.assertTrue(os.path.isfile(path + ext), path + ext)

    def test_from_ready_passes_paths(self):
        with tempfile.TemporaryDirectory() as home:
            ready = {"src": "/x/src", "dino": "/tmp/dino", "weights": {"a/b": "/w/a/b"}}
            json.dump(ready, open(os.path.join(home, "ready.json"), "w"))
            seen = {}
            saved = pipeline_512.load
            pipeline_512.load = lambda cfg_dir, **kw: seen.update(kw, cfg_dir=cfg_dir) or "pipe"
            try:
                self.assertEqual(pipeline_512.from_ready(home, dtype="fp32"), "pipe")
            finally:
                pipeline_512.load = saved
                sys.path.remove("/x/src")
        self.assertEqual(seen, {"cfg_dir": os.path.join(home, "config"), "dino": "/tmp/dino",
                                "local_dirs": {"a/b": "/w/a/b"}, "dtype": "fp32"})


@unittest.skipUnless(shutil.which("git") and shutil.which("flock"), "needs git and flock")
class SetupScript(unittest.TestCase):
    """setup.sh end to end without TRELLIS.2: a small fake upstream repo with its own patch, a fake DINOv3
    dataset and a stand-in Hugging Face hub. Extensions are skipped (they need nvcc and torch)."""

    def fake_upstream(self, tmp):
        repo, patches = os.path.join(tmp, "upstream"), os.path.join(tmp, "patches")
        os.makedirs(os.path.join(repo, "pkg")); os.makedirs(patches)
        git = lambda *a: subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout
        git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
        with open(os.path.join(repo, "pkg", "config.py"), "w") as f:
            f.write("BACKENDS = ['xformers', 'flash_attn']\n")
        git("add", "."); git("commit", "-qm", "upstream")
        commit = git("rev-parse", "HEAD").strip()
        with open(os.path.join(repo, "pkg", "config.py"), "w") as f:
            f.write("BACKENDS = ['xformers', 'flash_attn', 'sdpa']\n")
        with open(os.path.join(patches, "0001-fake.patch"), "w") as f:
            f.write(git("diff"))
        git("checkout", "-q", "--", ".")
        return repo, commit, patches

    def run_setup(self, tmp, **env):
        repo, commit, patches = self.fake_upstream(tmp) if not os.path.isdir(os.path.join(tmp, "upstream")) else \
            (os.path.join(tmp, "upstream"), self.commit, os.path.join(tmp, "patches"))
        self.commit = commit
        hub = fake_hub_dir(tmp) if not os.path.isdir(os.path.join(tmp, "fakehub")) else os.path.join(tmp, "fakehub")
        e = dict(os.environ, TRELLIS2_HOME=os.path.join(tmp, "home"), TRELLIS2_INPUT=os.path.join(tmp, "input"),
                 TRELLIS2_WEIGHTS=os.path.join(tmp, "weights"), TRELLIS2_DINO=os.path.join(tmp, "dino"),
                 TRELLIS2_REPO=repo, TRELLIS2_COMMIT=commit, TRELLIS2_PATCHES=patches, TRELLIS2_SKIP_EXT="1",
                 PYTHONPATH=hub, FAKE_HUB_LOG=os.path.join(tmp, "hub.log"), **env)
        return subprocess.run(["bash", os.path.join(ROOT, "worker/trellis2_t4/setup.sh")], env=e,
                              capture_output=True, text=True, timeout=600)

    def fake_dino_dataset(self, tmp):
        d = os.path.join(tmp, "input", "dinov3-vitl16-hf")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": "dinov3_vit", "hidden_size": 1024}, f, indent=2)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(os.urandom(4096))
        return d

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self.fake_dino_dataset(tmp)
            r = self.run_setup(tmp)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(tmp, "home", "ready.json")) as f:
                ready = json.load(f)
            self.assertEqual(ready["patches"], ["0001-fake.patch"])
            self.assertEqual(ready["commit"], self.commit)
            self.assertIs(ready["extensions"], False)
            with open(os.path.join(ready["src"], "pkg/config.py")) as f:
                self.assertIn("'sdpa'", f.read())                    # the patch was applied
            with open(os.path.join(ready["dino"], "model.safetensors"), "rb") as a, \
                 open(os.path.join(src, "model.safetensors"), "rb") as b:
                self.assertEqual(a.read(), b.read())
            self.assertEqual(sorted(ready["weights"]), ["microsoft/TRELLIS-image-large", "microsoft/TRELLIS.2-4B"])
            for repo, names in pipeline_512.weight_files(OFFICIAL).items():
                for n in names:
                    self.assertTrue(os.path.isfile(os.path.join(ready["weights"][repo], n)), n)
            again = self.run_setup(tmp)                              # second GPU's phase: nothing to do
            self.assertEqual(again.returncode, 0)
            self.assertIn("already set up", again.stdout)

    def test_patch_that_no_longer_applies_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fake_dino_dataset(tmp)
            repo, commit, patches = self.fake_upstream(tmp)
            self.commit = commit
            with open(os.path.join(patches, "0001-fake.patch")) as f:
                text = f.read()
            with open(os.path.join(patches, "0001-fake.patch"), "w") as f:
                f.write(text.replace("-BACKENDS = ['xformers', 'flash_attn']", "-BACKENDS = ['something else']"))
            r = self.run_setup(tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(os.path.join(tmp, "home", "ready.json")))

    def test_missing_dino_dataset_fails_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "input"))
            r = self.run_setup(tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no DINOv3", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "hub.log")))   # nothing downloaded


if __name__ == "__main__":
    unittest.main()
