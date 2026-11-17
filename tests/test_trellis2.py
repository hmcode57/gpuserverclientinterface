"""TRELLIS.2 patches, the 512-only loader and fp16 flow models, on CPU. Run:
    TRELLIS2_SRC=/path/to/patched/TRELLIS.2 python3 tests/test_trellis2.py
TRELLIS2_SRC is a checkout made by worker/trellis2_t4/setup.sh (it is $TRELLIS2_HOME/src). Needs torch,
torchvision, numpy, pillow, easydict, safetensors, transformers and huggingface_hub; the CUDA-only modules
(cumesh, flex_gemm) are replaced by empty stand-ins, since nothing here runs them. The pipeline.json test
downloads one small file from Hugging Face. Skips everything when TRELLIS2_SRC or torch is missing.
"""
import json, os, sys, tempfile, types, unittest
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
    from trellis2_t4 import pipeline_512

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


if __name__ == "__main__":
    unittest.main()
