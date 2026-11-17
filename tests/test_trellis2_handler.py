"""worker/handlers/trellis2.py with a fake pipeline and a fake o_voxel: settings, outputs, the fp32 retry and the
export retry. No TRELLIS.2 and no GPU needed. Run: python3 tests/test_trellis2_handler.py (needs torch and
Pillow; skips without them)."""
import io, json, os, sys, types, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "worker")]

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    from handlers import trellis2 as h

needs = unittest.skipUnless(torch is not None, "needs torch and Pillow")


class FakeMesh:
    def __init__(self, finite=True, n=10):
        self.vertices = torch.randn(n, 3) if finite else torch.full((n, 3), float("nan"))
        self.faces = torch.zeros(n, 3, dtype=torch.long)
        self.attrs, self.coords = torch.randn(n, 6), torch.zeros(n, 3)
        self.layout, self.voxel_size = {}, 1 / 512


class FakePipe:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def run(self, image, **kw):
        self.calls.append(kw)
        return [self.results.pop(0)]


class FakeGlb:
    faces = [0] * 1234

    def export(self, path, extension_webp=False):
        with open(path, "wb") as f:
            f.write(b"glTF-fake")


class FakeOVoxel:
    def __init__(self, oom_times=0):
        self.oom_times, self.sizes = oom_times, []
        self.postprocess = types.SimpleNamespace(to_glb=self.to_glb)

    def to_glb(self, **kw):
        self.sizes.append(kw["texture_size"])
        if self.oom_times:
            self.oom_times -= 1
            raise torch.cuda.OutOfMemoryError("fake")
        return FakeGlb()


class Ctx:
    def __init__(self, params=None, config=None):
        self.params, self._config, self.outputs, self.progressed = params or {}, config, {}, []
        buf = io.BytesIO()
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(buf, "PNG")
        self.png = buf.getvalue()

    def config(self): return self._config
    def input(self, name): return self.png
    def output(self, name, data): self.outputs[name] = data
    def progress(self, **f): self.progressed.append(f)


def handler(pipe, ovox):
    t = h.Trellis2("cpu", {})
    t.pipe, t.o_voxel, t.dtype = pipe, ovox, "fp16"
    t.dtypes = []
    t.pipeline_512 = types.SimpleNamespace(set_flow_dtype=lambda p, d: t.dtypes.append(d))
    return t


JOB = {"id": "j1", "files": {"image.png": {}}, "outputs": {}}


@needs
class Settings(unittest.TestCase):
    def test_precedence(self):
        s = h.settings({"_note": "x", "texture_size": 2048, "seed": 1}, {"seed": 7})
        self.assertEqual((s["texture_size"], s["seed"], s["decimation_target"]), (2048, 7, 50000))

    def test_unknown_key_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown settings \\['textur_size'\\]"):
            h.settings(None, {"textur_size": 512})

    def test_shipped_config_is_valid(self):
        with open(os.path.join(ROOT, "configs/trellis2-stylized/v1.json")) as f:
            h.settings(json.load(f), None)


@needs
class Run(unittest.TestCase):
    def test_outputs_and_sampler_steps(self):
        pipe, ovox = FakePipe([FakeMesh()]), FakeOVoxel()
        ctx = Ctx({"seed": 3, "shape_steps": 8}, {"texture_size": 2048})
        metrics = handler(pipe, ovox).run(JOB, ctx)
        self.assertEqual(ctx.outputs["model.glb"], b"glTF-fake")
        result = json.loads(ctx.outputs["result.json"])
        self.assertEqual(result["settings"]["seed"], 3)
        self.assertEqual(result["texture_size_used"], 2048)
        self.assertEqual(result["glb"]["faces"], 1234)
        self.assertFalse(result["fp32_fallback"])
        self.assertEqual(pipe.calls[0], {"seed": 3, "pipeline_type": "512",
                                         "shape_slat_sampler_params": {"steps": 8}})
        self.assertIn("generate_s", metrics)
        self.assertEqual([p["stage"] for p in ctx.progressed], ["generate", "export"])

    def test_non_finite_fp16_reruns_in_fp32_and_switches_back(self):
        pipe = FakePipe([FakeMesh(finite=False), FakeMesh()])
        t = handler(pipe, FakeOVoxel())
        ctx = Ctx()
        t.run(JOB, ctx)
        self.assertEqual(t.dtypes, ["fp32", "fp16"])
        self.assertTrue(json.loads(ctx.outputs["result.json"])["fp32_fallback"])

    def test_non_finite_even_in_fp32_fails(self):
        t = handler(FakePipe([FakeMesh(finite=False), FakeMesh(finite=False)]), FakeOVoxel())
        with self.assertRaisesRegex(RuntimeError, "even in fp32"):
            t.run(JOB, Ctx())
        self.assertEqual(t.dtypes, ["fp32", "fp16"])

    def test_export_oom_retries_with_half_texture(self):
        ovox = FakeOVoxel(oom_times=1)
        ctx = Ctx({"texture_size": 2048})
        handler(FakePipe([FakeMesh()]), ovox).run(JOB, ctx)
        self.assertEqual(ovox.sizes, [2048, 1024])
        self.assertEqual(json.loads(ctx.outputs["result.json"])["texture_size_used"], 1024)

    def test_missing_image(self):
        with self.assertRaisesRegex(ValueError, "image.png"):
            handler(FakePipe([]), FakeOVoxel()).run({"id": "j", "files": {}, "outputs": {}}, Ctx())


if __name__ == "__main__":
    unittest.main()
