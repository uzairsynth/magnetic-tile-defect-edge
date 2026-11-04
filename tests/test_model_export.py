import numpy as np
import onnxruntime as ort
import torch

from tiles import data
from tiles.export import export_onnx, quantize
from tiles.gradcam import energy_inside, grad_cam, pointing_hit
from tiles.metrics import classification_summary
from tiles.model import build_model


def test_model_outputs_one_logit_per_class():
    model = build_model("mobilenet_v3_small", len(data.CLASSES), pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 96, 96))
    assert tuple(out.shape) == (2, len(data.CLASSES))


def test_grad_cam_matches_input_size_and_range():
    torch.manual_seed(0)
    model = build_model("mobilenet_v3_small", len(data.CLASSES), pretrained=False)
    cam, target = grad_cam(model, torch.randn(3, 96, 96))
    assert cam.shape == (96, 96)
    assert cam.min() >= 0 and cam.max() <= 1
    assert 0 <= target < len(data.CLASSES)


def test_pointing_game_and_energy():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    on = np.zeros((50, 50))
    on[15, 15] = 1.0
    off = np.zeros((50, 50))
    off[40, 40] = 1.0
    assert pointing_hit(on, mask, tolerance=2)
    assert not pointing_hit(off, mask, tolerance=2)
    assert energy_inside(on, mask) == 1.0 and energy_inside(off, mask) == 0.0


def test_onnx_fp32_matches_pytorch_and_int8_runs(tmp_path):
    torch.manual_seed(0)
    model = build_model("mobilenet_v3_small", len(data.CLASSES), pretrained=False).eval()
    fp32, int8 = tmp_path / "m_fp32.onnx", tmp_path / "m_int8.onnx"
    export_onnx(model, fp32, img_size=64)
    x = np.random.default_rng(0).normal(size=(4, 3, 64, 64)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    assert np.allclose(got, expected, atol=1e-4)

    quantize(fp32, int8, x)
    q = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]).run(None, {"image": x[:1]})[0]
    assert q.shape == (1, len(data.CLASSES)) and np.isfinite(q).all()


def test_summary_counts_defects_and_false_alarms():
    y = np.array([0, 0, 0, 1, 2, 3])
    probs = np.eye(6)[[0, 0, 1, 1, 0, 3]]  # one false alarm, one missed defect
    s = classification_summary(y, probs, data.CLASSES)
    d = s["defect_detection"]
    assert (d["defects"], d["caught"], d["false_alarms"]) == (3, 2, 1)
    assert np.isclose(d["recall"], 2 / 3)
