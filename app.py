"""Gradio demo: upload a magnetic tile image, get the INT8 model's verdict and a Grad-CAM heatmap.

    python app.py      # http://127.0.0.1:7860, local only

Classification runs on the quantized ONNX model, as it would on an edge device.
Grad-CAM needs gradients, so the heatmap comes from the FP32 PyTorch model the
INT8 model was converted from (they agree on the test set, see the README).
"""

import json

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image

from tiles import data
from tiles.export import session, softmax
from tiles.gradcam import grad_cam
from tiles.model import load_checkpoint

INT8 = data.ROOT / "models" / "mobilenet_v3_small_int8.onnx"
CHECKPOINT = data.ROOT / "models" / "mobilenet_v3_small.pt"
METRICS = data.ROOT / "reports" / "metrics.json"


def alarm_threshold() -> float:
    """The defect-alarm threshold chosen on validation by tiles.evaluate (98% recall target)."""
    try:
        return float(json.loads(METRICS.read_text())["models"]["mobilenet_v3_small"]["alarm"]["test"]["threshold"])
    except (FileNotFoundError, KeyError):
        return 0.5


onnx_session = session(INT8)
model, _ = load_checkpoint(CHECKPOINT)
transform = data.eval_transform()
colormap = matplotlib.colormaps["jet"]
THRESHOLD = alarm_threshold()


def analyse(image: Image.Image):
    if image is None:
        return {}, None, ""
    gray = image.convert("L")
    x = transform(gray)
    probs = softmax(onnx_session.run(None, {"image": x.numpy()[None]})[0])[0]
    p_defect = 1 - float(probs[0])
    alarm = p_defect >= THRESHOLD
    # Explain the most likely defect when the alarm fires, otherwise the "Free" decision.
    target = 1 + int(probs[1:].argmax()) if alarm else 0
    cam, _ = grad_cam(model, x, target=target)

    base = np.asarray(gray.resize((data.IMG_SIZE, data.IMG_SIZE)), dtype=np.float32)[..., None] / 255
    overlay = (0.55 * base + 0.45 * colormap(cam)[..., :3]).clip(0, 1)
    if alarm:
        verdict = f"DEFECT ALARM: most likely {data.CLASSES[target]} (defect probability {p_defect:.0%}, alarm at {THRESHOLD:.0%})"
    else:
        verdict = f"Good tile (defect probability {p_defect:.0%}, alarm at {THRESHOLD:.0%})"
    return {c: float(p) for c, p in zip(data.CLASSES, probs)}, Image.fromarray((overlay * 255).astype(np.uint8)), verdict


def examples() -> list[list[str]]:
    """One held-out test image per class, if the dataset has been downloaded."""
    try:
        frame = data.load_splits()
    except FileNotFoundError:
        return []
    test = frame[frame["split"] == "test"]
    picks = [test[test["label"] == c]["path"].iloc[0] for c in data.CLASSES if (test["label"] == c).any()]
    return [[str(data.DATA_DIR / p)] for p in picks if (data.DATA_DIR / p).exists()]


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Magnetic tile image"),
    outputs=[gr.Label(num_top_classes=6, label="INT8 model"), gr.Image(label="Grad-CAM (where the model looked)"), gr.Textbox(label="Verdict")],
    examples=examples(),
    title="Magnetic tile defect inspection",
    description="MobileNetV3-Small fine-tuned on the Magnetic Tile Defect dataset, quantized to INT8 with ONNX Runtime.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
