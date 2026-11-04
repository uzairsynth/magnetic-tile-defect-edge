"""Export the chosen model to ONNX, quantize it to INT8 and benchmark it like an edge device would run it.

    python -m tiles.export

Every variant (PyTorch FP32, ONNX FP32, ONNX INT8 with all layers quantized,
and the shipped mixed-precision INT8) is timed on one CPU thread with batch
size one, the way a camera on a production line would feed them, and scored on
the same test images.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from tiles import data
from tiles.metrics import classification_summary
from tiles.model import load_checkpoint


def export_onnx(model: torch.nn.Module, path: Path, img_size: int = data.IMG_SIZE) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
        opset_version=18,
        external_data=False,
        verbose=False,  # its progress lines contain emoji that crash cp1252 consoles on Windows
    )
    onnx.checker.check_model(onnx.load(str(path)))


def load_images(frame, root: Path) -> np.ndarray:
    transform = data.eval_transform()
    batch = []
    for p in frame["path"]:
        with Image.open(root / p) as im:
            batch.append(transform(im.convert("L")).numpy())
    return np.stack(batch).astype(np.float32)


class Calibration(CalibrationDataReader):
    """Feeds training images one at a time so the quantizer can measure activation ranges."""

    def __init__(self, images: np.ndarray):
        self.items = iter([{"image": img[None]} for img in images])

    def get_next(self):
        return next(self.items, None)


# Only these run in INT8 in the shipped model. MobileNetV3's element-wise ops
# (hard-swish, squeeze-excite multiplies, residual adds) are cheap, and their
# activations are where the ranges are worst: after BatchNorm folding the stem
# outputs reach about +-130, so one shared INT8 scale rounds normal channels to zero.
MIXED_OP_TYPES = ["Conv", "Gemm"]


def quantize_prepared(prepared: Path, int8: Path, calibration_images: np.ndarray, exclude=(), only=None, op_types=None) -> None:
    quantize_static(
        str(prepared),
        str(int8),
        Calibration(calibration_images),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=op_types,
        nodes_to_exclude=list(exclude),
        nodes_to_quantize=list(only) if only else None,
    )


def quantize(fp32: Path, int8: Path, calibration_images: np.ndarray, exclude=(), op_types=None) -> None:
    prepared = fp32.with_name(fp32.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prepared))
    quantize_prepared(prepared, int8, calibration_images, exclude, op_types=op_types)
    prepared.unlink()


def logits_of(path: Path, images: np.ndarray) -> np.ndarray:
    sess = session(path)
    return np.concatenate([sess.run(None, {"image": x[None]})[0] for x in images])


def mixed_precision(fp32: Path, int8: Path, calibration_images: np.ndarray, probe: np.ndarray, target_agreement: float) -> dict:
    """Quantize to INT8 but keep the layers that INT8 damages most in FP32.

    1. Sensitivity: quantize one Conv/Gemm at a time and measure how far the
       logits move on the probe images (validation, never test).
    2. Keep the k most sensitive layers in FP32, growing k until the INT8 model
       agrees with FP32 on at least `target_agreement` of the probe images.
    """
    prepared = fp32.with_name(fp32.stem + "_prep.onnx")
    scratch = fp32.with_name(fp32.stem + "_layer.onnx")
    quant_pre_process(str(fp32), str(prepared))
    reference = logits_of(prepared, probe)
    layers = [n.name for n in onnx.load(str(prepared)).graph.node if n.op_type in ("Conv", "Gemm")]

    damage = {}
    for name in layers:
        quantize_prepared(prepared, scratch, calibration_images[:64], only=[name], op_types=MIXED_OP_TYPES)
        damage[name] = float(np.abs(logits_of(scratch, probe) - reference).mean())
    ranked = sorted(layers, key=damage.get, reverse=True)

    for k in sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, len(ranked)}):
        kept = ranked[:k]
        quantize_prepared(prepared, int8, calibration_images, exclude=kept, op_types=MIXED_OP_TYPES)
        agreement = float((logits_of(int8, probe).argmax(1) == reference.argmax(1)).mean())
        print(f"  {k:2d} of {len(ranked)} layers kept in FP32: agreement with FP32 on validation {agreement:.3f}")
        if agreement >= target_agreement:
            break
    prepared.unlink()
    scratch.unlink(missing_ok=True)
    return {
        "layers": len(ranked),
        "fp32_layers": kept,
        "val_agreement": agreement,
        "target_agreement": target_agreement,
        "most_sensitive": [{"layer": n, "mean_abs_logit_change": round(damage[n], 4)} for n in ranked[:8]],
    }


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def time_single_image(run, images: np.ndarray, warmup: int = 20, repeats: int = 200) -> dict:
    return time_interleaved({"model": run}, images, warmup, repeats)["model"]


def time_interleaved(runs: dict, images: np.ndarray, warmup: int = 20, repeats: int = 300) -> dict:
    """Time several models round-robin, one image at a time.

    Each round runs every model once on the same image, so background load or
    thermal throttling hits all of them equally and their ratios stay fair.
    """
    for i in range(warmup):
        for run in runs.values():
            run(images[i % len(images)][None])
    times = {name: [] for name in runs}
    for i in range(repeats):
        x = images[i % len(images)][None]
        for name, run in runs.items():
            start = time.perf_counter()
            run(x)
            times[name].append((time.perf_counter() - start) * 1000)
    return {name: {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))} for name, t in times.items()}


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, default=data.ROOT / "models" / "mobilenet_v3_small.pt")
    p.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    p.add_argument("--splits", type=Path, default=data.SPLITS_CSV)
    p.add_argument("--out-dir", type=Path, default=data.ROOT)
    p.add_argument("--calibration-images", type=int, default=200)
    p.add_argument("--target-agreement", type=float, default=0.98, help="INT8 vs FP32 top-1 agreement required on validation")
    p.add_argument("--benchmark-only", action="store_true", help="re-time the existing ONNX files without exporting again")
    args = p.parse_args(argv)

    torch.manual_seed(0)
    model, ckpt = load_checkpoint(args.checkpoint)
    models_dir = args.out_dir / "models"
    fp32_path, int8_path = models_dir / f"{ckpt['arch']}_fp32.onnx", models_dir / f"{ckpt['arch']}_int8.onnx"

    frame = data.load_splits(args.splits)
    train = frame[frame["split"] == "train"].sample(n=min(args.calibration_images, int((frame["split"] == "train").sum())), random_state=0)
    test, val = frame[frame["split"] == "test"], frame[frame["split"] == "val"]
    calib, x_val, x_test = load_images(train, args.data_dir), load_images(val, args.data_dir), load_images(test, args.data_dir)
    y_test = test["y"].to_numpy()

    naive_path = models_dir / f"{ckpt['arch']}_int8_naive.onnx"
    report_path = args.out_dir / "reports" / "edge_benchmark.json"
    if args.benchmark_only:
        # Re-time existing models (for example on an idle machine) without re-exporting.
        mixed = json.loads(report_path.read_text())["mixed_precision"]
    else:
        export_onnx(model, fp32_path)
        quantize(fp32_path, naive_path, calib)  # every op in INT8: kept only to show why it is not shipped
        print("Layer sensitivity search (validation images):")
        mixed = mixed_precision(fp32_path, int8_path, calib, x_val, args.target_agreement)

    torch.set_num_threads(1)
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x_test)).numpy()

        def torch_run(x):
            return model(torch.from_numpy(x))

        variants = {"PyTorch FP32": (args.checkpoint, torch_logits, torch_run)}
    for name, path in (("ONNX FP32", fp32_path), ("ONNX INT8, all layers", naive_path), ("ONNX INT8, mixed", int8_path)):
        sess = session(path)
        logits = np.concatenate([sess.run(None, {"image": x[None]})[0] for x in x_test])
        variants[name] = (path, logits, lambda x, s=sess: s.run(None, {"image": x}))

    reference = torch_logits.argmax(1)
    with torch.no_grad():
        latencies = time_interleaved({name: run for name, (_, _, run) in variants.items()}, x_test)
    results = {}
    for name, (path, logits, run) in variants.items():
        summary = classification_summary(y_test, softmax(logits), data.CLASSES)
        latency = latencies[name]
        results[name] = {
            "file_mb": round(Path(path).stat().st_size / 1e6, 2),
            **{k: round(v, 2) for k, v in latency.items()},
            "macro_f1": round(summary["macro_f1"], 4),
            "accuracy": round(summary["accuracy"], 4),
            "defect_recall": round(summary["defect_detection"]["recall"], 4),
            "false_alarm_rate": round(summary["defect_detection"]["false_alarm_rate"], 4),
            "agreement_with_pytorch": round(float((logits.argmax(1) == reference).mean()), 4),
            "max_abs_logit_diff": round(float(np.abs(logits - torch_logits).max()), 5),
        }

    report = {
        "model": ckpt["arch"],
        "test_images": int(len(y_test)),
        "calibration_images": int(len(calib)),
        "threads": 1,
        "batch_size": 1,
        "variants": results,
        "mixed_precision": mixed,
        "environment": {
            "cpu": platform.processor() or platform.machine(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "onnx_opset": 18,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"{'variant':22s} {'MB':>6s} {'median ms':>10s} {'p90 ms':>8s} {'macro-F1':>9s} {'acc':>6s} {'agree':>6s}")
    for name, r in results.items():
        print(f"{name:22s} {r['file_mb']:6.2f} {r['median_ms']:10.2f} {r['p90_ms']:8.2f} {r['macro_f1']:9.3f} {r['accuracy']:6.3f} {r['agreement_with_pytorch']:6.3f}")
    return report


if __name__ == "__main__":
    main()
