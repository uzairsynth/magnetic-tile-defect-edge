"""Classical baseline: hand-made image features and logistic regression.

Answers the question a reviewer should ask first: how much does a pretrained
CNN buy over a simple, fast model on the same leakage-free split?
"""

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tiles.metrics import macro_f1

C_GRID = (0.01, 0.1, 1.0, 10.0)


def image_features(path, root) -> np.ndarray:
    """A 24x24 thumbnail, intensity and gradient-strength histograms, and a few summary statistics."""
    with Image.open(root / path) as im:
        gray = im.convert("L").resize((64, 64))
    g = np.asarray(gray, dtype=np.float32) / 255
    thumb = np.asarray(gray.resize((24, 24)), dtype=np.float32).ravel() / 255
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    return np.concatenate(
        [
            thumb,
            np.histogram(g, 16, (0, 1))[0] / g.size,
            np.histogram(mag, 16, (0, 0.5))[0] / g.size,
            [g.mean(), g.std(), mag.mean(), mag.std(), np.percentile(mag, 99)],
        ]
    )


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Same square-root inverse-frequency weighting as the CNN's loss."""
    counts = np.bincount(y, minlength=6).astype(float)
    w = np.sqrt(counts.sum() / (6 * np.maximum(counts, 1)))
    return w[y] / w[y].mean()


def fit(frame, root) -> dict:
    """Pick C on validation macro-F1, then return validation and test probabilities."""
    X = np.stack([image_features(p, root) for p in frame["path"]])
    y = frame["y"].to_numpy()
    parts = {s: (frame["split"] == s).to_numpy() for s in ("train", "val", "test")}

    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        model.fit(X[parts["train"]], y[parts["train"]], logisticregression__sample_weight=sample_weights(y[parts["train"]]))
        score = macro_f1(y[parts["val"]], model.predict_proba(X[parts["val"]]))
        if best is None or score > best["val_macro_f1"]:
            best = {"C": c, "val_macro_f1": score, "model": model}

    model = best["model"]
    return {
        "C": best["C"],
        "features": int(X.shape[1]),
        "val": (model.predict_proba(X[parts["val"]]), y[parts["val"]]),
        "test": (model.predict_proba(X[parts["test"]]), y[parts["test"]]),
    }
