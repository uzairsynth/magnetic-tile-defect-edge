"""Grad-CAM, plus a check of whether the heatmap actually points at the labelled defect."""

import numpy as np
import torch
import torch.nn.functional as F

from tiles.model import cam_layer


def grad_cam(model: torch.nn.Module, image: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
    """Return a [0, 1] heatmap the size of the input, and the class it explains.

    Weights each channel of the last conv stage by its average gradient for the
    target class, sums them, and keeps the positive part (Selvaraju et al., 2017).
    """
    layer = cam_layer(model)
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.eval()
        x = image.unsqueeze(0) if image.dim() == 3 else image
        x = x.clone().requires_grad_(True)
        logits = model(x)
        target = int(logits.argmax(1)) if target is None else int(target)
        model.zero_grad()
        logits[0, target].backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
    finally:
        h1.remove()
        h2.remove()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 0 else np.zeros_like(cam), target


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (F.max_pool2d(t, 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0).numpy()


def pointing_hit(cam: np.ndarray, mask: np.ndarray, tolerance: int = 8) -> bool:
    """Pointing game: does the hottest pixel fall on the defect (within `tolerance` pixels)?"""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(dilate(mask, tolerance)[y, x])


def energy_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of heatmap mass on the defect; a random map would score the mask's area fraction."""
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else 0.0
