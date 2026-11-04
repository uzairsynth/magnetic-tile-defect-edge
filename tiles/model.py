"""MobileNetV3-Small, ImageNet-pretrained, with a new 6-class head."""

import torch
from torch import nn
from torchvision import models

ARCHS = ("mobilenet_v3_small",)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHS}")


def cam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage: the layer Grad-CAM reads activations and gradients from."""
    return model.features[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
