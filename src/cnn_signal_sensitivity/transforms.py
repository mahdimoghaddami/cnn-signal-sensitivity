"""Supplied operations. All functions accept BCHW floats in [0, 1]."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


def validate_images(images):
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Expected floating-point RGB images of shape (B, 3, H, W).")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Expected finite, unnormalized image values in [0, 1].")


def intensity(images):
    """Arithmetic RGB mean used for grayscale comparison."""
    return images.mean(dim=1, keepdim=True)


def grayscale(images):
    """Color-removal control matching the intensity definition of histogram equalization."""
    return intensity(images).expand(-1, 3, -1, -1).contiguous()


@dataclass(frozen=True)
class Condition:
    key: str
    family: str
    parameter: float | str
    parameter_name: str
    operation: Callable
    stochastic: bool = False
    plot_kind: str = "curve"
    settings: dict = field(default_factory=dict)

    def apply(self, images, image_ids, *, seed=5550):
        if self.stochastic:
            # Per-image noise makes results independent of batching and loader order.
            output = []
            for image, image_id in zip(images, image_ids, strict=True):
                generator = torch.Generator(device=images.device).manual_seed(seed + int(image_id))
                output.append(self.operation(image.unsqueeze(0), generator=generator))
            return torch.cat(output)
        return self.operation(images)


def identity_condition():
    return Condition("clean", "clean", 0, "identity", lambda images: images.clone())
