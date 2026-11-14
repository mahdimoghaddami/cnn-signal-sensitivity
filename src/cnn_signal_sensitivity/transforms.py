"""Supplied operations. All functions accept BCHW floats in [0, 1]."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def validate_images(images):
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Expected floating-point RGB images of shape (B, 3, H, W).")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Expected finite, unnormalized image values in [0, 1].")


def separable_filter(images, kernel):
    """Apply a supplied odd, 1-D kernel separately to each channel; reflect borders."""
    if kernel.ndim != 1 or len(kernel) % 2 != 1:
        raise ValueError("The kernel must be a 1-D odd-length tensor.")
    radius = len(kernel) // 2
    if radius >= min(images.shape[-2:]):
        raise ValueError("Reflect padding requires a kernel radius smaller than the image.")
    channels = images.shape[1]
    kernel = kernel.to(images)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    result = F.conv2d(
        F.pad(images, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels
    )
    return F.conv2d(
        F.pad(result, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels
    )


def resample(images, size, *, antialias=True, reconstruction="bilinear"):
    """Bilinear downsampling, optionally antialiased; explicitly chosen reconstruction."""
    if not 1 <= size <= min(images.shape[-2:]):
        raise ValueError("Requested size must be between 1 and the input size.")
    if tuple(images.shape[-2:]) == (size, size):
        return images.clone()
    small = F.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False, antialias=antialias
    )
    kwargs = {} if reconstruction == "nearest" else {"align_corners": False}
    return F.interpolate(small, size=images.shape[-2:], mode=reconstruction, **kwargs).clamp(0, 1)


def desaturate(images, factor):
    """Scale saturation toward arithmetic-mean intensity (HSI I), not luma or HSV V."""
    if not 0 <= factor <= 1:
        raise ValueError("Saturation factor must be in [0, 1].")
    intensity = images.mean(dim=1, keepdim=True)
    return (intensity + factor * (images - intensity)).clamp(0, 1)


def rotate(images, angle):
    """Positive angle is counterclockwise. Fixed canvas, bilinear sampling, gray fill."""
    return TF.rotate(
        images, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=[0.5, 0.5, 0.5]
    )


def additive_noise(images, std, generator=None, channel=None):
    if std < 0 or channel not in (None, 0, 1, 2):
        raise ValueError("std must be nonnegative; channel must be None, 0, 1, or 2.")
    noise = (
        torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
        * std
    )
    if channel is not None:
        mask = torch.zeros(1, 3, 1, 1, device=images.device, dtype=images.dtype)
        mask[:, channel] = 1
        noise = noise * mask
    return (images + noise).clamp(0, 1)


def intensity(images):
    """Arithmetic RGB mean used by detection and thresholding."""
    return images.mean(dim=1, keepdim=True)


def grayscale(images):
    """Color-removal control with the same intensity definition as the detectors."""
    return intensity(images).expand(-1, 3, -1, -1).contiguous()


def _correlate_gray(images, kernel):
    kernel = torch.as_tensor(kernel, device=images.device, dtype=images.dtype).view(1, 1, 3, 3)
    return F.conv2d(F.pad(intensity(images), (1, 1, 1, 1), mode="reflect"), kernel)


def point_response(images):
    """Absolute point-detector response / 8, using a zero-sum 3x3 mask."""
    return (
        (_correlate_gray(images, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).abs() / 8)
        .clamp(0, 1)
        .expand(-1, 3, -1, -1)
        .contiguous()
    )


def line_response(images, direction="horizontal"):
    kernels = {
        "horizontal": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        "vertical": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        "diagonal": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        "antidiagonal": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
    }
    if direction not in kernels:
        raise ValueError(f"Choose a line direction from {list(kernels)}.")
    response = (_correlate_gray(images, kernels[direction]).abs() / 6).clamp(0, 1)
    return response.expand(-1, 3, -1, -1).contiguous()


def sobel_edges(images):
    """Gradient magnitude scaled by a fixed bound, never per-image min-max scaled."""
    gx = _correlate_gray(images, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = _correlate_gray(images, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    magnitude = (torch.sqrt(gx.square() + gy.square()) / (4 * 2**0.5)).clamp(0, 1)
    return magnitude.expand(-1, 3, -1, -1).contiguous()


def threshold(images, value=0.5):
    if not 0 <= value <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    return (intensity(images) > value).to(images.dtype).expand(-1, 3, -1, -1).contiguous()


def otsu_threshold(images):
    """Per-image 256-bin Otsu threshold. Constant images use their occupied bin."""
    gray = intensity(images)
    masks = []
    for sample in gray:
        bins = torch.round(sample.flatten() * 255).long()
        histogram = torch.bincount(bins, minlength=256).to(images.dtype)
        probability = histogram / histogram.sum()
        omega = probability.cumsum(0)
        mu = (probability * torch.arange(256, device=images.device)).cumsum(0)
        denominator = omega * (1 - omega)
        score = (mu[-1] * omega - mu).square() / denominator.clamp_min(1e-12)
        score[denominator <= 1e-12] = -1
        cutoff = score.argmax() if (score >= 0).any() else bins[0]
        # Compare in the quantized space used to calculate the histogram.
        masks.append((bins.view_as(sample) > cutoff).to(images.dtype))
    return torch.stack(masks).expand(-1, 3, -1, -1).contiguous()


def overlay_edges(images, amount):
    if not 0 <= amount <= 1:
        raise ValueError("Blend amount must be in [0, 1].")
    return ((1 - amount) * images + amount * sobel_edges(images)).clamp(0, 1)


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
