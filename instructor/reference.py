"""Reference implementations used to build and verify the instructor notebook."""

import math

import torch
import torch.nn.functional as F

from cnn_signal_sensitivity.transforms import separable_filter


def sample_image(images: torch.Tensor, factor: int) -> torch.Tensor:
    """Reduce spatial sampling, then reconstruct using nearest neighbors.

    Args:
        images: Float RGB batch in [0, 1], shaped (B, 3, H, W).
            B is the number of images; H and W are height and width.
        factor: Positive integer spacing between retained rows and columns.

    Returns:
        Reconstructed batch with the same shape, dtype, and device as images.
        The input is not modified.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer.")
    height, width = images.shape[-2:]
    sampled = images[:, :, ::factor, ::factor]
    return sampled.repeat_interleave(factor, -2).repeat_interleave(factor, -1)[
        :, :, :height, :width
    ]


def affine_rotation(images: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate around the center using inverse mapping and bilinear interpolation.

    Args:
        images: Float RGB batch in [0, 1], shaped (B, 3, H, W).
            B is the number of images; H and W are height and width.
        angle: Counterclockwise rotation in degrees, applied to every image.

    Returns:
        Rotated batch with the same shape, dtype, and device as images.
        Outside regions use gray (0.5). The input is not modified.
    """
    if not math.isfinite(angle):
        raise ValueError("angle must be finite.")
    if angle == 0:
        return images.clone()
    batch, _, height, width = images.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=images.dtype),
        torch.arange(width, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    x, y = x - (width - 1) / 2, y - (height - 1) / 2
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    source_x = cosine * x - sine * y + (width - 1) / 2
    source_y = sine * x + cosine * y + (height - 1) / 2
    grid = torch.stack(
        (2 * (source_x + 0.5) / width - 1, 2 * (source_y + 0.5) / height - 1), dim=-1
    )
    grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
    return (
        F.grid_sample(
            images - 0.5, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        + 0.5
    ).clamp(0, 1)


def image_histogram(images: torch.Tensor) -> torch.Tensor:
    """Count pixels at each of 256 intensity levels, separately for every image.

    Args:
        images: Float RGB batch in [0, 1], shaped (B, 3, H, W).
            B is the number of images; H and W are height and width.

    Returns:
        Integer pixel counts shaped (B, 256), on the input device.
        Use the RGB channel mean as intensity and round intensity*255
        to select bins 0 through 255. Each histogram sums to H*W.
        The input is not modified.
    """
    bins = torch.round(images.mean(dim=1) * 255).long()
    return torch.stack([torch.bincount(image.flatten(), minlength=256) for image in bins])


def histogram_equalize(images: torch.Tensor) -> torch.Tensor:
    """Redistribute grayscale intensities using each image's cumulative histogram.

    Args:
        images: Float RGB batch in [0, 1], shaped (B, 3, H, W).
            B is the number of images; H and W are height and width.

    Returns:
        Equalized grayscale batch in [0, 1], with intensity replicated across
        three channels. Shape, dtype, and device match images. Each image is
        equalized independently, and the input is not modified.
    """
    gray = images.mean(dim=1, keepdim=True)
    output = []
    for image in gray:
        bins = torch.round(image * 255).long()
        histogram = torch.bincount(bins.flatten(), minlength=256).to(images.dtype)
        cdf = histogram.cumsum(0)
        minimum = cdf[histogram > 0][0]
        denominator = cdf[-1] - minimum
        if denominator == 0:
            output.append(image.clone())
        else:
            mapping = ((cdf - minimum) / denominator).clamp(0, 1)
            output.append(mapping[bins])
    return torch.stack(output).expand(-1, 3, -1, -1).contiguous()


def quantize(images: torch.Tensor, bits: int) -> torch.Tensor:
    """Reduce intensity resolution to 2**bits uniformly spaced levels in [0, 1].

    Args:
        images: Float RGB batch in [0, 1], shaped (B, 3, H, W).
            B is the number of images; H and W are height and width.
        bits: Integer bit depth from 1 to 8, applied to each color channel.

    Returns:
        Quantized batch with the same shape, dtype, and device as images.
        The input is not modified.
    """
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("bits must be an integer in [1, 8].")
    levels = 2**bits - 1
    return torch.round(images * levels) / levels


def gamma_transform(images, gamma):
    """Power law applied independently to encoded RGB channels; gamma > 0."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return images.pow(gamma)


def gaussian_kernel1d(sigma):
    """Normalized symmetric Gaussian with radius ceil(3*sigma); sigma=0 is identity."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if sigma == 0:
        return torch.ones(1, dtype=torch.float32)
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-offsets.square() / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(images, sigma):
    return separable_filter(images, gaussian_kernel1d(sigma)).clamp(0, 1)


def unsharp_mask(images, sigma, amount):
    """Original plus weighted highpass residual, followed by fixed [0,1] clipping."""
    if amount < 0:
        raise ValueError("amount must be nonnegative.")
    blurred = gaussian_blur(images, sigma)
    return (images + amount * (images - blurred)).clamp(0, 1)
