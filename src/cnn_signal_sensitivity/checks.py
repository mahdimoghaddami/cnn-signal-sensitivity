"""Small public exercise checks; passing these is not a complete correctness proof."""

import torch

from .transforms import validate_images


def check_quantize(function):
    values = torch.tensor([0, 0.1, 0.4, 0.8, 1.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    expected = torch.tensor([0, 0, 1 / 3, 2 / 3, 1]).view(1, 1, 1, 5).expand_as(values)
    output = function(values, bits=2)
    torch.testing.assert_close(output, expected)
    validate_images(output)
    levels = torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16).expand(1, 3, 16, 16) / 255
    torch.testing.assert_close(function(levels, bits=8), levels)
    print("Quantization checks passed.")


def check_gamma(function):
    values = torch.tensor([0, 0.25, 0.5, 1]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    torch.testing.assert_close(function(values, gamma=1), values)
    torch.testing.assert_close(function(values, gamma=2), values.square())
    torch.testing.assert_close(function(values, gamma=0.5), values.sqrt())
    print("Gamma checks passed.")


def check_kernel(function):
    torch.testing.assert_close(function(0), torch.ones(1))
    kernel = function(1)
    assert kernel.ndim == 1 and len(kernel) == 7, "Use radius ceil(3*sigma)."
    assert torch.all(kernel >= 0) and torch.isfinite(kernel).all()
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
    torch.testing.assert_close(kernel, kernel.flip(0))
    torch.testing.assert_close(kernel[2] / kernel[3], torch.exp(torch.tensor(-0.5)))
    print("Gaussian kernel checks passed.")


def check_unsharp(function):
    image = torch.full((1, 3, 17, 17), 0.5)
    torch.testing.assert_close(function(image, sigma=1, amount=2), image)
    impulse = image.clone()
    impulse[:, :, 8, 8] = 0.7
    torch.testing.assert_close(function(impulse, sigma=1, amount=0), impulse)
    output = function(impulse, sigma=1, amount=1)
    validate_images(output)
    assert output[0, 0, 8, 8] > impulse[0, 0, 8, 8], "The peak should sharpen."
    assert output[0, 0, 8, 7] < impulse[0, 0, 8, 7], "Inspect the negative mask halo."
    print("Unsharp masking checks passed.")


def check_sampling(function):
    image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47
    original = image.clone()
    torch.testing.assert_close(function(image, factor=1), image)
    expected = image[:, :, ::2, ::2].repeat_interleave(2, -2).repeat_interleave(2, -1)
    torch.testing.assert_close(function(image, factor=2), expected)
    torch.testing.assert_close(image, original)
    print("Sampling checks passed.")


def check_affine(function):
    image = torch.arange(75, dtype=torch.float32).reshape(1, 3, 5, 5) / 74
    torch.testing.assert_close(function(image, angle=0), image)
    torch.testing.assert_close(
        function(image, angle=90), torch.rot90(image, 1, (-2, -1)), atol=1e-6, rtol=1e-5
    )
    validate_images(function(image, angle=30))
    print("Affine rotation checks passed.")


def check_histogram(function):
    values = torch.tensor([0, 0, 0, 0.5, 0.5, 1.0]).reshape(1, 1, 2, 3).expand(1, 3, 2, 3)
    expected = torch.tensor([0, 0, 0, 2 / 3, 2 / 3, 1.0]).reshape(1, 1, 2, 3).expand_as(values)
    torch.testing.assert_close(function(values), expected)
    constant = torch.full((1, 3, 4, 4), 0.4)
    torch.testing.assert_close(function(constant), constant)
    print("Histogram equalization checks passed.")


def check_image_histogram(function):
    values = torch.tensor([[0, 0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]])
    images = values.reshape(2, 1, 2, 2).expand(2, 3, 2, 2)
    expected = torch.zeros(2, 256, dtype=torch.long)
    expected[0, 0], expected[0, 128], expected[0, 255] = 2, 1, 1
    expected[1, 255] = 4
    torch.testing.assert_close(function(images), expected)
    print("Image histogram checks passed.")
