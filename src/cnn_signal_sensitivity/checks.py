"""Small public exercise checks; passing these is not a complete correctness proof."""

import torch

from .transforms import validate_images


def check_sampling(function):
    small = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3) / 26
    original_small = small.clone()
    torch.testing.assert_close(function(small, factor=1), small)
    small_indices = torch.tensor([0, 0, 2])
    expected_small = small[:, :, small_indices][:, :, :, small_indices]
    torch.testing.assert_close(function(small, factor=2), expected_small)
    torch.testing.assert_close(small, original_small)

    rect = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5) / 59
    original_rect = rect.clone()
    row_indices = torch.tensor([0, 0, 2, 2])
    col_indices = torch.tensor([0, 0, 2, 2, 4])
    expected_rect = rect[:, :, row_indices][:, :, :, col_indices]
    torch.testing.assert_close(function(rect, factor=2), expected_rect)
    torch.testing.assert_close(rect, original_rect)
    print("Sampling checks passed.")


def check_quantize(function):
    values = torch.tensor([0, 0.1, 0.4, 0.8, 1.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5).clone()
    original = values.clone()
    expected_1bit = (
        torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32).view(1, 1, 1, 5).expand_as(values)
    )
    torch.testing.assert_close(function(values, bits=1), expected_1bit)
    expected_2bit = torch.tensor([0, 0, 1 / 3, 2 / 3, 1]).view(1, 1, 1, 5).expand_as(values)
    output = function(values, bits=2)
    torch.testing.assert_close(output, expected_2bit)
    validate_images(output)
    levels = torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16).expand(1, 3, 16, 16) / 255
    torch.testing.assert_close(function(levels, bits=8), levels)
    torch.testing.assert_close(values, original)
    print("Quantization checks passed.")


def check_affine(function):
    nonsquare = torch.arange(105, dtype=torch.float32).reshape(1, 3, 5, 7) / 104
    original_nonsquare = nonsquare.clone()
    torch.testing.assert_close(function(nonsquare, angle=0), nonsquare)
    torch.testing.assert_close(
        function(nonsquare, angle=180),
        torch.flip(nonsquare, (-2, -1)),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(nonsquare, original_nonsquare)

    square = torch.arange(75, dtype=torch.float32).reshape(1, 3, 5, 5) / 74
    torch.testing.assert_close(
        function(square, angle=90), torch.rot90(square, 1, (-2, -1)), atol=1e-6, rtol=1e-5
    )

    ones = torch.ones((1, 3, 9, 9), dtype=torch.float32)
    rotated = function(ones, angle=45)
    validate_images(rotated)
    corners = rotated[:, :, [0, 0, -1, -1], [0, -1, 0, -1]]
    torch.testing.assert_close(corners, torch.full_like(corners, 0.5))
    print("Affine rotation checks passed.")


def check_image_histogram(function):
    values = torch.tensor([[0, 0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]])
    images = values.reshape(2, 1, 2, 2).expand(2, 3, 2, 2).clone()
    original = images.clone()
    expected = torch.zeros(2, 256, dtype=torch.long)
    expected[0, 0], expected[0, 128], expected[0, 255] = 2, 1, 1
    expected[1, 255] = 4
    torch.testing.assert_close(function(images), expected)
    torch.testing.assert_close(images, original)
    print("Image histogram checks passed.")


def check_histogram(function):
    nonconstant = torch.tensor([0, 0, 0, 0.5, 0.5, 1.0]).reshape(1, 1, 2, 3).expand(1, 3, 2, 3)
    constant = torch.full((1, 3, 2, 3), 0.4)
    batch = torch.cat([nonconstant, constant], dim=0)
    original = batch.clone()
    expected_nonconstant = (
        torch.tensor([0, 0, 0, 2 / 3, 2 / 3, 1.0]).reshape(1, 1, 2, 3).expand_as(nonconstant)
    )
    expected = torch.cat([expected_nonconstant, constant], dim=0)
    output = function(batch)
    validate_images(output)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(batch, original)
    print("Histogram equalization checks passed.")


def check_gaussian_smooth(function):
    image = torch.full((1, 3, 17, 17), 0.4)
    torch.testing.assert_close(function(image, sigma=0), image)
    torch.testing.assert_close(function(image, sigma=1), image)
    impulse = torch.zeros_like(image)
    impulse[0, 0, 8, 8] = 1
    original = impulse.clone()
    smoothed = function(impulse, sigma=1)
    validate_images(smoothed)
    assert 0 < smoothed[0, 0, 8, 8] < 1
    torch.testing.assert_close(smoothed.sum(), torch.tensor(1.0))
    assert smoothed[:, 1:].count_nonzero() == 0
    torch.testing.assert_close(impulse, original)
    print("Gaussian smoothing checks passed.")


def check_laplacian_sharpen(function):
    image = torch.full((1, 3, 9, 9), 0.4)
    torch.testing.assert_close(function(image, amount=1), image)
    image[:, :, 4, 4] = 0.5
    original = image.clone()
    torch.testing.assert_close(function(image, amount=0), image)
    expected = image.clone()
    expected[:, :, 4, 4] = 0.6
    for y, x in [(3, 4), (5, 4), (4, 3), (4, 5)]:
        expected[:, :, y, x] = 0.375
    torch.testing.assert_close(function(image, amount=0.25), expected)
    validate_images(function(image, amount=10))
    torch.testing.assert_close(image, original)
    print("Laplacian sharpening checks passed.")
