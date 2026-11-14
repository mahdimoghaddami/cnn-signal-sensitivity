"""Condition registry for the six student operators and grayscale control."""

from functools import partial

from .transforms import Condition, grayscale, identity_condition


def lab_conditions(
    sample_image,
    quantize,
    affine_rotation,
    histogram_equalize,
    gaussian_smooth=None,
    laplacian_sharpen=None,
):
    """Evaluate student operator implementations across all 31 lab conditions."""
    conditions = [identity_condition()]
    for factor in [1, 2, 4, 8, 16]:
        conditions.append(
            Condition(
                f"sampling_{factor}",
                "sampling",
                factor,
                "sampling factor",
                partial(sample_image, factor=factor),
            )
        )
    for bits in [8, 6, 4, 3, 2, 1]:
        conditions.append(
            Condition(
                f"quantization_{bits}",
                "quantization",
                bits,
                "bits/channel",
                partial(quantize, bits=bits),
            )
        )
    for angle in [-45, -30, -15, 0, 15, 30, 45]:
        conditions.append(
            Condition(
                f"rotation_{angle}",
                "affine_rotation",
                angle,
                "rotation angle (degrees)",
                partial(affine_rotation, angle=angle),
                settings={"interpolation": "bilinear", "fill": 0.5},
            )
        )
    conditions.extend(
        [
            Condition(
                "grayscale",
                "histogram_equalization",
                "Original intensity",
                "image",
                grayscale,
                plot_kind="bar",
            ),
            Condition(
                "equalized",
                "histogram_equalization",
                "Equalized intensity",
                "image",
                histogram_equalize,
                plot_kind="bar",
                settings={"bins": 256},
            ),
        ]
    )
    if gaussian_smooth is not None:
        for sigma in [0, 0.5, 1, 2, 4]:
            conditions.append(
                Condition(
                    f"gaussian_{sigma:g}",
                    "gaussian_smoothing",
                    sigma,
                    "sigma (pixels)",
                    partial(gaussian_smooth, sigma=sigma),
                    settings={
                        "radius": "ceil(3*sigma)",
                        "border": "reflect",
                        "channels": "independent RGB",
                    },
                )
            )
    if laplacian_sharpen is not None:
        for amount in [0, 0.25, 0.5, 1, 2]:
            conditions.append(
                Condition(
                    f"laplacian_{amount:g}",
                    "laplacian_sharpening",
                    amount,
                    "sharpening amount",
                    partial(laplacian_sharpen, amount=amount),
                    settings={
                        "kernel": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        "border": "reflect",
                        "clip": [0, 1],
                    },
                )
            )
    return conditions
