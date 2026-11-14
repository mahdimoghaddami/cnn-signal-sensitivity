"""One registry for image transformations, detector responses, and thresholding."""

from functools import partial

from .transforms import (
    Condition,
    additive_noise,
    desaturate,
    grayscale,
    identity_condition,
    line_response,
    otsu_threshold,
    overlay_edges,
    point_response,
    resample,
    rotate,
    sobel_edges,
    threshold,
)

DEFAULT_GROUPS = (
    "sampling",
    "quantization",
    "gamma",
    "blur",
    "sharpen",
    "saturation",
    "detection",
    "thresholding",
)


def lab_conditions(
    sample_image,
    quantize,
    affine_rotation,
    histogram_equalize,
    gaussian_smooth=None,
    laplacian_sharpen=None,
):
    """Evaluate notebook implementations; optional filters preserve older notebook calls."""
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
    for bits in [8, 6, 4, 3, 2]:
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


def core_conditions(quantize, gamma_transform, gaussian_blur, unsharp_mask):
    """The original six transformation families, retained for older notebooks."""
    conditions = [identity_condition()]
    for size in [224, 112, 56, 28, 14]:
        conditions.append(
            Condition(
                f"sampling_{size}",
                "sampling",
                size,
                "side length (pixels)",
                partial(resample, size=size),
                settings={
                    "downsample": "bilinear",
                    "antialias": True,
                    "reconstruction": "bilinear",
                },
            )
        )
    for bits in [8, 6, 4, 3, 2]:
        conditions.append(
            Condition(
                f"quantization_{bits}",
                "quantization",
                bits,
                "bits/channel",
                partial(quantize, bits=bits),
            )
        )
    for family, values in [
        ("gamma_brighten", [1, 0.85, 0.7, 0.55, 0.4]),
        ("gamma_darken", [1, 1.25, 1.5, 2, 3]),
    ]:
        for value in values:
            conditions.append(
                Condition(
                    f"{family}_{value:g}",
                    family,
                    value,
                    "gamma",
                    partial(gamma_transform, gamma=value),
                )
            )
    for sigma in [0, 0.5, 1, 2, 3]:
        conditions.append(
            Condition(
                f"blur_{sigma:g}",
                "blur",
                sigma,
                "sigma (pixels)",
                partial(gaussian_blur, sigma=sigma),
                settings={"radius": "ceil(3*sigma)", "border": "reflect"},
            )
        )
    for amount in [0, 0.5, 1, 2, 3]:
        conditions.append(
            Condition(
                f"sharpen_{amount:g}",
                "sharpen",
                amount,
                "mask weight",
                partial(unsharp_mask, sigma=1, amount=amount),
                settings={"blur_sigma": 1, "border": "reflect", "clip": [0, 1]},
            )
        )
    for factor in [1, 0.75, 0.5, 0.25, 0]:
        conditions.append(
            Condition(
                f"saturation_{factor:g}",
                "saturation",
                factor,
                "saturation factor",
                partial(desaturate, factor=factor),
                settings={"intensity": "arithmetic RGB mean"},
            )
        )
    return conditions


def representation_conditions():
    """Detectors, edge blending, and thresholds share the ordinary evaluation pipeline."""
    return [
        Condition(
            "grayscale", "detectors", "Grayscale", "representation", grayscale, plot_kind="bar"
        ),
        Condition(
            "point",
            "detectors",
            "Point",
            "representation",
            point_response,
            plot_kind="bar",
            settings={"response": "absolute mask / 8"},
        ),
        *[
            Condition(
                f"line_{direction}",
                "detectors",
                direction.title(),
                "representation",
                partial(line_response, direction=direction),
                plot_kind="bar",
                settings={"response": "absolute mask / 6"},
            )
            for direction in ["horizontal", "vertical", "diagonal", "antidiagonal"]
        ],
        Condition(
            "sobel",
            "detectors",
            "Sobel",
            "representation",
            sobel_edges,
            plot_kind="bar",
            settings={"response": "magnitude / (4*sqrt(2))"},
        ),
        *[
            Condition(
                f"edge_blend_{a:g}",
                "edge_blend",
                a,
                "edge blend fraction",
                partial(overlay_edges, amount=a),
            )
            for a in [0, 0.25, 0.5, 0.75, 1]
        ],
        *[
            Condition(
                f"threshold_{t:g}",
                "thresholding",
                f"T={t:g}",
                "threshold method",
                partial(threshold, value=t),
                plot_kind="bar",
                settings={"threshold": t, "intensity": "arithmetic RGB mean"},
            )
            for t in [0.25, 0.4, 0.5, 0.6, 0.75]
        ],
        Condition(
            "otsu",
            "thresholding",
            "Otsu",
            "threshold method",
            otsu_threshold,
            plot_kind="bar",
            settings={"bins": 256, "threshold": "per image"},
        ),
    ]


def extension_conditions(name):
    """Additional groups; the old chapter10 alias keeps existing notebooks runnable."""
    if name in {"detection", "thresholding", "chapter10"}:
        conditions = representation_conditions()
        if name == "detection":
            return [c for c in conditions if c.family != "thresholding"]
        if name == "thresholding":
            return [c for c in conditions if c.family == "thresholding"]
        return conditions
    if name == "geometry":
        return [
            Condition(
                f"rotation_{angle}",
                "rotation",
                angle,
                "angle (degrees)",
                partial(rotate, angle=angle),
                settings={"interpolation": "bilinear", "fill": 0.5, "expand": False},
            )
            for angle in [-30, -15, 0, 15, 30]
        ]
    if name == "noise":
        return [
            Condition(
                f"noise_{channel}_{std:g}",
                f"noise_{channel}",
                std,
                "noise sigma in [0,1]",
                partial(additive_noise, std=std, channel=None if channel == "all" else 0),
                stochastic=True,
                settings={"channel": channel, "energy_control": "equal per-channel sigma"},
            )
            for channel in ["all", "red"]
            for std in [0, 0.02, 0.05, 0.1, 0.2]
        ]
    raise ValueError("Choose detection, thresholding, geometry, or noise.")


def experiment_conditions(quantize, gamma_transform, gaussian_blur, unsharp_mask, *, groups=None):
    """Select named groups in one suite; clean is included once regardless of selection."""
    groups = list(DEFAULT_GROUPS if groups is None else groups)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("Select at least one group, without duplicates.")
    core = core_conditions(quantize, gamma_transform, gaussian_blur, unsharp_mask)
    available = {
        name: [c for c in core if c.family == name]
        for name in DEFAULT_GROUPS[:6]
        if name != "gamma"
    }
    available["gamma"] = [c for c in core if c.family.startswith("gamma_")]
    for name in ["detection", "thresholding", "geometry", "noise"]:
        available[name] = extension_conditions(name)
    unknown = set(groups) - available.keys()
    if unknown:
        raise ValueError(f"Unknown experiment groups: {sorted(unknown)}")
    return [
        identity_condition(),
        *[condition for group in groups for condition in available[group]],
    ]


def sampling_comparison():
    return [
        Condition(
            f"sampling_aa_{aa}_{size}",
            f"sampling_aa_{aa}",
            size,
            "side length (pixels)",
            partial(resample, size=size, antialias=aa),
            settings={"downsample": "bilinear", "antialias": aa, "reconstruction": "bilinear"},
        )
        for aa in [False, True]
        for size in [112, 56, 28, 14]
    ]


def recovery_conditions(gaussian_blur, unsharp_mask):
    return [
        identity_condition(),
        Condition(
            "degraded",
            "recovery",
            "Blurred",
            "condition",
            partial(gaussian_blur, sigma=2),
            plot_kind="bar",
            settings={"blur_sigma": 2},
        ),
        Condition(
            "processed_clean",
            "recovery",
            "Sharpened clean",
            "condition",
            partial(unsharp_mask, sigma=1, amount=1.5),
            plot_kind="bar",
            settings={"mask_sigma": 1, "amount": 1.5},
        ),
        Condition(
            "processed_degraded",
            "recovery",
            "Sharpened blurred",
            "condition",
            lambda x: unsharp_mask(gaussian_blur(x, sigma=2), sigma=1, amount=1.5),
            plot_kind="bar",
            settings={"blur_sigma": 2, "mask_sigma": 1, "amount": 1.5},
        ),
    ]
