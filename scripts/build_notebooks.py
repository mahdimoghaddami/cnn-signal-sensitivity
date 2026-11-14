"""Build concise notebooks, archiving existing edits before replacement."""

from __future__ import annotations

import ast
import shutil
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / "instructor/reference.py").read_text(encoding="utf-8")
FUNCTIONS = {
    node.name: ast.get_source_segment(REFERENCE, node)
    for node in ast.parse(REFERENCE).body
    if isinstance(node, ast.FunctionDef)
}


def notebook(solution=False):
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))

    def code(source, *, tags=None):
        cells.append(nbf.v4.new_code_cell(dedent(source).strip(), metadata={"tags": tags or []}))

    def exercise(name, signature, text):
        md(text)
        source = (
            FUNCTIONS[name]
            if solution
            else (
                f"def {name}{signature}:\n"
                '    """'
                + ast.get_docstring(ast.parse(FUNCTIONS[name]).body[0]).replace("\n", "\n    ")
                + '"""\n'
                f'    raise NotImplementedError("Complete {name} before continuing.")'
            )
        )
        code(source, tags=["exercise", name])

    def demonstrate(name, arguments, title):
        code(
            f"demo_input = case_images[1:2].clone()  # one RGB image, with its batch dimension\n"
            f"demo_output = {name}(demo_input{arguments})\n"
            f'fig = show_input_output(demo_input, demo_output, "{title}")\n'
            f'fig.savefig(OUTPUT / "demo_{name}.png", dpi=150, bbox_inches="tight")\n'
            "plt.show()",
            tags=["demonstration", name],
        )

    md("""
    # From Pixels to Predictions
    **CSI 5550 - Visual Computing**

    Implement sampling, quantization, affine rotation, histogram equalization,
    Gaussian smoothing, and Laplacian sharpening.
    Then measure how **your implementations** affect a trained image classifier.
    """)
    if solution:
        md("**Instructor copy: includes completed implementations.**")
    md("""
    ## 1. Setup
    We use ResNet-18, already trained on ImageNet, with weights supplied by
    [TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
    No training is needed. We restrict its predictions to the ten Imagenette categories.
    Images are resized/cropped to 224×224 before your operators; model normalization happens afterward.
    """)
    code("""
    import math
    import os
    import sys
    from importlib import reload
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from IPython.display import display
    from torch.utils.data import DataLoader

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / "src/cnn_signal_sensitivity").exists())
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["TORCH_HOME"] = str(ROOT / "data/models")

    from cnn_signal_sensitivity import checks, experiments
    from cnn_signal_sensitivity.data import CachedInputs, case_studies, load_lab_data
    from cnn_signal_sensitivity.evaluation import predict, run_suite, seed_everything
    from cnn_signal_sensitivity.models import choose_device, load_classifier
    from cnn_signal_sensitivity.plotting import (
        image_grid,
        plot_curves,
        transformation_gallery,
    )
    from cnn_signal_sensitivity.transforms import identity_condition

    reload(checks)  # refresh checks if this kernel imported an older version
    seed_everything(5550)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    DEVICE = choose_device()
    OUTPUT = ROOT / "outputs/lab"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Using {DEVICE}")
    """)
    md("""
    ## 2. Original images and predictions
    We use the same 200 images for every comparison. Accuracy is the percentage classified
    correctly; confidence is the model's probability among the ten possible categories.
    """)
    code("""
    dataset = CachedInputs(load_lab_data(ROOT / "data"))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = load_classifier(DEVICE)
    clean = predict(model, loader, identity_condition())
    print(f"Original-image accuracy: {100 * clean.correct.mean():.1f}%")
    case_images, case_labels, case_ids = case_studies(dataset)
    fig = image_grid(case_images, case_labels)
    plt.show()
    """)
    md("""
    The six operations are:

    | Operation | What changes in the image |
    |---|---|
    | Sampling and interpolation | Reduce spatial samples, then reconstruct. |
    | Intensity quantization | Use fewer intensity levels. |
    | Affine rotation | Rotate and interpolate pixel values. |
    | Histogram equalization | Redistribute grayscale intensities to change contrast. |
    | Gaussian smoothing | Average neighboring pixels to reduce fine detail. |
    | Laplacian sharpening | Emphasize rapid intensity changes. |

    **Before running:** Which operations should change predictions most? Why?

    **Your prediction:** TODO
    """)
    md("""
    ## 3. Implement the operators
    Inputs are floating-point RGB tensors of shape (B,3,H,W), with values in [0,1].
    Preserve shape, dtype, and device; do not change the input in place.
    """)
    exercise(
        "sample_image",
        "(images: torch.Tensor, factor: int) -> torch.Tensor",
        """
    ### A. Image sampling and interpolation
    Sampling determines how finely an image's spatial detail is represented.
    Keep every factor-th row and column. Reconstruct the original size using
    nearest-neighbor interpolation: repeat samples, then crop to the original dimensions.
    Factor 1 should reproduce the input. Use slicing and tensor repetition.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    factors = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    axes[0].imshow(demo_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    for ax, factor in zip(axes[1:], factors):
        sampled = sample_image(demo_input, factor=factor)
        ax.imshow(sampled[0].permute(1, 2, 0).cpu())
        ax.set_title(f"Sampling factor {factor}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_sample_image.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "sample_image"],
    )
    code("checks.check_sampling(sample_image)")
    exercise(
        "quantize",
        "(images: torch.Tensor, bits: int) -> torch.Tensor",
        r"""
    ### B. Intensity quantization
    Quantization represents pixel intensities using a finite number of levels.
    Implement $Q_b(x)=\mathrm{round}[(2^b-1)x]/(2^b-1)$ for integer $1\leq b\leq8$.
    Eight bits preserves the original 8-bit samples. Lower bit depths retain fewer intensity levels.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    bit_depths = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    axes[0].imshow(demo_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    for ax, bits in zip(axes[1:], bit_depths):
        quantized = quantize(demo_input, bits=bits)
        ax.imshow(quantized[0].permute(1, 2, 0).cpu())
        ax.set_title(f"{bits} bit(s): {2**bits} levels/channel")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_quantize.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "quantize"],
    )
    code("checks.check_quantize(quantize)")
    exercise(
        "affine_rotation",
        "(images: torch.Tensor, angle: float) -> torch.Tensor",
        """
    ### C. Affine geometric transformation
    A geometric transformation changes pixel locations; interpolation assigns their intensities.
    Rotate about the image center using inverse coordinate mapping and bilinear interpolation.
    Positive angles are counterclockwise degrees. Keep the original canvas and fill outside
    regions with 0.5. Construct the coordinate grid yourself; grid_sample is allowed.
    See ASSIGNMENT.md for coordinate conventions.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    angles = [-45, -20, 20, 45]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    axes[0].imshow(demo_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    for ax, angle in zip(axes[1:], angles):
        rotated = affine_rotation(demo_input, angle=angle)
        ax.imshow(rotated[0].permute(1, 2, 0).cpu())
        ax.set_title(f"Rotation {angle:+d} degrees")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_affine_rotation.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "affine_rotation"],
    )
    code("checks.check_affine(affine_rotation)")
    exercise(
        "image_histogram",
        "(images: torch.Tensor) -> torch.Tensor",
        """
    ### D1. Image histogram
    A histogram counts how many pixels have each intensity level.
    First convert RGB to grayscale using the channel mean, then round intensity*255 into 256 bins.
    Return a separate row of counts for each image; do not combine the batch.
    Use this function to plot the histograms before and after equalization.
    """,
    )
    code(
        """
    histogram_input = case_images[1:2].clone()
    grayscale_input = histogram_input.mean(dim=1, keepdim=True)
    counts = image_histogram(histogram_input)[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(histogram_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[1].imshow(grayscale_input[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Grayscale intensity (RGB mean)")
    axes[1].axis("off")
    axes[2].bar(range(256), counts.cpu().numpy(), width=1)
    axes[2].set(title="Input intensity histogram", xlabel="Intensity level", ylabel="Pixel count", xlim=(0, 255))
    fig.savefig(OUTPUT / "demo_image_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "image_histogram"],
    )
    code("checks.check_image_histogram(image_histogram)")
    exercise(
        "histogram_equalize",
        "(images: torch.Tensor) -> torch.Tensor",
        """
    ### D2. Histogram equalization
    Histogram equalization redistributes intensity levels to spread contrast across their range.
    Convert RGB to intensity using its channel mean. For each image, compute a 256-bin
    histogram and cumulative distribution. Map intensities using (CDF−CDF_min)/(1−CDF_min).
    Compute the histogram inside this function, independently of image_histogram.
    Leave constant-intensity images unchanged. Replicate the result into three channels.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    demo_output = histogram_equalize(demo_input)
    twice_equalized = histogram_equalize(demo_output)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, image, title in zip(axes, [demo_input, demo_output, twice_equalized],
                               ["Input image", "Histogram-equalized grayscale", "Twice histogram-equalized"]):
        ax.imshow(image[0].permute(1, 2, 0).cpu())
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_histogram_equalize.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "histogram_equalize"],
    )
    code(
        """
    input_counts = image_histogram(demo_input)[0]
    output_counts = image_histogram(demo_output)[0]
    twice_counts = image_histogram(twice_equalized)[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True, layout="constrained")
    for ax, counts, title in zip(axes, [input_counts, output_counts, twice_counts],
                                ["Input intensity histogram", "Equalized output histogram", "Twice equalized histogram"]):
        ax.bar(range(256), counts.cpu().numpy(), width=1)
        ax.set(title=title, xlabel="Intensity level", xlim=(0, 255))
    axes[0].set_ylabel("Pixel count")
    fig.savefig(OUTPUT / "histogram_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["histogram-comparison"],
    )
    code("checks.check_histogram(histogram_equalize)")
    exercise(
        "gaussian_smooth",
        "(images: torch.Tensor, sigma: float) -> torch.Tensor",
        r"""
    ### E. Gaussian smoothing
    A smoothing filter replaces each pixel with a weighted neighborhood average.
    Construct $\exp[-(x^2+y^2)/(2\sigma^2)]$ within radius $\lceil3\sigma\rceil$ and
    normalize its sum to one. Filter each RGB channel independently with reflection padding.
    Return the input unchanged for zero sigma. Convolution with conv2d is allowed.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    sigmas = [0.5, 1, 2, 4]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    axes[0].imshow(demo_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    for ax, sigma in zip(axes[1:], sigmas):
        smoothed = gaussian_smooth(demo_input, sigma=sigma)
        ax.imshow(smoothed[0].permute(1, 2, 0).cpu())
        ax.set_title(f"Gaussian sigma {sigma:g}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_gaussian_smooth.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "gaussian_smooth"],
    )
    code("checks.check_gaussian_smooth(gaussian_smooth)")
    exercise(
        "laplacian_sharpen",
        "(images: torch.Tensor, amount: float) -> torch.Tensor",
        r"""
    ### F. Laplacian sharpening
    The Laplacian measures second-order intensity changes. Use the mask
    [[0,1,0],[1,-4,1],[0,1,0]] independently on RGB channels with reflection padding.
    Sharpen with $g=f-a\nabla^2f$, for $a\geq0$, then clip to [0,1].
    Keep the signed response until the final subtraction; do not take its absolute value.
    """,
    )
    code(
        """
    demo_input = case_images[1:2].clone()
    amounts = [0.25, 0.5, 1, 2]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    axes[0].imshow(demo_input[0].permute(1, 2, 0).cpu())
    axes[0].set_title("Input image")
    for ax, amount in zip(axes[1:], amounts):
        sharpened = laplacian_sharpen(demo_input, amount=amount)
        ax.imshow(sharpened[0].permute(1, 2, 0).cpu())
        ax.set_title(f"Sharpening amount {amount:g}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "demo_laplacian_sharpen.png", dpi=150, bbox_inches="tight")
    plt.show()
    """,
        tags=["demonstration", "laplacian_sharpen"],
    )
    code("checks.check_laplacian_sharpen(laplacian_sharpen)")
    md("""
    ## 4. Compare your results
    The evaluation below calls the six operator functions you just wrote.
    The grayscale comparison isolates histogram equalization from the removal of color.
    Rotation can also crop objects; inspect the images before interpreting accuracy changes.
    """)
    code("""
    reload(experiments)  # pick up updated rotation settings in an existing kernel
    conditions = experiments.lab_conditions(
        sample_image, quantize, affine_rotation, histogram_equalize,
        gaussian_smooth, laplacian_sharpen,
    )
    selected = [c for c in conditions if c.key in
                {'clean', 'sampling_8', 'quantization_2', 'rotation_-45', 'rotation_45', 'grayscale', 'equalized'}]
    fig = transformation_gallery(model, case_images[:2], case_labels[:2], case_ids[:2], selected)
    fig.savefig(OUTPUT / "examples.png", dpi=150, bbox_inches="tight")
    plt.show()
    filter_examples = [c for c in conditions if c.key in {'clean', 'gaussian_2', 'laplacian_1'}]
    fig = transformation_gallery(model, case_images[:2], case_labels[:2], case_ids[:2], filter_examples)
    fig.savefig(OUTPUT / "filter_examples.png", dpi=150, bbox_inches="tight")
    plt.show()
    """)
    code("""
    table, results = run_suite(model, loader, conditions, OUTPUT, baseline=clean,
                               metadata={"subset_manifest": "data/lab_subset.json"})
    display(table[['condition', 'accuracy_pct', 'drop_pp', 'flip_pct']].round(2))
    fig = plot_curves(table)
    fig.savefig(OUTPUT / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    """)
    md("""
    Accuracy drop is measured in percentage points from the original images.
    Prediction flips can introduce errors or correct them.

    **Discuss:** Did your results match your prediction? Explain the effects of sampling,
    intensity resolution, interpolation, and contrast. Did histogram equalization help
    compared with grayscale? How do smoothing and sharpening affect detail and predictions?
    Support your answers with numbers and examples.

    **Your discussion:** TODO
    """)
    for index, cell in enumerate(cells):
        cell.id = f"lab-{index:03d}"
    document = nbf.v4.new_notebook(cells=cells)
    document.metadata = {
        "kernelspec": {
            "display_name": "Python (cnn-signal-sensitivity)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.15"},
        "title": "From Pixels to Predictions" + (" - Instructor" if solution else ""),
    }
    nbf.validate(document)
    return document


def write_notebook(destination, document, *, backup_dir=None):
    destination = Path(destination)
    content = nbf.writes(document)
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        backup_dir = Path(backup_dir or ROOT / "outputs/notebook_backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_dir / f"{destination.stem}_{stamp}.ipynb"
        shutil.copy2(destination, backup)
        print(f"Preserved previous notebook: {backup}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def main():
    # Finalize the instructor notebook before regenerating student deliverables.
    destination = ROOT / "instructor/01_cnn_signal_sensitivity_solution.ipynb"
    write_notebook(destination, notebook(True))
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
