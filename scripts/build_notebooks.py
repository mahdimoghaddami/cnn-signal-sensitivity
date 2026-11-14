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

    Implement sampling, quantization, affine rotation, and histogram equalization.
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

    from cnn_signal_sensitivity import checks
    from cnn_signal_sensitivity.data import CachedInputs, case_studies, load_lab_data
    from cnn_signal_sensitivity.evaluation import predict, run_suite, seed_everything
    from cnn_signal_sensitivity.experiments import lab_conditions
    from cnn_signal_sensitivity.models import choose_device, load_classifier
    from cnn_signal_sensitivity.plotting import (
        image_grid,
        plot_curves,
        show_input_output,
        transformation_gallery,
    )
    from cnn_signal_sensitivity.transforms import identity_condition

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
    The four operations are:

    | Operation | What changes in the image |
    |---|---|
    | Sampling and interpolation | Reduce spatial samples, then reconstruct. |
    | Intensity quantization | Use fewer intensity levels. |
    | Affine rotation | Rotate and interpolate pixel values. |
    | Histogram equalization | Redistribute grayscale intensities to change contrast. |

    **Before running:** Which of the four operations should change predictions most? Why?

    **Your prediction:** TODO

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
    demonstrate("sample_image", ", factor=8", "Output: sampling factor 8")
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
    demonstrate("quantize", ", bits=2", "Output: 2 bits (4 intensity levels/channel)")
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
    demonstrate("affine_rotation", ", angle=30.0", "Output: 30-degree counterclockwise rotation")
    code("checks.check_affine(affine_rotation)")
    exercise(
        "histogram_equalize",
        "(images: torch.Tensor) -> torch.Tensor",
        """
    ### D. Histogram equalization
    Histogram equalization redistributes intensity levels to spread contrast across their range.
    Convert RGB to intensity using its channel mean. For each image, compute a 256-bin
    histogram and cumulative distribution. Map intensities using (CDF−CDF_min)/(1−CDF_min).
    Leave constant-intensity images unchanged. Replicate the result into three channels.
    """,
    )
    demonstrate("histogram_equalize", "", "Output: histogram-equalized grayscale")
    code("checks.check_histogram(histogram_equalize)")
    md("""
    ## 4. Compare your results
    The evaluation below calls the four functions you just wrote.
    The grayscale comparison isolates histogram equalization from the removal of color.
    Rotation can also crop objects; inspect the images before interpreting accuracy changes.
    """)
    code("""
    conditions = lab_conditions(sample_image, quantize, affine_rotation, histogram_equalize)
    selected = [c for c in conditions if c.key in
                {'clean', 'sampling_8', 'quantization_2', 'rotation_30', 'grayscale', 'equalized'}]
    fig = transformation_gallery(model, case_images[:2], case_labels[:2], case_ids[:2], selected)
    fig.savefig(OUTPUT / "examples.png", dpi=150, bbox_inches="tight")
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
    compared with grayscale? Support your answers with numbers and examples.

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
    for destination, solution in [
        (ROOT / "notebooks/01_cnn_signal_sensitivity.ipynb", False),
        (ROOT / "instructor/01_cnn_signal_sensitivity_solution.ipynb", True),
    ]:
        write_notebook(destination, notebook(solution))
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
