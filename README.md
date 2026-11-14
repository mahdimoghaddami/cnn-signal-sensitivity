# From Pixels to Predictions

A simple PyTorch notebook lab: students implement image sampling, intensity quantization,
affine rotation, and histogram equalization, then evaluate exactly those implementations
with an already trained ResNet-18 classifier on 200 Imagenette images.

## Start

~~~powershell
uv sync --locked
uv run --locked cnn-signal-sensitivity prepare
uv run --locked jupyter lab
~~~

Open notebooks/01_cnn_signal_sensitivity.ipynb for the student starter or
instructor/01_cnn_signal_sensitivity_solution.ipynb for completed implementations.
Select the project's .venv interpreter when using another notebook editor.

There is one image set, one workflow, and one output folder: outputs/lab/.
No mode, run label, interaction challenge, or feature-map analysis is required.
The four exercise stubs intentionally stop execution until students complete them.
The instructor notebook runs end to end. Full instructions are in ASSIGNMENT.md.

## Model and environment

The model is ResNet-18, already trained on ImageNet, loaded from TorchVision's published
IMAGENET1K_V1 weights. Predictions are restricted to the ten Imagenette categories.
Students do not train or fine-tune it. Imagenette is derived from ImageNet, so this
exercise measures sensitivity rather than independent generalization.

The existing Python 3.15 and CUDA 13.2 PyTorch environment is preserved. The notebook
selects GPU or CPU automatically. Downloads are cached under data/. The fixed subset
is recorded in data/lab_subset.json. Original data, earlier run results, and notebook
backups remain in place. Rerunning overwrites the current outputs/lab/ exports.

## Files

- ASSIGNMENT.md: exercise specifications and grading.
- notebooks/: student notebook with four exercises and their result comparisons.
- instructor/: solution notebook, reference implementations, and teaching notes.
- src/cnn_signal_sensitivity/: data loading, model, evaluation, and plotting utilities.
- scripts/build_notebooks.py: generates notebooks and archives changed existing copies.
- scripts/verify_notebook.py: executes an instructor copy under outputs/lab/.
- scripts/package_student.py: builds a student ZIP without solutions or data.

## Verify and distribute

~~~powershell
uv run --locked python scripts/build_notebooks.py
uv run --locked pytest -q
uv run --locked python scripts/verify_notebook.py
uv run --locked python scripts/package_student.py
~~~

Distribute dist/cnn-signal-sensitivity-student.zip, not the full repository.
Changed notebooks are archived in outputs/notebook_backups/ before regeneration.
Older helper functions remain for archived experiments; the active lab uses only
lab_conditions and the four matching notebook exercises.

Sources: Gonzalez and Woods, Digital Image Processing, 4e;
[Imagenette](https://github.com/fastai/imagenette);
[TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
