# Instructor guide

The active notebook now has four sections: setup, original predictions, operator
implementation, and comparison/discussion. It uses one balanced set of 200 images.
There are no development/evaluation modes, run labels, interaction challenges, or
feature-map analysis. Student code and evaluated operators match exactly.
The instructor version has six operators plus the histogram helper. Student notebook
and distribution files remain frozen until the user requests finalization.

## Exercises

- Sampling: integer decimation plus nearest-neighbor reconstruction. This exposes aliasing
  and detail loss without an additional smoothing operation.
- Quantization: endpoint-preserving uniform rounding, with 8-bit original-image identity.
- Affine rotation: explicit inverse coordinate mapping; bilinear sampling is supplied by
  PyTorch grid_sample. The student builds the grid. Borders use fixed gray fill.
- Histogram equalization: per-image 256-bin intensity histogram and normalized CDF.
  A grayscale control separates color removal from equalization. Constant images stay constant.
  Students also implement image_histogram to plot the input and output histograms using
  identical axes. Equalization computes its own histogram independently. Counts are per image.

- Gaussian smoothing: construct and normalize a 2-D Gaussian kernel, using radius
  ceil(3*sigma), reflection padding, and independent RGB filtering. Zero sigma is identity.
- Laplacian sharpening: use the four-neighbor mask with center -4, then subtract the
  scaled signed response from the original image. Clip only the final result to [0,1].
  Reflection padding preserves constant images; amount zero is identity.

All seven function implementations appear directly in the solution notebook. The public checks
follow an immediate side-by-side input/output demonstration for each operator. Both
versions provide parameter/return type hints and a short conceptual reminder.
The checks
cover meaningful non-identity examples. The result suite contains 30 conditions, including
original RGB and grayscale controls. The classifier is ResNet-18 with TorchVision's
ImageNet-trained weights and ten selected Imagenette outputs. No retraining is required.

The simplified lab is an exploratory comparison on one fixed image set. It does not use
an untouched final set for assessing parameter tuning. Students should state the limits
of findings from this model and sample; flat curves and negative results are valid.

## Maintenance

Rebuild only the instructor notebook with scripts/build_notebooks.py and verify with
scripts/verify_notebook.py. Verification saves successful outputs in the instructor notebook
and under outputs/lab/. Do not rebuild the student notebook or ZIP until finalization.
Existing notebook edits/outputs are archived before rebuilding. Previous run folders
and old subset manifests are preserved. Legacy utilities remain usable for archived runs
but are not part of this assignment or its notebook workflow.

The student ZIP excludes instructor solutions, generation scripts, tests, data, checkpoints,
and outputs. Distribute it rather than the whole repository.
