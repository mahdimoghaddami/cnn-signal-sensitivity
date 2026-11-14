# From Pixels to Predictions

**CSI 5550 - Visual Computing**

Study how sampling, quantization, geometric transformations, histogram processing,
Gaussian smoothing, and Laplacian sharpening affect the predictions of an already
trained convolutional neural network.

## Quick start

If `uv` is not installed yet, follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/):

~~~bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

~~~powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
~~~

From the workspace root, install dependencies, prepare the model/dataset, and open
`notebooks/visual_computing_lab.ipynb`:

~~~powershell
uv sync --locked
uv run --locked cnn-signal-sensitivity prepare
uv run --locked jupyter lab
~~~

## Model and data

ResNet-18 was trained on ImageNet; we load its published IMAGENET1K_V1 weights from
TorchVision. No training or fine-tuning is needed. We select the ten ImageNet outputs
corresponding to Imagenette's categories. Confidence is conditional on those ten choices.
This is a classroom sensitivity experiment, not an independent generalization benchmark.

The lab uses one fixed, class-balanced set of 200 Imagenette images. All comparisons use
the same images. Each image is provided to your functions as a 224×224 RGB tensor with
values in [0, 1].

## Implement six operators (seven exercises)

Use vectorized PyTorch arithmetic (computing across whole tensors at once rather than nested
Python pixel loops; iterating over the batch dimension `B` for per-image operations is fine).
Return tensors with the same shape, dtype, and device, without overwriting the input in place.
The six operators are independent: you can implement and test them in any order, skip a
function if you get stuck, and come back to it later (note that `histogram_equalize` may
optionally call `image_histogram`, and the full evaluation runs once all operators are
completed).

1. **Image sampling and nearest-neighbor interpolation (`sample_image`):** spatially
   subsample (keep only every factor-th row and column). Reconstruct the original
   dimensions via nearest-neighbor interpolation (fill each output pixel with the value of
   the closest retained sample by repeating retained rows/columns and cropping excess).
   Use positive integer factors (tested: 1, 2, 4, 8, 16). Factor 1 is identity. Anti-aliasing
   (pre-smoothing before subsampling to prevent false jagged patterns known as aliasing) is
   deliberately omitted so you can observe information loss and aliasing directly.
2. **Intensity quantization (`quantize`):** reduce precision to discrete levels using
   `round(x*(2**b-1))/(2**b-1)`, with integer bit depth `b` in `[1, 8]` (where `b` bits per
   channel represent `2**b` intensity levels; tested: 8, 6, 4, 3, 2, 1). This preserves the
   endpoints and the original samples at 8 bits.
3. **Affine geometric transformation (`affine_rotation`):** implement rotation about the
   image center using inverse coordinate mapping (for each output location, compute where
   it came from in the input image) and bilinear interpolation (a distance-weighted average
   of the four nearest input pixels; tested: -45°, -30°, -15°, 0°, 15°, 30°, 45°). Keep
   the original image dimensions, fill any out-of-bounds pixels with neutral gray (`0.5`),
   and clamp (restrict) outputs to `[0, 1]`. Positive angles are counterclockwise.
4. **Image histogram and histogram equalization (`image_histogram`, `histogram_equalize`):**
   first implement `image_histogram` to return integer counts shaped `(B, 256)`, one row per
   image, using arithmetic-mean RGB intensity. Use it to plot the input and output histograms.
   Then implement `histogram_equalize` using a 256-bin histogram and its cumulative
   distribution (CDF: the running total of pixels up to each bin; you may call
   `image_histogram`). Normalize by the first occupied CDF value, leaving a
   constant-intensity image unchanged. Return the intensity image replicated across three
   channels.
5. **Gaussian smoothing (`gaussian_smooth`):** construct a normalized 2-D Gaussian kernel
   `e^(-(x^2 + y^2) / (2 * sigma^2))` with radius `ceil(3 * sigma)` (smaller than H and W).
   Filter each RGB channel independently using reflection border padding (mirroring interior
   pixels across the image edges), then clamp (restrict) outputs to `[0, 1]`. Zero `sigma`
   is identity (tested: 0, 0.5, 1, 2, 4).
6. **Laplacian sharpening (`laplacian_sharpen`):** filter each RGB channel independently with
   the four-neighbor Laplacian mask `[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` using reflection
   border padding (mirroring interior pixels across the edges). Keep the signed response
   (retaining both positive and negative values), compute `images - amount * response` for
   nonnegative `amount` (tested: 0, 0.25, 0.5, 1, 2), and clip (clamp) only the final result
   to `[0, 1]`. Amount zero is identity.

These six operations cover spatial resolution, intensity resolution, spatial arrangement,
intensity distribution, smoothing, and sharpening. Every evaluated transformation calls one
of your implementations. Original RGB and original grayscale are controls.

## Affine mapping conventions

Coordinates `x` increase rightward across width `W` and `y` downward across height `H`.
Subtract the image center `((W-1)/2, (H-1)/2)`. For each output location `(x, y)`, find
its input position using:

- `source_x = cos(theta)*x - sin(theta)*y`
- `source_y = sin(theta)*x + cos(theta)*y`

Add the center back, use bilinear interpolation over valid in-bounds image samples, and fill
any location that maps outside `[0, W-1] × [0, H-1]` with neutral gray (`0.5`). Do not
replace the exercise with a ready-made rotation call.

## Histogram convention

Round `intensity*255` to assign pixels to bins `0–255`. Compute a separate histogram for
each image. With a normalized cumulative distribution `CDF`, map occupied bins through
`(CDF - CDF_min) / (1 - CDF_min)`, where `CDF_min` is the first occupied cumulative value.
Use the mapping to reconstruct the image. If all pixels occupy one bin, preserve their
original intensity. Compare against grayscale, since equalization also removes color.

## Optional helpful PyTorch functions

You are free to implement every operator in whatever clean PyTorch style you prefer.
If you want ideas, check out the PyTorch functions below to read what they do—using other
valid approaches is great too:

- **Sampling, grids, and resampling:**
  [`torch.arange`](https://docs.pytorch.org/docs/stable/generated/torch.arange.html),
  [`torch.meshgrid`](https://docs.pytorch.org/docs/stable/generated/torch.meshgrid.html),
  [`torch.repeat_interleave`](https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html),
  [`torch.Tensor.repeat`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html),
  [`torch.Tensor.expand`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html),
  [`torch.nn.functional.grid_sample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
- **Rounding, bounds, histograms, and CDFs:**
  [`torch.round`](https://docs.pytorch.org/docs/stable/generated/torch.round.html),
  [`torch.clamp`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html),
  [`torch.floor`](https://docs.pytorch.org/docs/stable/generated/torch.floor.html),
  [`torch.ceil`](https://docs.pytorch.org/docs/stable/generated/torch.ceil.html),
  [`torch.bincount`](https://docs.pytorch.org/docs/stable/generated/torch.bincount.html),
  [`torch.histc`](https://docs.pytorch.org/docs/stable/generated/torch.histc.html),
  [`torch.cumsum`](https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html)
- **Padding and 2-D spatial filtering:**
  [`torch.nn.functional.pad`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html),
  [`torch.nn.functional.conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html),
  [`torch.nn.functional.unfold`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html)

## Compare and explain

Write a prediction after inspecting your operator demos and before running the full evaluation.
Then run the supplied evaluation using your functions. Inspect the example images, accuracy
curves, and results table. Accuracy is the percentage classified correctly. Accuracy drop is
measured in percentage points from the original images. A prediction flip may introduce an
error or correct one.

Discuss:

- How do spatial and intensity resolution affect classification differently?
- How do rotation, interpolation, and cropping affect the evidence visible to the model?
- Does histogram equalization improve classification compared with the grayscale control?
- How do Gaussian smoothing and Laplacian sharpening affect visual detail, confidence, and predictions?
- Which findings match your expectations? Support explanations with numbers and examples.

Do not assume every operation must reduce accuracy. Settings across different operators
are not matched severities, and the findings apply to this model and these images.

## Submission and grading

Restart the kernel and run all cells so all figures, tables, and written responses are saved
inside the notebook. Submit only the completed and executed notebook file
(`notebooks/visual_computing_lab.ipynb`). Do not submit `.venv/`, `outputs/`, `data/`,
image archives, model weights, or any other project files.

| Component | Weight |
|---|---:|
| Operator correctness and mathematical understanding | 40% |
| Clear quantitative and visual comparisons | 25% |
| Explanation of results and limitations | 25% |
| Reproducible, readable notebook | 10% |

The instructor will announce the deadline and collaboration policy separately.
