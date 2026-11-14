# From Pixels to Predictions

**CSI 5550 - Visual Computing**

Study how sampling, quantization, geometric transformations, histogram processing,
Gaussian smoothing, and Laplacian sharpening affect the predictions of an already
trained convolutional neural network.

## Model and data

ResNet-18 was trained on ImageNet; we load its published IMAGENET1K_V1 weights from
TorchVision. No training or fine-tuning is needed. We select the ten ImageNet outputs
corresponding to Imagenette's categories. Confidence is conditional on those ten choices.
This is a classroom sensitivity experiment, not an independent generalization benchmark.

The lab uses one fixed, class-balanced set of 200 Imagenette images. All comparisons use
the same images. Inputs are RGB values in [0,1], resized/cropped to 224×224. Apply your
operators before the model's normalization. Do not modify the network or its preprocessing.

## Implement six operators (seven exercises)

Use vectorized PyTorch arithmetic (looping over the batch dimension `B` for per-image 1-D
`torch.bincount` is fine). Return tensors with the same shape, dtype, and device,
without changing the input in place. Each notebook exercise has a public check.

1. **Image sampling and nearest-neighbor interpolation (`sample_image`):** take every
   factor-th row/column. Repeat the retained samples to reconstruct the original dimensions,
   cropping excess rows/columns. Use positive integer factors (tested: 1, 2, 4, 8, 16).
   Factor 1 is identity. This deliberately omits smoothing before subsampling so you can
   observe information loss and aliasing.
2. **Intensity quantization (`quantize`):** use `round(x*(2**b-1))/(2**b-1)`, with integer
   `b` in `[1, 8]` (tested: 8, 6, 4, 3, 2, 1 bits/channel). This preserves the endpoints
   and the original samples at 8 bits.
3. **Affine geometric transformation (`affine_rotation`):** implement rotation about the
   image center using inverse mapping (tested: -45°, -30°, -15°, 0°, 15°, 30°, 45°).
   Keep the original image dimensions, use bilinear interpolation, and fill outside regions
   with gray (`0.5`). Positive angles are counterclockwise.
4. **Image histogram and histogram equalization (`image_histogram`, `histogram_equalize`):**
   first implement `image_histogram` to return integer counts shaped `(B, 256)`, one row per
   image, using arithmetic-mean RGB intensity. Use it to plot the input and output histograms.
   Then implement `histogram_equalize` using a 256-bin histogram and its cumulative
   distribution (you may call `image_histogram`). Normalize by the first occupied CDF value,
   leaving a constant-intensity image unchanged. Return the intensity image replicated across
   three channels.
5. **Gaussian smoothing (`gaussian_smooth`):** construct a normalized 2-D Gaussian kernel
   `exp(-(x^2 + y^2) / (2 * sigma^2))` with radius `ceil(3 * sigma)` (smaller than H and W).
   Filter each RGB channel independently using reflection padding (`F.pad(..., mode="reflect")`)
   and grouped convolution (`F.conv2d(..., groups=channels)`), then clip to `[0, 1]`.
   Zero `sigma` is identity (tested: 0, 0.5, 1, 2, 4).
6. **Laplacian sharpening (`laplacian_sharpen`):** filter each RGB channel independently with
   the four-neighbor Laplacian mask `[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` using reflection
   padding and grouped convolution. Keep the signed response, compute `images - amount * response`
   for nonnegative `amount` (tested: 0, 0.25, 0.5, 1, 2), and clip only the final result
   to `[0, 1]`. Amount zero is identity.

These six operations cover spatial resolution, intensity resolution, spatial arrangement,
intensity distribution, smoothing, and sharpening. Every evaluated transformation calls one
of your implementations. Original RGB and original grayscale are controls.

## Affine mapping conventions

Coordinates `x` increase rightward and `y` downward. Subtract the image center
`((W-1)/2, (H-1)/2)`. For each output location `(x, y)`, find its input position using:

- `source_x = cos(theta)*x - sin(theta)*y`
- `source_y = sin(theta)*x + cos(theta)*y`

Add the center back. Convert pixel coordinates to the normalized coordinates expected
by `grid_sample`: `gx = 2*(source_x+0.5)/W - 1`, and similarly for `gy` using `H`.
Use `align_corners=False`. You may use `grid_sample` to interpolate, but construct the
mapping yourself. Subtracting `0.5` before zero-padded sampling and adding it back afterward
provides the required gray fill. Do not replace the exercise with a ready-made rotation call.

## Histogram convention

Round `intensity*255` to assign pixels to bins `0–255`. Compute a separate histogram for
each image. With a normalized cumulative distribution `CDF`, map occupied bins through
`(CDF - CDF_min) / (1 - CDF_min)`, where `CDF_min` is the first occupied cumulative value.
Use the mapping to reconstruct the image. If all pixels occupy one bin, preserve their
original intensity. Compare against grayscale, since equalization also removes color.

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
