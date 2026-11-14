#let accent = rgb("#137c8b")
#let muted = rgb("#4a5568")
#let rulecol = rgb("#cbd5e0")
#let codebg = rgb("#f7fafc")

#set page(
  paper: "us-letter",
  margin: (x: 0.78in, y: 0.7in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: muted)
      CSI 5550 -- Visual Computing #h(1fr) From Pixels to Predictions
      #v(-0.4em)
      #line(length: 100%, stroke: 0.4pt + rulecol)
    ]
  },
  footer: context [
    #set text(size: 8.5pt, fill: muted)
    #align(center)[#counter(page).display()]
  ],
)

#set text(font: "New Computer Modern", size: 10.2pt, fill: black)
#set par(leading: 0.56em, spacing: 0.65em, justify: true)

#show heading.where(level: 1): it => {
  v(0.5em)
  block(breakable: false, [
    #set text(size: 13pt, weight: "bold", fill: accent)
    #it
    #v(-0.55em)
    #line(length: 100%, stroke: 0.6pt + rulecol)
  ])
  v(0.15em)
}

#show heading.where(level: 2): it => {
  v(0.35em)
  block(breakable: false, text(size: 11pt, weight: "bold", fill: black, it))
}

#show raw.where(block: false): box.with(
  fill: codebg,
  inset: (x: 3pt, y: 0pt),
  outset: (y: 2.5pt),
  radius: 2pt,
)

#show raw.where(block: true): block.with(
  fill: codebg,
  stroke: 0.5pt + rulecol,
  inset: 8pt,
  radius: 3pt,
  width: 100%,
)

#set heading(numbering: (..nums) => {
  let n = nums.pos()
  if n.len() == 1 { str(n.at(0)) + "." } else if n.len() == 2 { str(n.at(0)) + "." + str(n.at(1)) }
})

#align(center)[
  #text(size: 9pt, weight: "bold", fill: accent)[CSI 5550 -- VISUAL COMPUTING #h(0.8em) | #h(0.8em) LABORATORY ASSIGNMENT] \
  #v(-0.2em)
  #text(size: 20pt, weight: "bold")[From Pixels to Predictions] \
  #v(-0.3em)
  #text(size: 10.5pt, fill: muted)[Measuring CNN Sensitivity to Classical Image Processing Operators]
]
#line(length: 100%, stroke: 0.8pt + rulecol)

= Overview and Experimental Setup

In this laboratory, you will implement six classical image processing operators---spatial sampling, intensity quantization, affine rotation, histogram equalization (with a histogram counter helper), Gaussian smoothing, and Laplacian sharpening---from first principles in PyTorch. You will then evaluate *your own implementations* as front-end transformations before a pre-trained convolutional neural network to measure how signal resolution, geometry, contrast, blur, and high-frequency emphasis change predictions and confidence.

== Model, Data, and Pipeline
- *Classifier:* ResNet-18 loaded with TorchVision's published `IMAGENET1K_V1` weights. No training or fine-tuning is performed. Logits are restricted to the 10 Imagenette categories (_tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute_), and confidence is the conditional 10-class softmax probability. Because Imagenette is derived from ImageNet, this lab is a controlled sensitivity study rather than an independent generalization benchmark.
- *Dataset:* One fixed, class-balanced subset of 200 images (20 per class) from `imagenette2-320/val`, recorded in `data/lab_subset.json`. Every condition is evaluated on the exact same 200 images.
- *Preprocessing Order:* Each image is resized (shorter side 256, bilinear with antialiasing), center-cropped to $224 times 224$, and scaled to RGB values in $[0, 1]$. Your operators execute on these $[0, 1]$ RGB tensors; ImageNet mean/standard-deviation normalization happens afterward inside `RestrictedClassifier.forward`. Do not modify the network or its normalization.

== Quick Start
From the project root directory, synchronize the environment, cache the dataset and weights once, and open JupyterLab (select the project's `.venv` kernel):

```powershell
uv sync --locked
uv run --locked cnn-signal-sensitivity prepare
uv run --locked jupyter lab
```

Open `notebooks/01_cnn_signal_sensitivity.ipynb`. The seven exercise stubs raise `NotImplementedError` until completed. Each exercise is followed immediately by a multi-parameter visual demonstration and a public check in `cnn_signal_sensitivity.checks`.

= General Operator Contract

Unless noted otherwise, every function you implement must satisfy:

+ *Input/output tensor contract:* Accept a batch `images` of shape $(B, 3, H, W)$ with floating-point RGB values in $[0, 1]$. Return a tensor with the exact same shape, `dtype`, and `device` (except `image_histogram`, which returns `torch.long` counts of shape $(B, 256)$ on the input device).
+ *Immutability:* Never modify `images` in place; return a new tensor or clone when returning identity.
+ *Vectorized PyTorch:* Use PyTorch tensor operations rather than nested Python pixel loops (iterating over the batch dimension $B$ for per-image 1-D `torch.bincount` is fine).
+ *Parameter validation:* Raise `ValueError` for invalid arguments (e.g., non-positive or non-integer sampling factors, bit depths outside $[1, 8]$, non-finite angles, negative or non-finite filter strengths, or Gaussian kernel radius $>= min(H, W)$).
+ *Public checks:* Passing the supplied notebook checks is necessary to proceed, but not an exhaustive correctness proof---verify edge cases such as non-square inputs, multi-image batches, and constant images.

= Operator Specifications (Seven Exercises)

== A. Image Sampling and Nearest-Neighbor Interpolation (`sample_image`)
Subsample spatially by keeping every $s$-th row and column (`images[:, :, ::factor, ::factor]`) for positive integer `factor` $s >= 1$. Reconstruct the original $(H, W)$ spatial dimensions by repeating retained samples $s$ times along height and width, then cropping any excess rows or columns back to $(H, W)$. When $s = 1$, reproduce the input unchanged.

*Design note:* Anti-aliasing smoothing before subsampling is intentionally omitted so you can observe aliasing and high-frequency detail loss directly. *Evaluated settings:* $s in {1, 2, 4, 8, 16}$.

== B. Intensity Quantization (`quantize`)
Reduce per-channel intensity precision to $2^b$ uniformly spaced levels in $[0, 1]$ for integer bit depth $b in {1, 2, dots, 8}$:
$ Q_b (x) = (op("round")((2^b - 1) x)) / (2^b - 1). $
This endpoint-preserving formula keeps $0$ and $1$ exact and preserves standard 8-bit image samples when $b = 8$. *Evaluated settings:* $b in {8, 6, 4, 3, 2, 1}$ bits/channel.

== C. Affine Geometric Transformation (`affine_rotation`)
Rotate every image in the batch counterclockwise by `angle` $theta$ (in degrees) about its center using *inverse coordinate mapping* and bilinear interpolation, keeping the $(H, W)$ canvas fixed and filling outside regions with neutral gray ($0.5$). When $theta = 0$, return a clone of `images`. Do not replace this exercise with a ready-made rotation helper.

*Coordinate and sampling conventions:*
Let $x in {0, dots, W-1}$ increase rightward and $y in {0, dots, H-1}$ increase downward, with image center $(c_x, c_y) = ((W-1)/2, (H-1)/2)$. For each centered output coordinate $(hat(x), hat(y)) = (x - c_x, y - c_y)$, compute the source location in pixel coordinates:
$ x_s = hat(x) cos theta - hat(y) sin theta + c_x, quad y_s = hat(x) sin theta + hat(y) cos theta + c_y. $
Convert $(x_s, y_s)$ to normalized `F.grid_sample` coordinates in $[-1, 1]$ for `align_corners=False`:
$ g_x = (2(x_s + 0.5)) / W - 1, quad g_y = (2(y_s + 0.5)) / H - 1. $
To obtain the gray ($0.5$) fill outside the image boundary while using `padding_mode="zeros"`, sample `images - 0.5` with `mode="bilinear"` and `align_corners=False`, add `0.5` back, and clamp to $[0, 1]$. *Evaluated settings:* $theta in {-45 degree, -30 degree, -15 degree, 0 degree, 15 degree, 30 degree, 45 degree}$.

== D1. Image Histogram (`image_histogram`)
Compute the arithmetic-mean grayscale intensity $I = (R + G + B) / 3$ for each pixel, and assign bin indices $k = op("round")(255 I) in {0, 1, dots, 255}$. Return a `torch.long` tensor of shape $(B, 256)$ on the input device where row $i$ contains the pixel counts for image $i$ (each row sums to $H times W$). Looping over the batch dimension $B$ with 1-D `torch.bincount(..., minlength=256)` is fine. You will use this function to plot input, equalized, and twice-equalized histograms on shared axes.

== D2. Histogram Equalization (`histogram_equalize`)
Redistribute grayscale intensities independently for each image using its 256-bin histogram (you may call `image_histogram`) and cumulative distribution $op("CDF")(k) = sum_(j=0)^k h(j)$. Let $op("CDF")_min$ be the cumulative count of the first occupied bin (smallest $k$ with $h(k) > 0$). If all pixels occupy a single bin ($op("CDF")(255) - op("CDF")_min = 0$), preserve the constant grayscale intensity unchanged; otherwise map each pixel's bin $k$ through
$ T(k) = (op("CDF")(k) - op("CDF")_min) / (H W - op("CDF")_min) = (op("CDF")_"norm"(k) - op("CDF")_"norm,min") / (1 - op("CDF")_"norm,min"), $
where $op("CDF")_"norm" = op("CDF") / (H W)$ is the normalized CDF, clamped to $[0, 1]$. Replicate the equalized intensity across all three RGB channels so the output shape is $(B, 3, H, W)$.

*Experimental control:* Because equalization removes color as well as changing contrast, Section 4 evaluates both `grayscale` (arithmetic RGB mean replicated to 3 channels) and `equalized` to isolate contrast redistribution from color removal.

== E. Gaussian Smoothing (`gaussian_smooth`)
Attenuate high-frequency detail using an isotropic 2-D Gaussian kernel with standard deviation `sigma` $sigma >= 0$ (in pixels). When $sigma = 0$, return a clone of `images`. Otherwise set kernel radius $r = ceil(3 sigma)$ (raising `ValueError` if $r >= min(H, W)$), evaluate
$ K(x, y) = exp(- (x^2 + y^2) / (2 sigma^2)), quad x, y in {-r, dots, r}, $
and normalize so $sum_(x, y) K(x, y) = 1$. Pad borders via `F.pad(images, (r, r, r, r), mode="reflect")` and filter each RGB channel independently with grouped `F.conv2d(..., groups=channels)`, clamping to $[0, 1]$. *Evaluated settings:* $sigma in {0, 0.5, 1, 2, 4}$ pixels.

== F. Laplacian Sharpening (`laplacian_sharpen`)
Emphasize rapid intensity transitions using the discrete four-neighbor Laplacian kernel:
$ L = mat(0, 1, 0; 1, -4, 1; 0, 1, 0). $
For nonnegative sharpening strength `amount` $a >= 0$ ($a = 0$ returns a clone), pad `images` by $1$ pixel on all sides using `mode="reflect"`, filter each RGB channel independently with grouped `F.conv2d(..., groups=channels)` to obtain the *signed* response $nabla^2 f$, and compute
$ g = op("clamp")(f - a nabla^2 f, 0, 1). $
Keep the signed response before subtraction (do not take $|nabla^2 f|$) and clip only after subtracting from the original image. *Evaluated settings:* $a in {0, 0.25, 0.5, 1, 2}$.

= Evaluation, Analysis, and Discussion

Section 4 of the notebook evaluates all 31 conditions (the clean RGB baseline, 28 parameter settings across the five parameterized operators, plus the `grayscale` and `equalized` controls) via `run_suite` and saves CSV, NPZ, JSON, and figure exports to `outputs/lab/`.

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    align: left,
    table.hline(stroke: 0.9pt),
    [*Family*], [*Property Modified*], [*Evaluated Conditions*],
    table.hline(stroke: 0.5pt),
    [Baseline], [Unmodified input], [`clean`],
    [Sampling], [Spatial resolution], [Factor $s in {1, 2, 4, 8, 16}$],
    [Quantization], [Intensity resolution], [Bit depth $b in {8, 6, 4, 3, 2, 1}$],
    [Affine rotation], [Spatial orientation & fill], [Angle $theta in {-45 degree, -30 degree, -15 degree, 0 degree, 15 degree, 30 degree, 45 degree}$],
    [Histogram equalization], [Contrast & color removal], [`grayscale` control, `equalized` (256 bins)],
    [Gaussian smoothing], [Low-pass spatial filtering], [$sigma in {0, 0.5, 1, 2, 4}$ pixels],
    [Laplacian sharpening], [High-frequency emphasis], [Amount $a in {0, 0.25, 0.5, 1, 2}$],
    table.hline(stroke: 0.9pt),
  ),
  caption: [The 31 conditions evaluated in Section 4.],
)

== Reported Metrics
- `accuracy_pct`: Percentage of the 200 images classified correctly.
- `drop_pp`: Accuracy change in percentage points relative to the clean baseline ($"clean" - "condition"$).
- `flip_pct`: Percentage of images whose top-1 predicted class changed from the clean prediction (a flip may introduce an error or occasionally correct an initial error).
- `confidence_mean`: Mean predicted probability of the top-1 class across the 10 categories.
- `image_mse_mean`: Mean squared pixel difference between transformed and original images.

== Prediction and Written Discussion
+ *Before running Section 4:* After inspecting your visual demonstrations in Section 3, record your prediction in the notebook stating which operations and severities will alter ResNet-18 predictions most and why.
+ *After running Section 4:* Inspect the galleries (`examples.png`, `filter_examples.png`), accuracy curves (`accuracy.png`), and results table, then write a concise analysis addressing:
  - How spatial subsampling and intensity quantization affect predictions differently (compare accuracy drops, confidence, and pixel MSE).
  - How rotation, bilinear interpolation, border gray fill, and corner cropping affect visual evidence and accuracy.
  - Whether histogram equalization improves or degrades classification relative to the `grayscale` control.
  - How Gaussian smoothing and Laplacian sharpening affect fine texture, prediction confidence, and accuracy.
  - Which findings matched or surprised your expectations, supported by specific numbers and image examples, along with scope limitations (fixed 200-image sample, unmatched parameter severities across families, single pre-trained model).

= Submission and Grading

Before packaging your submission, select *Kernel $arrow.r$ Restart Kernel and Run All Cells* so every cell executes cleanly from top to bottom with saved outputs. Notice that rerunning replaces files in `outputs/lab/`; copy that directory elsewhere if you want to archive an earlier run.

- *Submit:* Completed `notebooks/01_cnn_signal_sensitivity.ipynb`, any added Python modules, `outputs/lab/`, and `data/lab_subset.json`.
- *Do not submit:* `.venv/`, downloaded Imagenette image archives, or cached model weights.

#figure(
  table(
    columns: (1fr, auto),
    stroke: none,
    inset: (x: 8pt, y: 5pt),
    align: (left, right),
    table.hline(stroke: 0.9pt),
    [*Component*], [*Weight*],
    table.hline(stroke: 0.5pt),
    [Operator correctness and mathematical understanding], [40%],
    [Clear quantitative and visual comparisons], [25%],
    [Explanation of results and limitations], [25%],
    [Reproducible, readable notebook], [10%],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Grading breakdown.],
)
