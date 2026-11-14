# From Pixels to Predictions

**CSI 5550 - Visual Computing**

Study how sampling, quantization, geometric transformations, and histogram processing
affect the predictions of an already trained convolutional neural network.

## Model and data

ResNet-18 was trained on ImageNet; we load its published IMAGENET1K_V1 weights from
TorchVision. No training or fine-tuning is needed. We select the ten ImageNet outputs
corresponding to Imagenette's categories. Confidence is conditional on those ten choices.
This is a classroom sensitivity experiment, not an independent generalization benchmark.

The lab uses one fixed, class-balanced set of 200 Imagenette images. All comparisons use
the same images. Inputs are RGB values in [0,1], resized/cropped to 224×224. Apply your
operators before the model's normalization. Do not modify the network or its preprocessing.

## Implement four operators

Use vectorized PyTorch arithmetic. Return tensors with the same shape, dtype, and device,
without changing the input in place. Each notebook exercise has a public check.

1. **Image sampling and nearest-neighbor interpolation:** take every factor-th row/column.
   Repeat the retained samples to reconstruct the original dimensions, cropping excess
   rows/columns. Use positive integer factors. Factor 1 is identity. This deliberately
   omits smoothing before subsampling so you can observe information loss and aliasing.
2. **Intensity quantization:** use round(x*(2**b-1))/(2**b-1), with integer b in [1,8].
   This preserves the endpoints and the original samples at 8 bits.
3. **Affine geometric transformation:** implement rotation about the image center using
   inverse mapping. Keep the original image dimensions, use bilinear interpolation, and
   fill outside regions with gray (0.5). Positive angles are counterclockwise.
4. **Histogram equalization:** use arithmetic-mean RGB intensity, a 256-bin histogram,
   and its cumulative distribution. Normalize by the first occupied CDF value, leaving
   a constant-intensity image unchanged. Return the intensity image in three channels.

These four operations cover different image properties: spatial resolution, intensity
resolution, spatial arrangement, and intensity distribution. Every evaluated transformation
calls one of these student implementations. Original RGB and original grayscale are controls.

## Affine mapping conventions

Coordinates x increase rightward and y downward. Subtract the image center
((W-1)/2, (H-1)/2). For each output location (x,y), find its input position using:

- source_x = cos(theta)*x - sin(theta)*y
- source_y = sin(theta)*x + cos(theta)*y

Add the center back. Convert pixel coordinates to the normalized coordinates expected
by grid_sample: gx = 2*(source_x+0.5)/W - 1, and similarly for gy using H.
Use align_corners=False. You may use grid_sample to interpolate, but construct the
mapping yourself. Subtracting 0.5 before zero-padded sampling and adding it back afterward
provides the required gray fill. Do not replace the exercise with a ready-made rotation call.

## Histogram convention

Round intensity*255 to assign pixels to bins 0–255. Compute a separate histogram for
each image. With a normalized cumulative distribution CDF, map occupied bins through
(CDF-CDF_min)/(1-CDF_min), where CDF_min is the first occupied cumulative value.
Use the mapping to reconstruct the image. If all pixels occupy one bin, preserve their
original intensity. Compare against grayscale, since equalization also removes color.

## Compare and explain

Write a prediction before running the experiments. Then run the supplied evaluation
using your functions. Inspect the example images, accuracy curves, and results table.
Accuracy is the fraction classified correctly. Accuracy drop is measured in percentage
points from the original images. A prediction flip may introduce an error or correct one.

Discuss:

- How do spatial and intensity resolution affect classification differently?
- How do rotation, interpolation, and cropping affect the evidence visible to the model?
- Does histogram equalization improve classification compared with the grayscale control?
- Which findings match your expectations? Support explanations with numbers and examples.

Do not assume every operation must reduce accuracy. Settings across different operators
are not matched severities, and the findings apply to this model and these images.

## Submission and grading

Restart the kernel and run all cells. Submit the completed notebook, any added source
files, outputs/lab/, and data/lab_subset.json. Do not submit image archives, model weights,
or the virtual environment. Results in outputs/lab/ are replaced when you rerun; copy
that folder if you want to retain an earlier experiment.

| Component | Weight |
|---|---:|
| Operator correctness and mathematical understanding | 40% |
| Clear quantitative and visual comparisons | 25% |
| Explanation of results and limitations | 25% |
| Reproducible, readable notebook | 10% |

The instructor will announce the deadline and collaboration policy separately.
