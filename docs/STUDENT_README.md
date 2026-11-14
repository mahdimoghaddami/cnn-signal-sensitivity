# From Pixels to Predictions

Read the lab instructions (`ASSIGNMENT.md` or the included PDF) and complete
`notebooks/visual_computing_lab.ipynb`.

~~~powershell
uv sync --locked
uv run --locked cnn-signal-sensitivity prepare
uv run --locked jupyter lab
~~~

Use the project's `.venv` notebook kernel. The project uses Python 3.15 and the course's
PyTorch environment; it automatically selects GPU or CPU for inference.
Preparation downloads Imagenette and the already trained ResNet-18 weights once.

Implement the six operators and the histogram helper (seven exercises), then run their
comparisons. The lab always uses the same 200 images and saves to `outputs/lab/`; reruns
replace those exports. Copy the folder if you want to preserve an earlier result.

Submit only the executed notebook (`notebooks/visual_computing_lab.ipynb`).
Do not submit `.venv`, `outputs/`, `data/`, images, or model weights. Instructor solutions are not included.
