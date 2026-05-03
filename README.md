# ml-digits

Handwritten digit classification for digits `0` through `9`.

The repository contains a PyTorch experiments, trained model export tooling, and a SvelteKit web demo.

## Documentation

- [Experiment details](docs/experiments.md): detailed notes for every numbered experiment and the model layer sequences.
- [Web demo readme](web/README.md): setup, development, build the SvelteKit app.

## Project Structure

```text
ml-digits/
├─ docs/
│  ├─ experiments.md       # Experiment descriptions and model diagrams
├─ model/
│  ├─ experiments/         # Training and dataset scripts
│  │  ├─ 00.py             # Dataset split details
│  │  ├─ 01.py             # Logistic regression baseline
│  │  ├─ 02.py             # Logistic regression learning-rate sweep
│  │  ├─ 03.py             # Fully connected ReLU layer sweep
│  │  ├─ 04.py             # Logistic regression optimizer sweep
│  │  ├─ 05.py             # CNN configuration sweep
│  │  ├─ 10.py             # Final CNN with dataset transformations
│  │  ├─ download.py       # Dataset download and directory setup
│  ├─ ml_digits/           # Shared dataset, path, environment, and utility code
│  ├─ tools/
│  │  ├─ onnx-exporter.py  # Export trained PyTorch weights to ONNX
│  ├─ nb.ipynb             # Exploration / testing notebook (not primary code!)
│  ├─ pyproject.toml       # Python project configuration
│  ├─ uv.lock              # Python dependency lockfile
├─ web/
│  ├─ ...
```

## Stack

- **PyTorch** for model training and evaluation.
- **Pillow** and NumPy for image preprocessing and dataset transformations.
- **TensorBoard** for training metrics.
- *ONNX* and *ONNX Runtime* for model export and validation.
- **SvelteKit**, **Tailwind CSS**, and *ONNX Runtime Web* for the browser demo.
- **KaggleHub** for downloading the HWD-V1 handwritten digit dataset.

## Generated Paths

| Path | Purpose |
| --- | --- |
| `.temp/dataset/HWD-V1` | Downloaded HWD-V1 dataset. |
| `.torch/runs` | TensorBoard logs. |
| `.torch/experiments` | Saved `.pt` weights and `results.json` files. |
| `web/static/model` | Exported ONNX models served by the web demo. |

## Model Setup

Requirements:

- Python `3.14+`
- [uv](https://github.com/astral-sh/uv) package manager

Install Python dependencies from the `model` project:

```bash
cd model
uv sync
```

## Dataset Setup

Download the dataset and create required runtime directories:

```bash
cd model
uv run ./experiments/download.py
```

The download uses KaggleHub dataset `metricasecuador/handwritten-digits-version-1-hwd-v1` and stores it under `.temp/dataset/HWD-V1` at the repository root.

## Running Experiments

Run experiments from the `model` directory:

```bash
cd model
uv run ./experiments/00.py
uv run ./experiments/01.py
uv run ./experiments/02.py
uv run ./experiments/03.py
uv run ./experiments/04.py
uv run ./experiments/05.py
uv run ./experiments/10.py
```

The numbered scripts write TensorBoard runs to `.torch/runs` and experiment artifacts to `.torch/experiments`. Experiment `10.py` trains the final CNN and can run much longer than the earlier comparison experiments.

For the full experiment breakdown, see [docs/experiments.md](docs/experiments.md).

## TensorBoard

Start TensorBoard from the model environment:

```bash
cd model
uv run tensorboard --logdir ../.torch/runs
```

## Exporting Models To ONNX

Export trained `.pt` weights from `.torch/experiments` to browser-ready ONNX files under `web/static/model`:

```bash
cd model
uv run ./tools/onnx-exporter.py --overwrite
```

Exporter options:

| Option | Meaning |
| --- | --- |
| `--dry-run` | Print supported exports without writing files. |
| `--overwrite` | Replace existing `.onnx` files. |
| `--source-dir <path>` | Read `.pt` weights from a custom directory. |
| `--output-dir <path>` | Write `.onnx` files to a custom directory. |
| `--opset <number>` | Use a custom ONNX opset. The default is `18`. |

The web app loads model URLs listed in `web/src/routes/models.json`, so exported files must exist at the paths referenced there.

## Web Demo

The web app lets users draw a digit on a canvas and compare predictions from exported ONNX models in the browser.

See [web/README.md](web/README.md) for install, development, build, and preview commands.

## ROCm

When using an AMD GPU, uncomment the ROCm dependency and index configuration already present in `model/pyproject.toml`. The relevant lines are:

```toml
# In the existing [project].dependencies array
"triton-rocm>=3.6.0",

[tool.uv.sources]
torch = [{ index = "pytorch-rocm" }]
torchvision = [{ index = "pytorch-rocm" }]
triton-rocm = [{ index = "pytorch-rocm" }]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true
```

Then run `uv sync` again inside `model`.

## Notebook Hygiene

The repository uses `nbstripout` for notebooks. Install the Git filter before committing notebook changes:

```bash
nbstripout --install
```
