from __future__ import annotations

from pathlib import Path

from ml_digits import RESOLUTION
from onnx_export_common import ModelFactory, load_experiment_module

experiment_name = "03_layers_relu"


def model_factory_for(source: Path) -> ModelFactory | None:
    module = load_experiment_module("03.py")
    hidden_sizes = hidden_sizes_from_model_name(source.stem)
    return lambda: module.LayeredModel(r=RESOLUTION, hidden_sizes=hidden_sizes)


def hidden_sizes_from_model_name(model_name: str) -> list[int]:
    layer_name = model_name.removeprefix("model_epoch50-")

    if not layer_name.startswith("l"):
        raise ValueError(f"Could not read layer sizes from {model_name}")

    return [int(size) for size in layer_name[1:].split("_")]
