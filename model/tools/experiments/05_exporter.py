from __future__ import annotations

import json
from pathlib import Path

from ml_digits import RESOLUTION
from onnx_export_common import ModelFactory, load_experiment_module

experiment_name = "05_convolution_configs_adamw"


def model_factory_for(source: Path) -> ModelFactory | None:
    module = load_experiment_module("05.py")
    config = conv_config_for(source)

    return lambda: module.ConvModel(
        r=RESOLUTION,
        conv_channels=config["conv_channels"],
        kernel_size=config["kernel_size"],
        use_maxpool=config["use_maxpool"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
    )


def conv_config_for(source: Path) -> dict:
    config_name = source.stem.removeprefix("model_")
    results = json.loads((source.parent / "results.json").read_text())

    try:
        return results[config_name]["config"]
    except KeyError as error:
        raise KeyError(f"Could not find config for {config_name}") from error
