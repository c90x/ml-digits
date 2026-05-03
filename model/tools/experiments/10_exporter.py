from __future__ import annotations

from pathlib import Path

from onnx_export_common import ModelFactory, load_experiment_module

experiment_name = "10_final_model"


def model_factory_for(_source: Path) -> ModelFactory | None:
    module = load_experiment_module("10.py")
    return module.FinalDigitModel
