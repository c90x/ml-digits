from __future__ import annotations

from pathlib import Path

from ml_digits import RESOLUTION
from onnx_export_common import ModelFactory, load_experiment_module

experiment_name = "04_logistic_regression_optimizers"


def model_factory_for(_source: Path) -> ModelFactory | None:
    module = load_experiment_module("04.py")
    return lambda: module.LogisticRegressionModel(r=RESOLUTION)
