from __future__ import annotations

from pathlib import Path

from ml_digits import RESOLUTION
from onnx_export_common import ModelFactory, load_experiment_module

experiment_name = "02_logistic_regression_lr"


def model_factory_for(_source: Path) -> ModelFactory | None:
    module = load_experiment_module("02.py")
    return lambda: module.LogisticRegressionModel(r=RESOLUTION)
