from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import onnx
import torch
import torch.nn as nn

model_dir = Path(__file__).resolve().parents[1]

if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from ml_digits import RESOLUTION, project_dir

ModelFactory = Callable[[], nn.Module]
ExperimentExporter = Callable[[Path], ModelFactory | None]


@dataclass(frozen=True)
class ExportSpec:
    source: Path
    target: Path
    model_factory: ModelFactory


experiment_module_by_script: dict[str, ModuleType] = {}


def collect_export_specs(
    source_dir: Path,
    output_dir: Path,
    exporters: Mapping[str, ExperimentExporter],
) -> list[ExportSpec]:
    specs: list[ExportSpec] = []

    for source in sorted(source_dir.rglob("*.pt")):
        experiment_name = experiment_name_for(source)
        exporter = exporters.get(experiment_name or "")

        if exporter is None:
            print(f"skip unsupported {display_path(source)}")
            continue

        model_factory = exporter(source)

        if model_factory is None:
            print(f"skip unsupported {display_path(source)}")
            continue

        relative_source = source.relative_to(source_dir)
        target = output_dir / relative_source.with_suffix(".onnx")
        specs.append(
            ExportSpec(source=source, target=target, model_factory=model_factory)
        )

    return specs


def experiment_name_for(source: Path) -> str | None:
    parts = source.parts

    try:
        experiments_index = parts.index("experiments")
    except ValueError:
        return None

    if experiments_index + 1 >= len(parts):
        return None

    return parts[experiments_index + 1]


def load_experiment_module(script_name: str) -> ModuleType:
    existing_module = experiment_module_by_script.get(script_name)

    if existing_module is not None:
        return existing_module

    script_path = project_dir / "model" / "experiments" / script_name
    module_name = f"ml_digits_experiment_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {display_path(script_path)}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    experiment_module_by_script[script_name] = module

    return module


def export_model(spec: ExportSpec, *, opset: int) -> None:
    model = spec.model_factory()
    state_dict = torch.load(spec.source, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    spec.target.parent.mkdir(parents=True, exist_ok=True)
    sample_input = torch.zeros(1, 1, RESOLUTION, RESOLUTION, dtype=torch.float32)

    torch.onnx.export(
        model,
        sample_input,
        str(spec.target),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(str(spec.target))


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return project_dir / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)
