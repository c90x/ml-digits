from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

model_dir = Path(__file__).resolve().parents[1]

if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from ml_digits import experiments_path, project_dir  # noqa: E402

from onnx_export_common import (  # noqa: E402
    ExperimentExporter,
    collect_export_specs,
    display_path,
    export_model,
    project_path,
)


def main() -> int:
    args = parse_args()
    source_dir = project_path(args.source_dir)
    output_dir = project_path(args.output_dir)
    exporters = load_experiment_exporters()

    specs = collect_export_specs(source_dir, output_dir, exporters)

    if not specs:
        print(f"No supported .pt models found under {display_path(source_dir)}")
        return 0

    if args.dry_run:
        for spec in specs:
            print(
                f"would export {display_path(spec.source)} -> {display_path(spec.target)}"
            )
        return 0

    exported = 0
    skipped = 0

    for spec in specs:
        if spec.target.exists() and not args.overwrite:
            skipped += 1
            print(f"skip existing {display_path(spec.target)}")
            continue

        export_model(spec, opset=args.opset)
        exported += 1
        print(f"exported {display_path(spec.target)}")

    print(f"exported={exported} skipped={skipped}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pretrained PyTorch digit models to ONNX."
    )
    parser.add_argument("--source-dir", type=Path, default=experiments_path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "web" / "static" / "model",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiment_exporters() -> dict[str, ExperimentExporter]:
    exporter_by_experiment_name: dict[str, ExperimentExporter] = {}
    exporter_dir = Path(__file__).resolve().parent / "experiments"

    for script_path in sorted(exporter_dir.glob("*_exporter.py")):
        module_name = f"onnx_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {display_path(script_path)}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        exporter_by_experiment_name[module.experiment_name] = module.model_factory_for

    return exporter_by_experiment_name


if __name__ == "__main__":
    raise SystemExit(main())
