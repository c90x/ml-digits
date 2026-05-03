from pathlib import Path

import kagglehub

from .paths import temp_dir, dataset_path, create_temp_dir


def download_dataset() -> None:
    create_temp_dir()

    path = (
        Path(
            kagglehub.dataset_download(
                "metricasecuador/handwritten-digits-version-1-hwd-v1",
                output_dir=str(temp_dir / "dataset"),
            )
        )
        / "HWD-V1"
    )

    if not path.samefile(dataset_path):
        raise RuntimeError("Paths do not match.")
