from .env import project_dir

temp_dir = project_dir / ".temp"
dataset_path = temp_dir / "dataset" / "HWD-V1"

torch_path = project_dir / ".torch"
checkpoints_path = torch_path / "checkpoints"
runs_path = torch_path / "runs"
experiments_path = torch_path / "experiments"


def create_temp_dir() -> None:
    global temp_dir
    temp_dir.mkdir(exist_ok=True)


def create_torch_dir() -> None:
    global torch_path, checkpoints_path, runs_path, experiments_path
    torch_path.mkdir(exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True)
    runs_path.mkdir(exist_ok=True)
    experiments_path.mkdir(exist_ok=True)


def create_dirs() -> None:
    create_temp_dir()
    create_torch_dir()
