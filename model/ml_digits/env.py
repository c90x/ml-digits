from pathlib import Path

RNG = 99

RATIO_TEST = 0.8
RATIO_VAL = 0.8

RESOLUTION = 28

project_dir = Path(__file__).parent.parent.parent
assert project_dir.exists()

# Sanity check
assert (project_dir / ".gitattributes").exists()
