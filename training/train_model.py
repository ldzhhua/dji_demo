"""Model training entry point using OpenCD."""

import argparse
import subprocess
from pathlib import Path

# OpenCD is expected to be installed as a dependency.
# This file provides a simple command-line interface to start training.

def train(config_path: Path, work_dir: Path) -> None:
    """Train a change detection model with the given config using OpenCD."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_path} not found")

    try:
        import opencd  # noqa: F401 - only check availability
    except ImportError as e:
        raise RuntimeError(
            "OpenCD is required to run training. Please install it first"
        ) from e

    command = [
        "python",
        "-m",
        "opencd.tools.train",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a change detection model"
    )
    parser.add_argument("config", type=Path, help="Path to OpenCD config file")
    parser.add_argument("work_dir", type=Path, help="Directory to save outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.work_dir)
