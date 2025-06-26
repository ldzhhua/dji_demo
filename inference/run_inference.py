"""Run model inference for change detection."""

import argparse
import subprocess
from pathlib import Path


def infer(model_path: Path, input_dir: Path, output_dir: Path) -> None:
    """Apply a trained model to new imagery using OpenCD."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import opencd  # noqa: F401 - only check availability
    except ImportError as e:
        raise RuntimeError(
            "OpenCD is required to run inference. Please install it first"
        ) from e

    command = [
        "python",
        "-m",
        "opencd.tools.infer",
        str(model_path),
        str(input_dir),
        "--out",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("model", type=Path, help="Path to trained model")
    parser.add_argument("input_dir", type=Path, help="Directory with imagery")
    parser.add_argument("output_dir", type=Path, help="Directory for predictions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.model, args.input_dir, args.output_dir)
