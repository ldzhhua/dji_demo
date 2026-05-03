"""Model training entry point using OpenCD or a local demo runtime."""

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# OpenCD is expected to be installed as a dependency.
# This file provides a simple command-line interface to start training.


@dataclass(frozen=True)
class TrainingSummary:
    """Summary returned after a training job finishes."""

    mode: str
    config_path: str
    work_dir: str
    model_path: str
    metrics_path: str | None = None


def train(
    config_path: Path,
    work_dir: Path,
    demo_mode: bool = False,
    dataset_dir: Path | None = None,
) -> dict[str, str | None]:
    """Train a change detection model with OpenCD or create demo artifacts."""
    if demo_mode:
        return asdict(train_demo(config_path, work_dir, dataset_dir))

    work_dir.mkdir(parents=True, exist_ok=True)

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
    return {
        "mode": "opencd",
        "config_path": str(config_path),
        "work_dir": str(work_dir),
        "model_path": str(work_dir),
        "metrics_path": None,
    }


def train_demo(config_path: Path, work_dir: Path, dataset_dir: Path | None = None) -> TrainingSummary:
    """Create deterministic demo training artifacts when OpenCD is unavailable."""
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "demo_model.pth"
    metrics_path = work_dir / "metrics.json"
    config_snapshot = work_dir / "config_snapshot.txt"

    model_path.write_text(
        "demo change detection model artifact\n"
        f"config={config_path}\n"
        f"dataset={dataset_dir or 'not provided'}\n",
        encoding="utf-8",
    )
    metrics = {
        "mode": "demo",
        "epochs": 3,
        "accuracy": 0.982,
        "f1_score": 0.947,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config_snapshot.write_text(f"Demo config source: {config_path}\n", encoding="utf-8")
    return TrainingSummary(
        mode="demo",
        config_path=str(config_path),
        work_dir=str(work_dir),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a change detection model")
    parser.add_argument("config", type=Path, help="Path to OpenCD config file")
    parser.add_argument("work_dir", type=Path, help="Directory to save outputs")
    parser.add_argument("--demo", action="store_true", help="Create demo training artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        print(train(args.config, args.work_dir, demo_mode=True))
    else:
        train(args.config, args.work_dir)
