"""Model training entry point using OpenCD or a local demo runtime."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
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
    command: list[str] | None = None


def train(
    config_path: Path,
    work_dir: Path,
    demo_mode: bool = False,
    dataset_dir: Path | None = None,
    opencd_root: Path | None = None,
) -> dict[str, str | None]:
    """Train a change detection model with OpenCD or create demo artifacts."""
    if demo_mode:
        return asdict(train_demo(config_path, work_dir, dataset_dir))

    work_dir.mkdir(parents=True, exist_ok=True)
    train_script = resolve_opencd_tool("train.py", opencd_root)

    command = [
        sys.executable,
        str(train_script),
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
        "command": command,
    }


def resolve_opencd_tool(tool_name: str, opencd_root: Path | None = None) -> Path:
    """Resolve an OpenCD tool script from OPENCD_ROOT or an installed package."""
    candidates: list[Path] = []
    if opencd_root is not None:
        candidates.append(opencd_root / "tools" / tool_name)
    if os.environ.get("OPENCD_ROOT"):
        candidates.append(Path(os.environ["OPENCD_ROOT"]) / "tools" / tool_name)

    spec = importlib.util.find_spec("opencd")
    if spec and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        candidates.extend(
            [
                package_dir.parent / "tools" / tool_name,
                package_dir / "tools" / tool_name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "OpenCD training tool was not found. Install OpenCD from "
        "https://github.com/likyoo/open-cd and set OPENCD_ROOT to that checkout."
    )


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
