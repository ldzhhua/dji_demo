"""Run model inference for change detection."""

import argparse
import subprocess
from pathlib import Path


def infer(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    demo_mode: bool = False,
) -> dict[str, str | int | list[str]]:
    """Apply a trained model to new imagery using OpenCD or the demo runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if demo_mode:
        return run_demo_inference(model_path, input_dir, output_dir)

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
    return {
        "mode": "opencd",
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }


def run_demo_inference(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, str | int | list[str]]:
    """Create deterministic demo prediction masks without OpenCD."""
    if not model_path.exists():
        raise ValueError(f"Model file {model_path} does not exist")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")

    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    generated: list[str] = []
    for image in images:
        mask_path = output_dir / f"{image.stem}_change_mask.txt"
        mask_path.write_text(
            "\n".join(
                [
                    "demo_change_mask=true",
                    f"source_image={image}",
                    f"model={model_path}",
                    "changed_pixels_estimate=128",
                ]
            ),
            encoding="utf-8",
        )
        generated.append(str(mask_path))

    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(
        "{\n"
        f'  "mode": "demo",\n'
        f'  "model_path": "{model_path}",\n'
        f'  "input_dir": "{input_dir}",\n'
        f'  "prediction_count": {len(generated)}\n'
        "}\n",
        encoding="utf-8",
    )
    return {
        "mode": "demo",
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "prediction_count": len(generated),
        "prediction_files": generated,
        "summary_path": str(summary_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("model", type=Path, help="Path to trained model")
    parser.add_argument("input_dir", type=Path, help="Directory with imagery")
    parser.add_argument("output_dir", type=Path, help="Directory for predictions")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run a local deterministic demo inference instead of OpenCD",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(infer(args.model, args.input_dir, args.output_dir, args.demo_mode))
