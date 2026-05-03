"""Run model inference for change detection."""

import argparse
import json
from pathlib import Path


def infer(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    demo_mode: bool = False,
    image_a_dir: Path | None = None,
    image_b_dir: Path | None = None,
) -> dict[str, str | int | list[str]]:
    """Apply a trained model to new imagery using OpenCD or the demo runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if demo_mode:
        return run_demo_inference(model_path, input_dir, output_dir)

    if config_path is None:
        raise ValueError("OpenCD inference requires config_path")

    image_pairs = collect_image_pairs(input_dir, image_a_dir, image_b_dir)
    try:
        from opencd.apis import OpenCDInferencer
    except ImportError as e:
        raise RuntimeError(
            "OpenCD inference requires opencd.apis.OpenCDInferencer. "
            "Install OpenCD and its OpenMMLab dependencies first"
        ) from e

    inferencer = OpenCDInferencer(
        model=str(config_path),
        weights=str(model_path),
        classes=("unchanged", "changed"),
        palette=[[0, 0, 0], [255, 255, 255]],
    )
    inferencer(image_pairs, show=False, out_dir=str(output_dir))
    summary_path = output_dir / "inference_summary.json"
    summary = {
        "mode": "opencd",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pair_count": len(image_pairs),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "mode": "opencd",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pair_count": len(image_pairs),
        "summary_path": str(summary_path),
    }


def collect_image_pairs(
    input_dir: Path,
    image_a_dir: Path | None = None,
    image_b_dir: Path | None = None,
) -> list[list[str]]:
    """Collect A/B image pairs for OpenCDInferencer."""
    a_dir = image_a_dir or input_dir / "A"
    b_dir = image_b_dir or input_dir / "B"
    if not a_dir.is_dir() or not b_dir.is_dir():
        raise ValueError(
            "OpenCD inference expects paired images in A/ and B/ directories "
            "or explicit image_a_dir/image_b_dir"
        )

    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    a_images = {
        image.name: image for image in a_dir.iterdir() if image.suffix.lower() in image_suffixes
    }
    b_images = {
        image.name: image for image in b_dir.iterdir() if image.suffix.lower() in image_suffixes
    }
    shared_names = sorted(a_images.keys() & b_images.keys())
    if not shared_names:
        raise RuntimeError(f"No paired images found in {a_dir} and {b_dir}")
    return [[str(a_images[name]), str(b_images[name])] for name in shared_names]


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
    summary = {
        "mode": "demo",
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "prediction_count": len(generated),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    parser.add_argument("--config", type=Path, help="Path to OpenCD config file")
    parser.add_argument("--image-a-dir", type=Path, help="Directory containing pre-change images")
    parser.add_argument("--image-b-dir", type=Path, help="Directory containing post-change images")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run a local deterministic demo inference instead of OpenCD",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        infer(
            args.model,
            args.input_dir,
            args.output_dir,
            args.config,
            args.demo_mode,
            args.image_a_dir,
            args.image_b_dir,
        )
    )
