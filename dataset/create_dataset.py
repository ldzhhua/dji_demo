"""Dataset preparation utilities for UAV change detection."""

import argparse
import random
import shutil
from pathlib import Path

def create_dataset(
    input_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    *,
    seed: int | None = None,
) -> None:
    """Process raw imagery into a simple train/val split.

    This function copies image files from ``input_dir`` into ``output_dir``
    arranged in ``train`` and ``val`` folders. A ``val_ratio`` fraction of
    images will be placed in the validation set.
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    images = [
        p
        for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    ]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    if seed is not None:
        random.seed(seed)
    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.copy(img, train_dir / img.name)
    for img in val_images:
        shutil.copy(img, val_dir / img.name)

    print(f"Created {len(train_images)} training and {len(val_images)} validation images")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataset for change detection"
    )
    parser.add_argument("input_dir", type=Path, help="Path to raw imagery")
    parser.add_argument("output_dir", type=Path, help="Path to processed dataset")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images used for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        args.input_dir,
        args.output_dir,
        args.val_ratio,
        seed=args.seed,
    )
