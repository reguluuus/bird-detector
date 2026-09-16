from pathlib import Path
import argparse
import shutil

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26n for bird detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dataset.yaml"),
        help="Path to dataset configuration.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="bird_yolo26n",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {args.data}"
        )

    device = args.device or (
        "0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dataset: {args.data}")
    print(f"Device:  {device}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"Image:   {args.imgsz}")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        patience=15,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        seed=42,
        deterministic=True,
        cache=False,
        project=str(Path("runs").resolve()),
        name=args.name,
        plots=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    output_dir = Path("weights")
    output_dir.mkdir(exist_ok=True)

    output_weights = output_dir / "best.pt"
    shutil.copy2(best_weights, output_weights)

    print(f"\nTraining results: {run_dir}")
    print(f"Best weights:     {output_weights}")


if __name__ == "__main__":
    main()