from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


INPUT_DIR = Path("examples/input")
OUTPUT_DIR = Path("examples/output")
DEFAULT_WEIGHTS = Path("weights/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bird detection on a video."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input video name or path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output video name or path.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to YOLO weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. 0 or cpu.",
    )

    return parser.parse_args()


def resolve_input(value: str) -> Path:
    path = Path(value)

    # Full or relative path supplied directly
    if path.exists():
        return path

    # Short video name from examples/input/
    if path.suffix.lower() != ".mp4":
        path = path.with_suffix(".mp4")

    path = INPUT_DIR / path

    if path.exists():
        return path

    raise FileNotFoundError(f"Video not found: {path}")


def resolve_output(value: str | None, input_path: Path) -> Path:
    if value is None:
        return OUTPUT_DIR / f"{input_path.stem}_detected.mp4"

    path = Path(value)

    if path.parent == Path("."):
        return OUTPUT_DIR / path

    return path


def main() -> None:
    args = parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_value = args.input
    else:
        input_value = input(
            f"Input video name from {INPUT_DIR}/: "
        ).strip()

    input_path = resolve_input(input_value)
    output_path = resolve_output(args.output, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or (
        "0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Weights: {args.weights}")
    print(f"Device:  {device}")
    print(f"Conf:    {args.conf}")

    model = YOLO(str(args.weights))

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    with tqdm(
        total=frame_count,
        desc="Processing",
        unit="frame",
    ) as progress:
        while True:
            success, frame = capture.read()

            if not success:
                break

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            for box in result.boxes:
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().tolist(),
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"bird {confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            progress.update(1)

    capture.release()
    writer.release()

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
