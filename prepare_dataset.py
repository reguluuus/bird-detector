from __future__ import annotations

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


FRAME_NAME_PATTERN = re.compile(r"^(bird_\d+)_(\d+)$")


@dataclass
class Sample:
    video_name: str
    frame_index: int
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]]

    @property
    def is_positive(self) -> bool:
        return bool(self.boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FBD-SV-2024 for YOLO training."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/fbd/FBD-SV-2024"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--hard-negative-distance",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def parse_annotation(path: Path) -> Sample:
    root = ET.parse(path).getroot()

    match = FRAME_NAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected annotation name: {path.name}")

    video_name = match.group(1)
    frame_index = int(match.group(2))

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {path}")

    boxes = []

    for obj in root.findall("object"):
        if obj.findtext("name", "").strip().lower() != "bird":
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width))
        ymin = max(0.0, min(ymin, height))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {path}")

        boxes.append((xmin, ymin, xmax, ymax))

    return Sample(
        video_name=video_name,
        frame_index=frame_index,
        width=width,
        height=height,
        boxes=boxes,
    )


def load_samples(labels_dir: Path) -> list[Sample]:
    annotation_files = sorted(labels_dir.glob("*.xml"))

    if not annotation_files:
        raise FileNotFoundError(
            f"No XML annotations found in {labels_dir}"
        )

    return [
        parse_annotation(path)
        for path in tqdm(
            annotation_files,
            desc=f"Reading {labels_dir.name} annotations",
        )
    ]


def find_hard_negatives(
    samples: list[Sample],
    max_distance: int,
) -> set[tuple[str, int]]:
    positive_frames = defaultdict(list)

    for sample in samples:
        if sample.is_positive:
            positive_frames[sample.video_name].append(
                sample.frame_index
            )

    hard_negatives = set()

    for sample in samples:
        if sample.is_positive:
            continue

        nearby_positives = positive_frames.get(
            sample.video_name,
            [],
        )

        if any(
            abs(sample.frame_index - frame_index) <= max_distance
            for frame_index in nearby_positives
        ):
            hard_negatives.add(
                (sample.video_name, sample.frame_index)
            )

    return hard_negatives


def select_samples(
    samples: list[Sample],
    total_size: int,
    positive_ratio: float,
    hard_negative_distance: int,
    seed: int,
) -> list[Sample]:
    rng = random.Random(seed)

    positives = [
        sample for sample in samples if sample.is_positive
    ]
    negatives = [
        sample for sample in samples if not sample.is_positive
    ]

    positive_count = round(total_size * positive_ratio)
    negative_count = total_size - positive_count

    if len(positives) < positive_count:
        raise ValueError("Not enough positive samples.")

    if len(negatives) < negative_count:
        raise ValueError("Not enough negative samples.")

    selected_positives = rng.sample(
        positives,
        positive_count,
    )

    hard_keys = find_hard_negatives(
        samples,
        hard_negative_distance,
    )

    hard_negatives = [
        sample
        for sample in negatives
        if (sample.video_name, sample.frame_index) in hard_keys
    ]

    regular_negatives = [
        sample
        for sample in negatives
        if (sample.video_name, sample.frame_index) not in hard_keys
    ]

    rng.shuffle(hard_negatives)
    rng.shuffle(regular_negatives)

    selected_negatives = hard_negatives[:negative_count]

    if len(selected_negatives) < negative_count:
        remaining = negative_count - len(selected_negatives)
        selected_negatives.extend(
            regular_negatives[:remaining]
        )

    selected = selected_positives + selected_negatives
    rng.shuffle(selected)

    print(f"Available positives:   {len(positives)}")
    print(f"Available backgrounds: {len(negatives)}")
    print(f"Hard backgrounds:      {len(hard_negatives)}")
    print(f"Selected positives:    {len(selected_positives)}")
    print(f"Selected backgrounds:  {len(selected_negatives)}")

    return selected


def to_yolo(sample: Sample) -> list[str]:
    labels = []

    for xmin, ymin, xmax, ymax in sample.boxes:
        x_center = ((xmin + xmax) / 2) / sample.width
        y_center = ((ymin + ymax) / 2) / sample.height
        width = (xmax - xmin) / sample.width
        height = (ymax - ymin) / sample.height

        labels.append(
            f"0 {x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )

    return labels


def create_output_dirs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_dir / split / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )


def extract_frames(
    samples: list[Sample],
    split: str,
    source_dir: Path,
    output_dir: Path,
) -> None:
    by_video = defaultdict(dict)

    for sample in samples:
        by_video[sample.video_name][sample.frame_index] = sample

    images_dir = output_dir / split / "images"
    labels_dir = output_dir / split / "labels"

    saved = 0

    for video_name, required_frames in tqdm(
        sorted(by_video.items()),
        desc=f"Extracting {split}",
    ):
        video_path = (
            source_dir
            / "videos"
            / split
            / f"{video_name}.mp4"
        )

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        frame_index = 0
        remaining = len(required_frames)

        while remaining:
            success, frame = capture.read()

            if not success:
                break

            sample = required_frames.get(frame_index)

            if sample is not None:
                stem = f"{video_name}_{frame_index:06d}"

                image_path = images_dir / f"{stem}.jpg"
                label_path = labels_dir / f"{stem}.txt"

                if not cv2.imwrite(
                    str(image_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(
                        f"Could not save {image_path}"
                    )

                labels = to_yolo(sample)

                label_path.write_text(
                    "\n".join(labels)
                    + ("\n" if labels else ""),
                    encoding="utf-8",
                )

                remaining -= 1
                saved += 1

            frame_index += 1

        capture.release()

        if remaining:
            raise RuntimeError(
                f"Could not extract all required frames "
                f"from {video_path}"
            )

    print(f"{split}: extracted {saved} images")


def validate_split(
    output_dir: Path,
    split: str,
) -> dict[str, int]:
    images_dir = output_dir / split / "images"
    labels_dir = output_dir / split / "labels"

    images = sorted(images_dir.glob("*.jpg"))

    positives = 0
    backgrounds = 0
    instances = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}"
            )

        labels = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if labels:
            positives += 1
            instances += len(labels)
        else:
            backgrounds += 1

        for label in labels:
            parts = label.split()

            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(
                    f"Invalid YOLO label in {label_path}"
                )

            coordinates = map(float, parts[1:])

            if not all(
                0.0 <= value <= 1.0
                for value in coordinates
            ):
                raise ValueError(
                    f"Invalid coordinates in {label_path}"
                )

    return {
        "images": len(images),
        "positives": positives,
        "backgrounds": backgrounds,
        "instances": instances,
    }


def write_dataset_config(output_dir: Path) -> None:
    config = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "bird"},
    }

    with (output_dir / "dataset.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def print_stats(
    train_stats: dict[str, int],
    val_stats: dict[str, int],
) -> None:
    print("\nDataset prepared successfully.\n")

    print("Train")
    print(f"  Images:      {train_stats['images']}")
    print(f"  Bird images: {train_stats['positives']}")
    print(f"  Backgrounds: {train_stats['backgrounds']}")
    print(f"  Bird boxes:  {train_stats['instances']}")

    print("\nValidation")
    print(f"  Images:      {val_stats['images']}")
    print(f"  Bird images: {val_stats['positives']}")
    print(f"  Backgrounds: {val_stats['backgrounds']}")
    print(f"  Bird boxes:  {val_stats['instances']}")


def main() -> None:
    args = parse_args()

    if not 0 < args.positive_ratio < 1:
        raise ValueError(
            "--positive-ratio must be between 0 and 1."
        )

    train_samples = load_samples(
        args.source / "labels" / "train"
    )
    val_samples = load_samples(
        args.source / "labels" / "val"
    )

    print("\nSelecting training samples")
    selected_train = select_samples(
        train_samples,
        args.train_size,
        args.positive_ratio,
        args.hard_negative_distance,
        args.seed,
    )

    print("\nSelecting validation samples")
    selected_val = select_samples(
        val_samples,
        args.val_size,
        args.positive_ratio,
        args.hard_negative_distance,
        args.seed + 1,
    )

    create_output_dirs(args.output)

    extract_frames(
        selected_train,
        "train",
        args.source,
        args.output,
    )
    extract_frames(
        selected_val,
        "val",
        args.source,
        args.output,
    )

    write_dataset_config(args.output)

    train_stats = validate_split(args.output, "train")
    val_stats = validate_split(args.output, "val")

    print_stats(train_stats, val_stats)


if __name__ == "__main__":
    main()