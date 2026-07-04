from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from defect_detection.data_utils import build_yolo_manifest


def write_data_config(data_root: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(data_root)
    output = Path(output_path or root / "data.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)

    content = f"""path: {root}
train: images/train
val: images/val

ncfg: yolov8n.yaml
names:
  0: defect
"""
    output.write_text(content)
    return output


def prepare_sample_dataset(output_root: str | Path) -> Path:
    root = Path(output_root)
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, split in enumerate(["train", "train", "val", "val"]):
        image = np.ones((160, 160, 3), dtype=np.uint8) * 220
        if index % 2 == 0:
            image[50:95, 50:95] = 35
        else:
            image[70:110, 70:110] = 30
        path = root / "images" / split / f"sample_{index}.jpg"
        cv2.imwrite(str(path), image)

        label_path = root / "labels" / split / f"sample_{index}.txt"
        label_path.write_text("0 0.5 0.5 0.28 0.28\n")

    return root


def prepare_real_dataset(output_root: str | Path) -> Path:
    root = Path(output_root)
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(root.rglob("*.jpg"))
    if not image_paths:
        image_paths = sorted(root.rglob("*.png"))

    for index, image_path in enumerate(image_paths):
        split = "train" if index % 5 != 0 else "val"
        dest_image = root / "images" / split / image_path.name
        dest_image.write_bytes(image_path.read_bytes())

        mask_path_candidates = [
            image_path.with_name(f"{image_path.stem}_label.bmp"),
            image_path.with_name(f"{image_path.stem}label.bmp"),
            image_path.with_suffix(".bmp"),
        ]
        mask_path = next((p for p in mask_path_candidates if p.exists()), None)

        label_path = root / "labels" / split / f"{image_path.stem}.txt"
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    w = max(1, x_max - x_min + 1)
                    h = max(1, y_max - y_min + 1)
                    center_x = (x_min + x_max) / (2 * mask.shape[1])
                    center_y = (y_min + y_max) / (2 * mask.shape[0])
                    box_w = w / mask.shape[1]
                    box_h = h / mask.shape[0]
                    label_path.write_text(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")
                    continue

        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

    return root


def export_model_artifact(output_dir: str | Path, weights_path: str | Path) -> dict:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = Path(weights_path)
    target = artifact_dir / source.name
    if source.exists():
        shutil.copy2(source, target)
    manifest = {
        "artifact_dir": str(artifact_dir),
        "weights_path": str(target),
        "model": "yolov8n",
    }
    (artifact_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_model_for_edge(
    weights_path: str | Path,
    output_dir: str | Path | None = None,
    formats: Iterable[str] = ("onnx", "engine"),
    imgsz: int = 640,
) -> dict:
    source = Path(weights_path)
    if not source.exists():
        raise FileNotFoundError(f"Weights file not found: {source}")

    export_root = Path(output_dir or source.parent / "edge")
    export_root.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics to export YOLO model artifacts") from exc

    model = YOLO(str(source))
    exported_paths: dict[str, str | dict[str, str]] = {}
    for fmt in formats:
        try:
            exported_path = str(model.export(format=fmt, imgsz=imgsz, simplify=True))
            exported_file = Path(exported_path)
            if exported_file.exists():
                target_path = export_root / exported_file.name
                if exported_file != target_path:
                    shutil.copy2(exported_file, target_path)
                exported_paths[fmt] = str(target_path)
            else:
                exported_paths[fmt] = exported_path
        except Exception as exc:  # pragma: no cover - defensive path for Jetson environments
            exported_paths[fmt] = {"error": str(exc)}

    manifest = {
        "weights_path": str(source.resolve()),
        "export_dir": str(export_root.resolve()),
        "exports": exported_paths,
    }
    (export_root / "edge_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def train_model(
    data_root: str | Path,
    epochs: int = 1,
    dry_run: bool = True,
    output_path: str | Path | None = None,
) -> dict:
    root = Path(data_root)
    default_config = root / "data.yaml"
    parent_config = root.parent / "data.yaml"
    if output_path is None:
        output_path = default_config if default_config.exists() else parent_config if parent_config.exists() else default_config
    config_path = write_data_config(root, output_path)
    train_files = build_yolo_manifest(root, split="train")
    val_files = build_yolo_manifest(root, split="val")

    plan = {
        "data_root": str(root),
        "config_path": str(config_path),
        "epochs": epochs,
        "train_images": len(train_files),
        "val_images": len(val_files),
        "dry_run": dry_run,
        "model": "yolov8n.pt",
        "notes": "Ready for YOLOv8 training with a KolektorSDD-style dataset.",
    }

    if dry_run:
        return plan

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics to run a real YOLOv8 training job") from exc

    model = YOLO("yolov8n.pt")
    model.train(data=str(config_path), epochs=epochs, imgsz=640, project=str(root / "runs"), name="defect_detection")
    best_weights = root / "runs" / "defect_detection" / "weights" / "best.pt"
    export_model_artifact(root / "artifacts", best_weights)
    export_model_for_edge(best_weights, output_dir=root / "artifacts" / "edge")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8n training configuration for KolektorSDD-style data")
    parser.add_argument("--data-root", default="data/kolektorsdd", help="Path to a dataset root")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--run", action="store_true", help="Execute a real YOLOv8 training run")
    parser.add_argument("--prepare-sample", action="store_true", help="Generate a small synthetic defect dataset for local testing")
    parser.add_argument("--export-edge", action="store_true", help="Export trained YOLO weights to ONNX/TensorRT artifacts")
    parser.add_argument("--weights", default=None, help="Optional path to a YOLO weights file for edge export")
    parser.add_argument("--export-format", nargs="+", default=["onnx", "engine"], choices=["onnx", "engine"], help="Model export formats for Jetson")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.prepare_sample:
        prepare_sample_dataset(root)
        print(f"Prepared sample dataset at {root}")
    else:
        prepare_real_dataset(root)

    if args.export_edge:
        weights_path = Path(args.weights) if args.weights else root / "runs" / "defect_detection" / "weights" / "best.pt"
        print(json.dumps(export_model_for_edge(weights_path, output_dir=root / "artifacts" / "edge", formats=tuple(args.export_format)), indent=2))
        return

    result = train_model(root, epochs=args.epochs, dry_run=not args.run, output_path=root / "data.yaml")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
