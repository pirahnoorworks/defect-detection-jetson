from pathlib import Path

from defect_detection.data_utils import build_yolo_manifest
from defect_detection.db import init_db, list_inspections


def test_build_yolo_manifest_finds_images(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)

    (tmp_path / "images" / "train" / "sample.jpg").write_bytes(b"fake")
    (tmp_path / "images" / "val" / "sample2.jpg").write_bytes(b"fake")

    train_files = build_yolo_manifest(tmp_path, split="train")
    val_files = build_yolo_manifest(tmp_path, split="val")

    assert len(train_files) == 1
    assert len(val_files) == 1
    assert train_files[0].name == "sample.jpg"


def test_database_initialization_creates_table(tmp_path: Path):
    db_path = tmp_path / "inspect.db"
    init_db(db_path)
    inspections = list_inspections(db_path)
    assert inspections == []
