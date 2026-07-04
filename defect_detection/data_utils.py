from __future__ import annotations

from pathlib import Path
from typing import List


def build_yolo_manifest(data_root: str | Path, split: str = "train") -> List[Path]:
    """Build a simple manifest of image files for a YOLO-style dataset."""
    root = Path(data_root)
    image_dir = root / "images" / split
    if not image_dir.exists():
        return []
    return sorted(image_dir.glob("*"))
