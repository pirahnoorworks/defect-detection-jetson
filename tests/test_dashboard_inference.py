from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from defect_detection.app import app, _build_preview_image
from defect_detection.inference import analyze_image
from defect_detection.train import prepare_sample_dataset


def test_prepare_sample_dataset_creates_images_and_labels(tmp_path: Path):
    output_dir = prepare_sample_dataset(tmp_path / "synthetic_ds")

    assert (output_dir / "images" / "train").exists()
    assert (output_dir / "labels" / "train").exists()
    assert len(list((output_dir / "images" / "train").glob("*.jpg"))) >= 2
    assert len(list((output_dir / "labels" / "train").glob("*.txt"))) >= 2


def test_analyze_image_returns_boxes_for_defect_patch():
    image = np.ones((160, 160, 3), dtype=np.uint8) * 220
    image[55:95, 55:95] = 35
    _, encoded = cv2.imencode(".png", image)

    result = analyze_image(encoded.tobytes(), "defect.png")

    assert result["label"] == "defect"
    assert len(result["boxes"]) >= 1
    assert result["boxes"][0]["class_name"] == "defect"


def test_predict_endpoint_returns_preview_payload():
    image = np.ones((160, 160, 3), dtype=np.uint8) * 220
    image[55:95, 55:95] = 35
    _, encoded = cv2.imencode(".png", image)

    client = TestClient(app)
    response = client.post("/predict", files={"image": ("defect.png", encoded.tobytes(), "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "defect"
    assert payload["image_preview"].startswith("data:image/png;base64,")


def test_build_preview_image_clamps_boxes_without_index_error():
    image = np.ones((64, 64, 3), dtype=np.uint8) * 220
    _, encoded = cv2.imencode(".png", image)
    result = {"boxes": [{"x": 60, "y": 60, "w": 20, "h": 20}]}

    preview = _build_preview_image(encoded.tobytes(), result)

    assert preview.startswith("data:image/png;base64,")
