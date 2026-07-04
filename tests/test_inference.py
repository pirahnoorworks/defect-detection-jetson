import cv2
import numpy as np

from defect_detection.inference import analyze_image


def test_blank_image_is_classified_as_normal():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)

    result = analyze_image(encoded.tobytes(), "blank.png")

    assert result["filename"] == "blank.png"
    assert result["label"] == "normal"
    assert result["confidence"] >= 0.0


def test_defect_like_patch_is_classified_as_defect():
    image = np.ones((160, 160, 3), dtype=np.uint8) * 200
    image[60:100, 60:100] = 40
    _, encoded = cv2.imencode(".png", image)

    result = analyze_image(encoded.tobytes(), "defect.png")

    assert result["label"] == "defect"
    assert result["defect_count"] >= 1
