from __future__ import annotations

import cv2
import numpy as np


def analyze_image(contents: bytes, filename: str) -> dict:
    image_bytes = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    variance = float(np.var(blur))

    if variance < 120:
        return {
            "filename": filename,
            "label": "normal",
            "confidence": 0.95,
            "defect_count": 0,
            "boxes": [],
        }

    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_contours = [c for c in contours if cv2.contourArea(c) > 80]
    defect_found = len(significant_contours) > 0
    label = "defect" if defect_found else "normal"
    confidence = min(0.99, 0.7 + min(len(significant_contours), 5) * 0.05)

    boxes = []
    for contour in significant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": round(float(confidence), 3),
                "class_name": "defect",
            }
        )

    return {
        "filename": filename,
        "label": label,
        "confidence": round(float(confidence), 3),
        "defect_count": len(significant_contours),
        "boxes": boxes,
    }
