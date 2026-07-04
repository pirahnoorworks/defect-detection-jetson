from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from defect_detection.db import init_db, list_inspections, save_inspection

app = FastAPI(title="Defect Detection API", version="0.1.0")

DB_PATH = os.getenv("DEFECT_DB_PATH", "defect_inspections.db")
init_db(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return "<h1>Defect Detection API</h1><p>Upload an image to inspect it.</p>"


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> JSONResponse:
    contents = image.file.read()
    image_bytes = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Unable to read image"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_found = len(contours) > 0
    label = "defect" if defect_found else "normal"
    confidence = min(0.99, 0.6 + min(len(contours), 10) * 0.03)

    save_inspection(DB_PATH, image.filename or "upload.jpg", label, confidence)

    return JSONResponse(
        content={
            "filename": image.filename,
            "label": label,
            "confidence": round(float(confidence), 3),
            "defect_count": len(contours),
        }
    )


@app.get("/inspections")
def inspections() -> list[dict]:
    rows = list_inspections(DB_PATH)
    return [
        {
            "id": row[0],
            "image_name": row[1],
            "label": row[2],
            "confidence": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def main() -> None:
    import uvicorn

    uvicorn.run("defect_detection.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
