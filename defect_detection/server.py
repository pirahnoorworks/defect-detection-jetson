from __future__ import annotations

import os

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from defect_detection.db import init_db, list_inspections, save_inspection
from defect_detection.inference import analyze_image

app = FastAPI(title="Defect Detection API", version="0.1.0")

DB_PATH = os.getenv("DEFECT_DB_PATH", "defect_inspections.db")
init_db(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return "<h1>Defect Detection API</h1><p>Upload an image to inspect it.</p>"


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "service": "defect-detection-jetson"})


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> JSONResponse:
    contents = image.file.read()
    result = analyze_image(contents, image.filename or "upload.jpg")
    save_inspection(DB_PATH, result["filename"], result["label"], result["confidence"])
    return JSONResponse(content=result)


@app.get("/camera/stream")
def camera_stream() -> StreamingResponse:
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return JSONResponse(status_code=503, content={"error": "Unable to open camera"})

    def generate() -> object:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = analyze_image(cv2.imencode(".jpg", frame)[1].tobytes(), "camera.jpg")
                annotated = frame.copy()
                for box in result.get("boxes", []):
                    x1 = int(box["x"])
                    y1 = int(box["y"])
                    x2 = int(box["x"] + box["w"])
                    y2 = int(box["y"] + box["h"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, result["label"], (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                _, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                payload = encoded.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                )
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


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

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("defect_detection.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
