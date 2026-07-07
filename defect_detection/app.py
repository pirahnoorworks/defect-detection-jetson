from __future__ import annotations

import os
from pathlib import Path

import base64
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from defect_detection.db import init_db, list_inspections, save_inspection
from defect_detection.inference import analyze_image
from defect_detection.train import train_model


def _build_preview_image(contents: bytes, result: dict) -> str:
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = Image.new("RGB", pil_image.size, (255, 255, 255))
    draw.paste(pil_image, (0, 0))

    width, height = draw.size
    for box in result.get("boxes", []):
        x1 = max(0, min(width - 1, int(box.get("x", 0))))
        y1 = max(0, min(height - 1, int(box.get("y", 0))))
        x2 = max(0, min(width, int(box.get("x", 0) + box.get("w", 0))))
        y2 = max(0, min(height, int(box.get("y", 0) + box.get("h", 0))))
        if x2 <= x1 or y2 <= y1:
            continue
        for x in range(x1, x2):
            draw.putpixel((x, y1), (255, 0, 0))
            draw.putpixel((x, y2 - 1), (255, 0, 0))
        for y in range(y1, y2):
            draw.putpixel((x1, y), (255, 0, 0))
            draw.putpixel((x2 - 1, y), (255, 0, 0))

    buffered = io.BytesIO()
    draw.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def _run_yolo_inference(contents: bytes, filename: str) -> dict:
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
    except ImportError:
        return {"filename": filename, "label": "error", "confidence": 0.0, "defect_count": 0, "boxes": [], "image_preview": "", "model": "yolov8n", "note": "YOLO not available"}
    
    boxes = []
    label = "normal"
    confidence = 0.0
    
    try:
        model = YOLO("yolov8n.pt")
        image_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"filename": filename, "label": "error", "confidence": 0.0, "defect_count": 0, "boxes": [], "image_preview": "", "model": "yolov8n", "note": "Failed to decode image"}
        
        results = model(frame, conf=0.3, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    boxes.append({
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                        "confidence": round(conf, 3),
                        "class_name": "object",
                    })
        
        if boxes:
            label = "defect"
            confidence = max([b["confidence"] for b in boxes])
    except Exception as e:
        pass
    
    return {
        "filename": filename,
        "label": label,
        "confidence": round(confidence, 3),
        "defect_count": len(boxes),
        "boxes": boxes,
        "image_preview": _build_preview_image(contents, {"boxes": boxes}),
        "model": "yolov8n",
        "note": "YOLOv8 Nano inference results",
    }

app = FastAPI(title="Defect Detection Management System", version="0.1.0")

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DB_PATH = os.getenv("DEFECT_DB_PATH", "defect_inspections.db")
init_db(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset='utf-8' />
        <title>Defect Detection Dashboard</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #f8fafc; }
          .card { background: #111827; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; }
          form { margin-top: 1rem; }
          button { padding: 0.6rem 1rem; background: #38bdf8; border: none; border-radius: 6px; cursor: pointer; }
          pre { background: #020617; padding: 1rem; border-radius: 8px; overflow: auto; }
          .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: #14532d; color: #dcfce7; font-size: 0.8rem; }
        </style>
      </head>
      <body>
        <div class='card'>
          <h1>Defect Detection Dashboard</h1>
          <p>Edge AI inspection workflow for industrial defect detection.</p>
          <span class='badge'>YOLOv8-ready training pipeline</span>
        </div>
        <div class='card'>
          <h2>Upload image for inspection</h2>
          <form action='/predict' method='post' enctype='multipart/form-data'>
            <input type='file' name='image' accept='image/*' required />
            <button type='submit'>Inspect</button>
          </form>
        </div>
        <div class='card'>
          <h2>Training status</h2>
          <pre id='training'></pre>
          <p><a href='/model-download' download>Download latest model weights</a></p>
        </div>
        <div class='card'>
          <h2>Detection preview</h2>
          <div id='result'>
            <p style='opacity:0.75;'>Upload an image and click Inspect to compare the contour-based and placeholder YOLO-style outputs.</p>
          </div>
        </div>
        <script>
          fetch('/training-status').then(r => r.json()).then(data => {
            document.getElementById('training').textContent = JSON.stringify(data, null, 2);
          });

          document.querySelector('form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const result = document.getElementById('result');
            result.innerHTML = "<p>Running inference…</p>";
            const formData = new FormData(event.target);
            try {
              const response = await fetch('/predict', { method: 'POST', body: formData });
              const data = await response.json();
              result.innerHTML = `
              <div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem;'>
                <div class='card'>
                  <h3>Contour-based inference</h3>
                  <p><strong>Label:</strong> ${data.contour.label} &nbsp; <strong>Confidence:</strong> ${data.contour.confidence}</p>
                  <img src='${data.contour.image_preview}' alt='contour-based preview' style='max-width:100%; border-radius:8px;' />
                  <p>Detected boxes: ${data.contour.boxes.length}</p>
                </div>
                <div class='card'>
                  <h3>YOLO inference</h3>
                  <p><strong>Label:</strong> ${data.yolo.label} &nbsp; <strong>Confidence:</strong> ${data.yolo.confidence}</p>
                  <img src='${data.yolo.image_preview}' alt='yolo preview' style='max-width:100%; border-radius:8px;' />
                  <p>Detected boxes: ${data.yolo.boxes.length}</p>
                  <p><em>${data.yolo.note}</em></p>
                </div>
              </div>
            `;
            } catch (error) {
              result.innerHTML = `<p style='color:#fda4af;'>Inference failed. Please try again.</p>`;
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> JSONResponse:
    contents = image.file.read()
    contour_result = analyze_image(contents, image.filename or "upload.jpg")
    save_inspection(DB_PATH, contour_result["filename"], contour_result["label"], contour_result["confidence"])

    contour_result["image_preview"] = _build_preview_image(contents, contour_result)
    yolo_result = _run_yolo_inference(contents, contour_result["filename"])
    return JSONResponse(
        content={
            "filename": contour_result["filename"],
            "label": contour_result["label"],
            "confidence": contour_result["confidence"],
            "defect_count": contour_result["defect_count"],
            "boxes": contour_result["boxes"],
            "image_preview": contour_result["image_preview"],
            "contour": contour_result,
            "yolo": yolo_result,
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


@app.get("/training-status")
def training_status() -> dict:
    return train_model("data/kolektorsdd", epochs=3, dry_run=True)


@app.get("/model-download")
def model_download() -> FileResponse:
    artifact_path = Path("data/kolektorsdd/artifacts/best.pt")
    if not artifact_path.exists():
        artifact_path = Path("data/kolektorsdd/artifacts/model_manifest.json")
    return FileResponse(artifact_path, media_type="application/octet-stream", filename=artifact_path.name)


def main() -> None:
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("defect_detection.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
