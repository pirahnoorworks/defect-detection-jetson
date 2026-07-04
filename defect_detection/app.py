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
          <h2>Recent inspections</h2>
          <pre id='inspections'></pre>
        </div>
        <div class='card'>
          <h2>Detection preview</h2>
          <div id='result'></div>
        </div>
        <script>
          fetch('/inspections').then(r => r.json()).then(data => {
            document.getElementById('inspections').textContent = JSON.stringify(data, null, 2);
          });
          fetch('/training-status').then(r => r.json()).then(data => {
            document.getElementById('training').textContent = JSON.stringify(data, null, 2);
          });

          document.querySelector('form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(event.target);
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            const result = document.getElementById('result');
            result.innerHTML = `<p><strong>Label:</strong> ${data.label} &nbsp; <strong>Confidence:</strong> ${data.confidence}</p><img src='${data.image_preview}' alt='inspection preview' style='max-width:100%; border-radius:8px;' />`;
            if (data.boxes && data.boxes.length > 0) {
              result.innerHTML += `<p>Detected boxes: ${data.boxes.length}</p>`;
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> JSONResponse:
    contents = image.file.read()
    result = analyze_image(contents, image.filename or "upload.jpg")
    save_inspection(DB_PATH, result["filename"], result["label"], result["confidence"])

    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = Image.new("RGB", pil_image.size, (255, 255, 255))
    draw.paste(pil_image, (0, 0))

    for box in result.get("boxes", []):
        x1 = int(box["x"])
        y1 = int(box["y"])
        x2 = int(box["x"] + box["w"])
        y2 = int(box["y"] + box["h"])
        for x in range(x1, min(x2, draw.width - 1)):
            draw.putpixel((x, y1), (255, 0, 0))
            draw.putpixel((x, y2 - 1), (255, 0, 0))
        for y in range(y1, min(y2, draw.height - 1)):
            draw.putpixel((x1, y), (255, 0, 0))
            draw.putpixel((x2 - 1, y), (255, 0, 0))

    buffered = io.BytesIO()
    draw.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    result["image_preview"] = f"data:image/png;base64,{image_b64}"
    return JSONResponse(content=result)


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
