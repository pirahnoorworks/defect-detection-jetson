# Jetson deployment guide

This branch adds a deployment-oriented path for running the defect detection service on NVIDIA Jetson devices.

## What changed

- Added a dedicated server entry point in [defect_detection/server.py](defect_detection/server.py)
- Added a `/health` endpoint for readiness checks
- Added a Jetson container definition in [Dockerfile.jetson](Dockerfile.jetson)
- Updated the package metadata in [pyproject.toml](pyproject.toml)

## Run locally

```bash
python -m defect_detection.server
```

Then open:

- http://localhost:8000/health
- http://localhost:8000/docs

## Run with Docker on Jetson

```bash
docker build -f Dockerfile.jetson -t defect-detection-jetson:jetson .
docker run --rm --runtime nvidia -p 8000:8000 -e PORT=8000 defect-detection-jetson:jetson
```

## Next steps for production edge use

- Replace the current classical image-based inference with a YOLOv8 ONNX or TensorRT model export
- Add model versioning and artifact storage
- Add logging, metrics, and monitoring
- Move from SQLite to a more durable database if required

## New capabilities in this branch

### 1. YOLO export for Jetson

You can now export a trained YOLO weights file for edge deployment:

```bash
python -m defect_detection.train --data-root data/kolektorsdd --export-edge --weights runs/defect_detection/weights/best.pt --export-format onnx engine
```

This generates an edge export manifest and copies the exported artifacts into the artifact directory for later deployment.

### 2. Camera streaming inference endpoint

The server now exposes a streaming endpoint at `/camera/stream` that opens a local camera device and returns annotated JPEG frames in real time.

```bash
python -m defect_detection.server
```

Then open:

- http://localhost:8000/camera/stream

### 3. GitHub Actions deployment workflow

A sample workflow is included in [.github/workflows/jetson-deploy.yml](.github/workflows/jetson-deploy.yml) that runs the test/compile step and builds the Jetson container image automatically.
