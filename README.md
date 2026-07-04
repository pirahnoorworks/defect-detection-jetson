# Jetson Defect Detection Management System

This project demonstrates an edge-friendly defect detection workflow for industrial inspection scenarios. It combines a lightweight FastAPI dashboard, image-based inference, and a YOLOv8 training pipeline in a single local-first prototype that can also be adapted for Jetson deployment.

## Portfolio note

This demo project is created for portfolio purposes. It is intended to showcase a practical end-to-end workflow for image inspection, model training, and local deployment. Further discussion is welcome via email.

This demo project was created with GitHub Copilot assistance.

## Highlights

- Image upload and inspection workflow with a simple web dashboard
- Defect detection inference with annotated preview output
- SQLite-backed inspection history
- YOLOv8 training workflow with a real dataset preparation path
- Deployment-oriented structure suitable for future Jetson and container workflows

## How to run locally

```bash
cd /path/to/defect-detection-jetson
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m defect_detection.app
```

Open http://127.0.0.1:8000/ in a browser to use the dashboard.

## Training workflow

```bash
python -m defect_detection.train --data-root data/kolektorsdd --epochs 3 --run
```

This command prepares the dataset layout, writes a YOLO-compatible configuration, and runs a real training job when the required dependencies are available.

## Project structure

- defect_detection/app.py: FastAPI dashboard and upload workflow
- defect_detection/inference.py: Image analysis and detection logic
- defect_detection/train.py: Dataset preparation and YOLO training entry point
- defect_detection/db.py: Inspection history persistence
- tests/: Regression tests for inference, dataset preparation, and dashboard behavior

## Jetson deployment notes

For Jetson deployment, the repository now includes a dedicated server entry point and a container-oriented workflow:

```bash
python -m defect_detection.server
```

Or build and run the edge container image:

```bash
docker build -f Dockerfile.jetson -t defect-detection-jetson:jetson .
docker run --rm --runtime nvidia -p 8000:8000 -e PORT=8000 defect-detection-jetson:jetson
```

The service exposes a health check at `/health` and uses the same inference logic as the dashboard path.

## Notes

The current implementation is designed as a portfolio demo and a practical foundation for future enhancements such as model serving, better training quality, and deployment packaging.
