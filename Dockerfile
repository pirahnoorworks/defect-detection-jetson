FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY defect_detection ./defect_detection
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "-m", "defect_detection.server"]
