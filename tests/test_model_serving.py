from pathlib import Path

from defect_detection.train import export_model_artifact


def test_export_model_artifact_creates_manifest(tmp_path: Path):
    output_dir = tmp_path / "model_artifact"
    manifest = export_model_artifact(output_dir, "best.pt")

    assert manifest["weights_path"].endswith("best.pt")
    assert manifest["artifact_dir"] == str(output_dir)
    assert (output_dir / "model_manifest.json").exists()
