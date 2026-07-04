from pathlib import Path

from defect_detection.train import train_model, write_data_config


def test_train_model_dry_run_creates_plan(tmp_path: Path):
    data_root = tmp_path / "kolektorsdd"
    (data_root / "images" / "train").mkdir(parents=True)
    (data_root / "images" / "val").mkdir(parents=True)
    (data_root / "labels" / "train").mkdir(parents=True)
    (data_root / "labels" / "val").mkdir(parents=True)

    config_path = write_data_config(data_root, tmp_path / "data.yaml")
    plan = train_model(data_root, epochs=1, dry_run=True)

    assert config_path.exists()
    assert plan["config_path"] == str(config_path)
    assert plan["epochs"] == 1
    assert plan["dry_run"] is True
