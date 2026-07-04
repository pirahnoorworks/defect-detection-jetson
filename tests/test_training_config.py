from pathlib import Path

from defect_detection.train import write_data_config


def test_write_data_config_creates_yaml(tmp_path: Path):
    data_root = tmp_path / "kolektorsdd"
    (data_root / "images" / "train").mkdir(parents=True)
    (data_root / "images" / "val").mkdir(parents=True)
    (data_root / "labels" / "train").mkdir(parents=True)
    (data_root / "labels" / "val").mkdir(parents=True)

    yaml_path = write_data_config(data_root, tmp_path / "data.yaml")
    content = yaml_path.read_text()

    assert yaml_path.exists()
    assert "train: " in content
    assert "val: " in content
    assert "names:" in content
