import pytest
from pathlib import Path

from vlm_annotate_verify.config import Config, load_config, ConfigError

FIXTURE = Path(__file__).parent / "fixtures" / "mini_dataset"


def test_load_valid_config():
    config = load_config(FIXTURE)
    assert config.operator_id == "nathan"
    assert config.station_id == "home_lab"
    assert config.robot_id == "aloha_01"
    assert config.is_eval_episode is False
    assert config.dataset_name == "mini_dataset"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path)


def test_missing_required_key_raises(tmp_path):
    (tmp_path / "roboannotate.toml").write_text(
        'operator_id = "nathan"\nstation_id = "home"\n'
    )
    with pytest.raises(ConfigError, match="missing"):
        load_config(tmp_path)


def test_bad_toml_raises(tmp_path):
    (tmp_path / "roboannotate.toml").write_text("operator_id = \n")
    with pytest.raises(ConfigError, match="parse"):
        load_config(tmp_path)


def test_returns_frozen_dataclass(tmp_path):
    (tmp_path / "roboannotate.toml").write_text(
        'operator_id = "n"\nstation_id = "s"\nrobot_id = "r"\n'
        'is_eval_episode = true\ndataset_name = "d"\n'
    )
    config = load_config(tmp_path)
    with pytest.raises((AttributeError, Exception)):
        config.operator_id = "other"
