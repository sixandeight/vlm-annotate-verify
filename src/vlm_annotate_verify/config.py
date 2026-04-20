"""Load and validate roboannotate.toml at the dataset root."""
import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("operator_id", "station_id", "robot_id", "is_eval_episode", "dataset_name")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    operator_id: str
    station_id: str
    robot_id: str
    is_eval_episode: bool
    dataset_name: str


def load_config(dataset_root: Path) -> Config:
    config_path = dataset_root / "roboannotate.toml"
    if not config_path.exists():
        raise ConfigError(f"config not found at {config_path}")
    try:
        data = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"could not parse {config_path}: {e}") from e
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ConfigError(f"config missing required keys: {missing}")
    return Config(
        operator_id=data["operator_id"],
        station_id=data["station_id"],
        robot_id=data["robot_id"],
        is_eval_episode=data["is_eval_episode"],
        dataset_name=data["dataset_name"],
    )
