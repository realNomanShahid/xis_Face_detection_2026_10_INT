"""Load Face Blur pipeline YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_all() -> dict[str, dict[str, Any]]:
    """Load every Face Blur config used by the DeepStream + RetinaFace pipeline."""
    return {
        "pipeline": load_yaml("pipeline.yaml"),
        "detection": load_yaml("detection.yaml"),
        "blur": load_yaml("blur.yaml"),
        "camera": load_yaml("camera.yaml"),
        "encode": load_yaml("encode.yaml"),
        "preview": load_yaml("preview.yaml"),
        "logging": load_yaml("logging.yaml"),
    }


def get(section: str, key: str, default: Any = None) -> Any:
    """Convenience: load_all()[section][key] with optional default."""
    cfg = load_all()
    if section not in cfg:
        raise KeyError(f"Unknown config section: {section}")
    return cfg[section].get(key, default)


if __name__ == "__main__":
    import json

    print(json.dumps(load_all(), indent=2, default=str))
