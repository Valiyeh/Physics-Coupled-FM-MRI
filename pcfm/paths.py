"""Load the single paths file. Code does not hard-code data or weight locations."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH_KEYS = (
    "fastmri_root",
    "skmtea_root",
    "radiodino_weights",
    "output_dir",
    "cache_dir",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def load_paths(path: str | Path) -> dict[str, Path]:
    """Resolve every entry in the paths file to an absolute path."""
    path = Path(path).resolve()
    raw = load_yaml(path)
    missing = [key for key in PATH_KEYS if key not in raw]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    root = repo_root()
    resolved = {}
    for key in PATH_KEYS:
        entry = Path(str(raw[key])).expanduser()
        if not entry.is_absolute():
            entry = (root / entry).resolve()
        resolved[key] = entry
    return resolved


def run_name(config: dict) -> str:
    dataset = str(config["dataset"]).lower()
    acceleration = int(config["acceleration"])
    return f"{dataset}_r{acceleration}"
