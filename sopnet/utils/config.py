from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: Union[str, Path, None],
    overrides: Iterable[Union[str, Path]] = (),
    inline: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load a YAML config, optionally merging override files and inline values."""
    config: Dict[str, Any] = {}
    if path is not None:
        config = load_yaml(path)
    for override in overrides:
        config = deep_update(config, load_yaml(override))
    if inline:
        config = deep_update(config, inline)
    return config
