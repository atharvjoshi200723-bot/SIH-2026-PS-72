"""
config.py — Load and expose the master config.yaml as a simple Python object.

Usage:
    from src.config import cfg
    print(cfg.grid.height)  # 128

This keeps all config access in one place so we never have to hunt for
hard-coded numbers scattered across files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _dict_to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a nested dict into dot-accessible SimpleNamespace."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """
    Load config.yaml from the given path, or use the default location.

    The default is configs/config.yaml relative to the repository root,
    which we locate by walking up from this file.
    """
    if path is None:
        # Walk up from src/ to find the repo root (where configs/ lives).
        repo_root = Path(__file__).parent.parent
        path = repo_root / "configs" / "config.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)


# Module-level singleton so callers just do: from src.config import cfg
cfg = load_config()
