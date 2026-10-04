"""
test_phase0.py — Phase 0 sanity checks.

We verify:
1. config.yaml loads correctly via src.config.
2. All key third-party packages import without error.
3. The FastAPI app starts and /health returns 200.

These tests do not need any data or a GPU.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


# ── Test 1: config loads ───────────────────────────────────────────────────────
def test_config_loads():
    """config.yaml must load and expose expected top-level keys."""
    from src.config import cfg

    assert hasattr(cfg, "grid"), "cfg.grid missing"
    assert hasattr(cfg, "time"), "cfg.time missing"
    assert hasattr(cfg, "model"), "cfg.model missing"
    assert cfg.grid.height == 128, f"Expected grid.height=128, got {cfg.grid.height}"
    assert cfg.time.step_minutes == 15, "Expected step_minutes=15"
    assert cfg.time.output_frames == 12, "Expected 12 output frames (180 min)"


# ── Test 2: required packages import ──────────────────────────────────────────
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "scipy",
    "xarray",
    "h5py",
    "pandas",
    "yaml",  # pyyaml exposes as 'yaml'
    "shapely",
    "fastapi",
    "uvicorn",
    "pydantic",
    "lxml",
    "matplotlib",
    "pytest",
    # pysteps: optional on macOS — see optional-requirements.txt
    # ruff: a CLI tool, not importable as a Python module
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_package_importable(package: str):
    """Each required package must be importable in the current environment."""
    try:
        importlib.import_module(package)
    except ImportError as e:
        pytest.fail(f"Package '{package}' failed to import: {e}")


# ── Test 3: API health endpoint ────────────────────────────────────────────────
def test_api_health():
    """The /health endpoint must return HTTP 200 and status='ok'."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    body = response.json()
    assert body["status"] == "ok", f"Expected status='ok', got {body}"
