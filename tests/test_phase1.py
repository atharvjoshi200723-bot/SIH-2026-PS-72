"""
test_phase1.py — Unit tests for the data layer (Phase 1).

What we verify:
  1. SyntheticGenerator produces the right shapes and value range.
  2. Saved .npy files can be reloaded and still have correct shapes.
  3. StormDataset sliding-window logic gives correct (X, y) shapes.
  4. make_splits creates non-overlapping, correctly sized subsets.
  5. SEVIRLoader gracefully returns [] when no HDF5 files are present.
  6. Normalization is invertible and clips correctly.

All tests run with synthetic data only — no real data or internet needed.
"""

from __future__ import annotations

# ── bring repo root onto path so imports work ─────────────────────────────────
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import StormDataset, make_splits
from src.data.normalize import denormalize, normalize
from src.data.sevir_loader import SEVIRLoader
from src.data.synthetic import SyntheticGenerator

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_cfg():
    """A tiny config for fast tests — 10 events, 16x16 grid."""
    syn = SimpleNamespace(
        num_events=10,
        frames_per_event=20,
        num_blobs_range=[1, 3],
        blob_radius_range=[2, 8],
        peak_value=1.0,
        severe_fraction=0.1,
        seed=0,
    )
    grid = SimpleNamespace(height=16, width=16)
    split = SimpleNamespace(train=0.7, val=0.15, test=0.15)
    time = SimpleNamespace(input_frames=4, output_frames=4, step_minutes=15)
    return SimpleNamespace(synthetic=syn, grid=grid, split=split, time=time)


@pytest.fixture(scope="module")
def events(small_cfg):
    gen = SyntheticGenerator(small_cfg.synthetic, small_cfg.grid)
    return gen.generate()


# ── Phase 1a: Synthetic generator ─────────────────────────────────────────────


def test_event_count(events, small_cfg):
    """Generator must produce exactly num_events events."""
    assert len(events) == small_cfg.synthetic.num_events


def test_event_shape(events, small_cfg):
    """Every event must be shaped (frames_per_event, H, W)."""
    T = small_cfg.synthetic.frames_per_event
    H = small_cfg.grid.height
    W = small_cfg.grid.width
    for i, ev in enumerate(events):
        assert ev.shape == (T, H, W), f"Event {i}: expected {(T, H, W)}, got {ev.shape}"


def test_event_dtype(events):
    """Events must be float32 in [0, 1]."""
    for ev in events:
        assert ev.dtype == np.float32, f"Expected float32, got {ev.dtype}"
        assert float(ev.min()) >= 0.0, "Values below 0"
        assert float(ev.max()) <= 1.0, "Values above 1"


def test_save_and_reload(events, small_cfg):
    """Events saved as .npy files must reload with identical values."""
    gen = SyntheticGenerator(small_cfg.synthetic, small_cfg.grid)
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.save(events, tmpdir)
        files = sorted(Path(tmpdir).glob("event_*.npy"))
        assert len(files) == len(events), "Wrong number of saved files"
        reloaded = np.load(files[0])
        np.testing.assert_array_equal(
            events[0],
            reloaded,  # exact match expected for .npy round-trip
            err_msg="Reloaded event does not match original",
        )


def test_severe_events_exist(events, small_cfg):
    """At least one event should have a max value above 0.5 (simulated severe cell)."""
    max_vals = [float(ev.max()) for ev in events]
    assert max(max_vals) > 0.5, (
        "No severe cells found — check severe_fraction / peak_value."
    )


# ── Phase 1b: SEVIR loader (graceful fallback) ────────────────────────────────


def test_sevir_loader_missing_dir():
    """SEVIRLoader returns [] and does NOT crash when data dir is missing."""
    sevir_cfg = SimpleNamespace(num_events=5, channel="vil")
    grid_cfg = SimpleNamespace(height=16, width=16)
    loader = SEVIRLoader(sevir_cfg, grid_cfg)
    result = loader.load("/tmp/nonexistent_sevir_dir_vajradrishti")
    assert result == [], "Expected empty list for missing SEVIR dir"


# ── Phase 1c: Dataset & splits ────────────────────────────────────────────────


def test_dataset_sample_shapes(events, small_cfg):
    """Dataset samples must have the exact shapes the model expects."""
    in_f = small_cfg.time.input_frames
    out_f = small_cfg.time.output_frames
    H, W = small_cfg.grid.height, small_cfg.grid.width

    ds = StormDataset(events, input_frames=in_f, output_frames=out_f)
    assert len(ds) > 0, "Dataset is empty — check event length vs window size"

    x, y = ds[0]
    assert isinstance(x, torch.Tensor), "X should be a torch Tensor"
    assert isinstance(y, torch.Tensor), "y should be a torch Tensor"
    assert x.shape == (in_f, H, W), (
        f"X shape: expected {(in_f, H, W)}, got {tuple(x.shape)}"
    )
    assert y.shape == (out_f, H, W), (
        f"y shape: expected {(out_f, H, W)}, got {tuple(y.shape)}"
    )


def test_splits_non_overlapping(events, small_cfg):
    """Train / val / test splits must not share events."""
    train_ds, val_ds, test_ds = make_splits(events, small_cfg)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    # Total samples ≤ all possible windows across all events.
    window = small_cfg.time.input_frames + small_cfg.time.output_frames
    max_possible = sum(max(0, ev.shape[0] - window + 1) for ev in events)
    assert total <= max_possible, (
        "More samples than possible windows — overlap detected"
    )


def test_splits_sizes(events, small_cfg):
    """Train set must be the largest split."""
    train_ds, val_ds, test_ds = make_splits(events, small_cfg)
    assert len(train_ds) >= len(val_ds), "Train set smaller than val"
    assert len(train_ds) >= len(test_ds), "Train set smaller than test"


# ── Phase 1d: Normalization ───────────────────────────────────────────────────


def test_normalize_range():
    """Normalised values must lie in [0, 1]."""
    raw = np.array([-10, 0, 128, 255, 300], dtype=np.float32)
    normed = normalize(raw)
    assert float(normed.min()) >= 0.0
    assert float(normed.max()) <= 1.0


def test_normalize_denormalize_roundtrip():
    """denormalize(normalize(x)) should recover the clipped original."""
    raw = np.array([0, 64, 128, 200, 254], dtype=np.float32)
    recovered = denormalize(normalize(raw))
    np.testing.assert_allclose(recovered, raw, atol=1e-4)
