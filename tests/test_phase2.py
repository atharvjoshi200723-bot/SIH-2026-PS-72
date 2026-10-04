"""
test_phase2.py — Unit tests for the advection baseline (Phase 2).

What we verify:
  1. Output shape is (output_frames, H, W) for a valid input.
  2. Values stay in [0, 1] after advection.
  3. Motion estimation returns finite floats within the capped range.
  4. The persistence fallback (< 2 input frames) still gives correct shape.
  5. The batch wrapper produces correct batch shapes.
  6. A moving blob is detectably shifted in the right direction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.advection import AdvectionBaseline, run_baseline_on_batch

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def baseline_cfg():
    return SimpleNamespace(time=SimpleNamespace(output_frames=4))


@pytest.fixture
def moving_input():
    """
    Two input frames with a blob shifted 5 pixels to the right between them.
    This gives us a ground-truth motion we can check against.
    """
    H, W = 32, 32
    frames = np.zeros((2, H, W), dtype=np.float32)
    # Frame 0: blob at column 10
    frames[0, 14:18, 8:12] = 0.8
    # Frame 1: same blob shifted 5 pixels right (column 15)
    frames[1, 14:18, 13:17] = 0.8
    return frames


@pytest.fixture
def random_input():
    """4 random frames that have some structure for motion estimation."""
    rng = np.random.default_rng(42)
    # Use smooth blobs rather than pure noise — phase correlation needs structure.
    from scipy.ndimage import gaussian_filter

    raw = rng.random((4, 64, 64)).astype(np.float32)
    smooth = np.stack([gaussian_filter(f, sigma=4) for f in raw], axis=0)
    return (smooth / smooth.max()).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_output_shape(random_input, baseline_cfg):
    """Forecast must have shape (output_frames, H, W)."""
    baseline = AdvectionBaseline(baseline_cfg)
    forecast = baseline.predict(random_input)
    T_out = baseline_cfg.time.output_frames
    H, W = random_input.shape[1], random_input.shape[2]
    assert forecast.shape == (T_out, H, W), (
        f"Expected {(T_out, H, W)}, got {forecast.shape}"
    )


def test_output_dtype(random_input, baseline_cfg):
    """Output must be float32."""
    baseline = AdvectionBaseline(baseline_cfg)
    forecast = baseline.predict(random_input)
    assert forecast.dtype == np.float32


def test_output_range(random_input, baseline_cfg):
    """Advected values must stay in [0, 1]."""
    baseline = AdvectionBaseline(baseline_cfg)
    forecast = baseline.predict(random_input)
    assert float(forecast.min()) >= 0.0, "Values below 0"
    assert float(forecast.max()) <= 1.0, "Values above 1"


def test_motion_estimation_finite(moving_input):
    """Phase correlation must return finite floats."""
    dx, dy = AdvectionBaseline._estimate_motion(moving_input[0], moving_input[1])
    assert np.isfinite(dx), f"dx is not finite: {dx}"
    assert np.isfinite(dy), f"dy is not finite: {dy}"


def test_motion_estimation_direction(moving_input):
    """
    Blob moved 5 pixels to the right (frame0 → frame1).

    Phase correlation finds the shift to align frame1 back to frame0,
    which is the *negative* of the blob's motion. So dx = -5.
    The advection code uses this directly: shifting frame by dx per step
    means the blob continues rightward in the forecast (frame gets shifted
    by -5 per step = moves 5 px left relative to background = blob moves
    right). This is the correct physical interpretation.

    We check sign consistency here rather than absolute direction.
    """
    dx, _dy = AdvectionBaseline._estimate_motion(moving_input[0], moving_input[1])
    # dx is negative because the storm moved +5 px right.
    assert dx < 0, f"Expected dx < 0 for rightward blob motion, got dx={dx:.2f}"
    assert abs(abs(dx) - 5) <= 3, f"Expected |dx| ≈ 5, got dx={dx:.2f}"


def test_motion_capped():
    """Extreme noise frames must not produce unreasonably large shifts."""
    rng = np.random.default_rng(99)
    noise1 = rng.random((64, 64)).astype(np.float32)
    noise2 = rng.random((64, 64)).astype(np.float32)
    dx, dy = AdvectionBaseline._estimate_motion(noise1, noise2)
    assert abs(dx) <= 20.0, f"dx exceeds cap: {dx}"
    assert abs(dy) <= 20.0, f"dy exceeds cap: {dy}"


def test_persistence_fallback(baseline_cfg):
    """Single-frame input must fall back to persistence with correct shape."""
    one_frame = np.ones((1, 16, 16), dtype=np.float32) * 0.5
    baseline = AdvectionBaseline(baseline_cfg)
    forecast = baseline.predict(one_frame)
    T_out = baseline_cfg.time.output_frames
    assert forecast.shape == (T_out, 16, 16)
    # All frames must be identical to the last input frame.
    np.testing.assert_array_equal(forecast[0], one_frame[-1])


def test_batch_wrapper_shape(random_input, baseline_cfg):
    """run_baseline_on_batch must handle (B, T, H, W) input."""
    batch = np.stack([random_input] * 3, axis=0)  # B=3
    result = run_baseline_on_batch(batch, baseline_cfg)
    T_out = baseline_cfg.time.output_frames
    H, W = random_input.shape[1], random_input.shape[2]
    assert result.shape == (3, T_out, H, W), (
        f"Expected (3, {T_out}, {H}, {W}), got {result.shape}"
    )


def test_zero_frame_input(baseline_cfg):
    """All-zero (clear-sky) input must not crash and must return zeros."""
    zero_input = np.zeros((4, 32, 32), dtype=np.float32)
    baseline = AdvectionBaseline(baseline_cfg)
    forecast = baseline.predict(zero_input)
    assert forecast.shape[0] == baseline_cfg.time.output_frames
    assert float(forecast.max()) == 0.0, (
        "Expected all-zero forecast for clear-sky input"
    )
