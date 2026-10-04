"""
advection.py — Simple optical-flow advection baseline for nowcasting.

This replaces pysteps (which fails to build on macOS Apple clang) with a
pure numpy + scipy implementation that does the same conceptual thing:
estimate motion from recent frames and extrapolate forward.

Algorithm (two steps):
1. **Motion estimation** — phase correlation between the last two input
   frames gives a global (dx, dy) shift per time step. Phase correlation
   is an FFT-based technique: the peak of the inverse FFT of the
   normalised cross-power spectrum is the translation between two images.
   Reference: Kuglin & Hines, "The phase correlation image alignment
   method", ICASSP 1975.

2. **Extrapolation** — we apply that same shift cumulatively to the last
   input frame for each future time step using scipy.ndimage.shift.
   This is the "semi-Lagrangian" advection idea from pysteps, simplified
   to a single global motion vector.

Limitations (say this in the viva):
  - One global motion vector per forecast. Real storms have spatially
    varying motion; pysteps and AI models capture this, we don't.
  - No growth/decay model. We just move the last frame.
  - Works best for short lead times (< 60 min); degrades gracefully
    (forecast → blurry persistence) at longer lead times.

Usage:
    from src.baselines.advection import AdvectionBaseline
    baseline = AdvectionBaseline(cfg)
    forecast = baseline.predict(input_frames)  # input: (T_in, H, W)
    # forecast shape: (T_out, H, W)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import shift as nd_shift

logger = logging.getLogger(__name__)


class AdvectionBaseline:
    """
    Optical-flow advection nowcasting baseline.

    This is our "dumb but honest" benchmark. The AI model needs to beat
    this to justify its complexity.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        """
        Args:
            cfg: The global config namespace.
                 Reads cfg.time.output_frames.
        """
        self.output_frames = cfg.time.output_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, input_frames: np.ndarray) -> np.ndarray:
        """
        Produce a multi-step forecast by advecting the last input frame.

        Args:
            input_frames: float32 array shaped (T_in, H, W), values in [0, 1].
                          Must have at least 2 frames for motion estimation.

        Returns:
            float32 array shaped (output_frames, H, W), values in [0, 1].
        """
        if input_frames.ndim != 3:
            raise ValueError(f"Expected (T, H, W), got shape {input_frames.shape}")
        if input_frames.shape[0] < 2:
            logger.warning(
                "Need ≥2 input frames for motion estimation; using persistence."
            )
            return self._persistence(input_frames[-1])

        dx, dy = self._estimate_motion(input_frames[-2], input_frames[-1])
        logger.debug("Estimated motion: dx=%.2f, dy=%.2f pixels/frame", dx, dy)

        last_frame = input_frames[-1]
        forecast = np.stack(
            [
                self._advect(last_frame, dx * (t + 1), dy * (t + 1))
                for t in range(self.output_frames)
            ],
            axis=0,
        )
        return forecast.astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_motion(frame1: np.ndarray, frame2: np.ndarray) -> tuple[float, float]:
        """
        Phase-correlation motion estimation between two frames.

        Returns (dx, dy) in pixels — the shift from frame1 to frame2.
        A positive dx means the storm moved to the right.
        """
        # Apodise edges with a Hann window to reduce spectral leakage.
        H, W = frame1.shape
        win_y = np.hanning(H)
        win_x = np.hanning(W)
        window = np.outer(win_y, win_x)

        f1 = frame1 * window
        f2 = frame2 * window

        F1 = np.fft.fft2(f1)
        F2 = np.fft.fft2(f2)
        cross_power = F1 * np.conj(F2)
        denom = np.abs(cross_power)
        # Avoid division by zero in clear-sky (all-zero) frames.
        with np.errstate(invalid="ignore", divide="ignore"):
            cross_power = np.where(denom > 1e-10, cross_power / denom, 0.0)
        correlation = np.fft.ifft2(cross_power).real

        # The peak of the correlation surface gives the shift.
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        raw_dy, raw_dx = peak

        # FFT shifts are circular: a shift of N/2+k means -(N/2-k).
        dy = raw_dy if raw_dy < H / 2 else raw_dy - H
        dx = raw_dx if raw_dx < W / 2 else raw_dx - W

        # Cap at 20 pixels/frame — larger shifts are almost always noise.
        max_shift = 20.0
        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))
        return dx, dy

    @staticmethod
    def _advect(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        Shift `frame` by (dx, dy) pixels using scipy bilinear interpolation.

        scipy.ndimage.shift uses (row, col) order, so row = dy, col = dx.
        We use mode='nearest' at boundaries so edges don't go to zero
        immediately (more realistic than cval=0 for a moving storm).
        """
        shifted = nd_shift(frame, shift=(dy, dx), order=1, mode="nearest")
        return np.clip(shifted, 0.0, 1.0)

    def _persistence(self, last_frame: np.ndarray) -> np.ndarray:
        """
        Fallback: repeat the last frame for all forecast steps.

        The weakest possible baseline — any model that can't beat this
        should be discarded.
        """
        return np.stack([last_frame.copy()] * self.output_frames, axis=0)


def run_baseline_on_batch(
    input_batch: np.ndarray,
    cfg: SimpleNamespace,
) -> np.ndarray:
    """
    Convenience wrapper: run the advection baseline on a NumPy batch.

    Args:
        input_batch: float32 shaped (B, T_in, H, W).
        cfg:         Global config namespace.

    Returns:
        float32 shaped (B, T_out, H, W).
    """
    baseline = AdvectionBaseline(cfg)
    forecasts = [baseline.predict(input_batch[i]) for i in range(input_batch.shape[0])]
    return np.stack(forecasts, axis=0)
