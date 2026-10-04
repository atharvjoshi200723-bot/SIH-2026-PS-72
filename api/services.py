"""
services.py — Application state, ML inference runner, and shared engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.alerts.district_join import DistrictJoinEngine
from src.alerts.workflow import AlertWorkflowManager
from src.config import cfg
from src.inference import InferenceEngine, InferenceResult


class NowcastService:
    """Manages ML engines and currently cached forecast grids."""

    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.inference_engine = InferenceEngine(config_obj=cfg, device=self.device)
        self.district_join = DistrictJoinEngine(config_obj=cfg)
        self.alert_workflow = AlertWorkflowManager()

        # Cached latest run
        self.latest_result: InferenceResult | None = None
        self.latest_input_frames: np.ndarray | None = None
        self.latest_impacts: list[Any] = []

    def get_sample_input(self, event_idx: int | None = None) -> np.ndarray:
        """Load 4 input frames from synthetic events or generate synthetic storm."""
        files = sorted(Path(cfg.paths.synthetic_root).glob("event_*.npy"))
        if files:
            idx = (event_idx if event_idx is not None else 0) % len(files)
            event_data = np.load(files[idx])  # (20, 128, 128)
            return event_data[: int(cfg.time.input_frames)].copy()

        # Fallback if no files exist: create Gaussian convective blob
        T_in = int(cfg.time.input_frames)
        H, W = int(cfg.grid.height), int(cfg.grid.width)
        grid = np.zeros((T_in, H, W), dtype=np.float32)
        grid[:, 45:75, 50:80] = 0.8
        return grid


# Singleton instance
service = NowcastService()
