"""
test_phase5.py — Verification tests for Phase 5 (Inference Pipeline & Tiered Fallback Engine).

Tests:
1. Sensor health and freshness checking against configured thresholds
2. Fallback tier selection logic (T0 -> T1 -> T2 -> T3)
3. Inference forward pass under T0, T1, T2, and T3 tiers
4. Output grid shapes (12, 128, 128) and range [0, 1]
5. IMD risk classification mapping (Green, Yellow, Orange, Red)
"""

import numpy as np
import pytest

from src.config import load_config
from src.inference import InferenceEngine, InferenceResult


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def engine(cfg):
    return InferenceEngine(config_obj=cfg)


def test_sensor_health_evaluation(engine):
    # All sensors fresh
    health_good = engine.evaluate_sensor_health(
        radar_age_min=5.0,
        satellite_age_min=10.0,
        aws_age_min=15.0,
        nwp_age_min=60.0,
    )
    assert health_good["radar"].is_available
    assert health_good["satellite"].is_available
    assert health_good["aws"].is_available
    assert health_good["nwp"].is_available

    # Radar stale (35 min > max 30 min)
    health_radar_stale = engine.evaluate_sensor_health(
        radar_age_min=35.0,
        satellite_age_min=10.0,
        aws_age_min=15.0,
        nwp_age_min=60.0,
    )
    assert not health_radar_stale["radar"].is_available
    assert health_radar_stale["satellite"].is_available


def test_fallback_tier_selection(engine):
    # 1. Fresh radar -> T0
    h_t0 = engine.evaluate_sensor_health(radar_age_min=5.0)
    assert engine.select_tier(h_t0) == "T0"

    # 2. Radar down (40 min), satellite fresh (15 min) -> T1
    h_t1 = engine.evaluate_sensor_health(radar_age_min=40.0, satellite_age_min=15.0)
    assert engine.select_tier(h_t1) == "T1"

    # 3. Radar down (40 min), satellite down (50 min), NWP fresh (60 min) -> T2
    h_t2 = engine.evaluate_sensor_health(
        radar_age_min=40.0, satellite_age_min=50.0, aws_age_min=90.0, nwp_age_min=60.0
    )
    assert engine.select_tier(h_t2) == "T2"

    # 4. All sensors down / stale -> T3 (Advection / Persistence)
    h_t3 = engine.evaluate_sensor_health(
        radar_age_min=90.0, satellite_age_min=90.0, aws_age_min=120.0, nwp_age_min=300.0
    )
    assert engine.select_tier(h_t3) == "T3"


def test_inference_execution_shapes_and_ranges(engine, cfg):
    T_in = cfg.time.input_frames
    H = cfg.grid.height
    W = cfg.grid.width
    T_out = cfg.time.output_frames

    # Synthetic storm blob in input
    sample_input = np.zeros((T_in, H, W), dtype=np.float32)
    sample_input[:, 40:70, 40:70] = 0.8  # Strong convective core

    res = engine.predict(sample_input, radar_age_min=5.0)
    assert isinstance(res, InferenceResult)
    assert res.forecast_grid.shape == (T_out, H, W)
    assert res.forecast_grid.min() >= 0.0
    assert res.forecast_grid.max() <= 1.0
    assert len(res.risk_by_lead_time) == T_out
    assert res.max_overall_risk in ["Green", "Yellow", "Orange", "Red"]
    assert res.latency_ms > 0.0


def test_forced_tiers_execution(engine, cfg):
    T_in, H, W = cfg.time.input_frames, cfg.grid.height, cfg.grid.width
    sample_input = np.zeros((T_in, H, W), dtype=np.float32)
    sample_input[:, 50:60, 50:60] = 0.6

    for tier in ["T0", "T1", "T2", "T3"]:
        res = engine.predict(sample_input, forced_tier=tier)
        assert res.tier_used == tier
        assert res.forecast_grid.shape == (cfg.time.output_frames, H, W)
        assert 0.0 <= res.confidence <= 1.0
