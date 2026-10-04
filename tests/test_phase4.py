"""
test_phase4.py — Verification tests for Phase 4 (Training & Evaluation pipeline).

Tests:
1. Contingency table calculations (TP, FP, FN, TN)
2. Skill score formulas (CSI, POD, FAR)
3. Fractions Skill Score (FSS) neighborhood evaluation
4. Training execution test (saves best_model.pt checkpoint)
5. Evaluation execution test (generates results/metrics.json and skill_scores.png)
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.evaluate import (
    compute_contingency,
    compute_csi,
    compute_far,
    compute_fss,
    compute_pod,
    evaluate_forecasts,
    run_evaluation,
)
from src.train import train_pipeline


@pytest.fixture
def cfg():
    return load_config()


def test_contingency_table_and_scores():
    # 10x10 grid with 4 quadrants
    pred = np.zeros((10, 10), dtype=np.float32)
    target = np.zeros((10, 10), dtype=np.float32)

    # TP region (top-left 5x5 = 25 pixels)
    pred[:5, :5] = 0.8
    target[:5, :5] = 0.8

    # FP region (top-right 5x5 = 25 pixels)
    pred[:5, 5:] = 0.8
    target[:5, 5:] = 0.1

    # FN region (bottom-left 5x5 = 25 pixels)
    pred[5:, :5] = 0.1
    target[5:, :5] = 0.8

    # TN region (bottom-right 5x5 = 25 pixels)
    pred[5:, 5:] = 0.1
    target[5:, 5:] = 0.1

    threshold = 0.5
    tp, fp, fn, tn = compute_contingency(pred, target, threshold)

    assert tp == 25
    assert fp == 25
    assert fn == 25
    assert tn == 25

    csi = compute_csi(tp, fp, fn)  # 25 / (25 + 25 + 25) = 1/3
    pod = compute_pod(tp, fn)  # 25 / (25 + 25) = 0.5
    far = compute_far(tp, fp)  # 25 / (25 + 25) = 0.5

    assert abs(csi - 1.0 / 3.0) < 1e-5
    assert abs(pod - 0.5) < 1e-5
    assert abs(far - 0.5) < 1e-5


def test_fss_perfect_and_worst():
    # Perfect forecast
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 10:20] = 0.9

    fss_perfect = compute_fss(img, img, threshold=0.5, window_size=5)
    assert abs(fss_perfect - 1.0) < 1e-5

    # Completely non-overlapping forecast far apart
    img_far = np.zeros((32, 32), dtype=np.float32)
    img_far[0:5, 0:5] = 0.9

    fss_bad = compute_fss(img_far, img, threshold=0.5, window_size=3)
    assert fss_bad < 0.2


def test_evaluate_forecasts_structure(cfg):
    N, T_out, H, W = 2, 12, 32, 32
    preds = np.random.rand(N, T_out, H, W).astype(np.float32)
    targets = np.random.rand(N, T_out, H, W).astype(np.float32)

    metrics = evaluate_forecasts(
        preds,
        targets,
        lead_times_min=[15, 30, 60],
        step_minutes=15,
        storm_threshold=0.4,
    )

    for key in ["csi", "pod", "far", "fss", "mse", "mae"]:
        assert key in metrics
        assert "15" in metrics[key]
        assert "30" in metrics[key]
        assert "60" in metrics[key]
        assert 0.0 <= metrics[key]["15"] <= 1.0 or key in ["mse", "mae"]


def test_training_and_evaluation_pipeline(tmp_path, cfg):
    # Run training for 1 epoch on 10 synthetic events
    ckpt_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"

    train_res = train_pipeline(
        config_obj=cfg,
        max_epochs=1,
        batch_size=2,
        device_name="cpu",
        data_dir="data/synthetic",
        limit_events=10,
        save_dir=str(ckpt_dir),
    )

    best_ckpt = Path(train_res["best_checkpoint"])
    assert best_ckpt.exists()

    # Verify checkpoint contents
    ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_data
    assert "train_loss" in ckpt_data

    # Run evaluation using this checkpoint
    _eval_res = run_evaluation(
        config_obj=cfg,
        checkpoint_path=str(best_ckpt),
        data_dir="data/synthetic",
        output_dir=str(results_dir),
        limit_events=10,
    )

    # Check metrics.json
    metrics_file = results_dir / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        saved_metrics = json.load(f)
    assert "ai_model" in saved_metrics
    assert "advection_baseline" in saved_metrics
    assert saved_metrics["num_test_samples"] > 0

    # Check skill_scores.png
    plot_file = results_dir / "skill_scores.png"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000  # valid image file
