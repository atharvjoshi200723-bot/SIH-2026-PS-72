"""
evaluate.py — Comparative evaluation of VajraDrishti AI vs Advection Baseline.

Computes standard meteorological verification metrics per lead time:
- Critical Success Index (CSI / Threat Score)
- Probability of Detection (POD / Hit Rate)
- False Alarm Ratio (FAR)
- Fractions Skill Score (FSS, Roberts & Lean 2008)
- Mean Squared Error (MSE) & Mean Absolute Error (MAE)

Saves results to:
- results/metrics.json
- results/skill_scores.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader

from src.baselines.advection import AdvectionBaseline
from src.config import cfg, load_config
from src.data.dataset import make_splits
from src.models.unet_convlstm import UNetConvLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_contingency(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """
    Compute 2x2 contingency table elements (TP, FP, FN, TN) for binary event.
    """
    p_bin = pred >= threshold
    t_bin = target >= threshold

    tp = int(np.sum(p_bin & t_bin))
    fp = int(np.sum(p_bin & ~t_bin))
    fn = int(np.sum(~p_bin & t_bin))
    tn = int(np.sum(~p_bin & ~t_bin))
    return tp, fp, fn, tn


def compute_csi(tp: int, fp: int, fn: int) -> float:
    """Critical Success Index (CSI) = TP / (TP + FP + FN). Range: [0, 1]."""
    denom = tp + fp + fn
    if denom == 0:
        return 1.0  # Perfect prediction when no storm occurred and none predicted
    return float(tp / denom)


def compute_pod(tp: int, fn: int) -> float:
    """Probability of Detection (POD) = TP / (TP + FN). Range: [0, 1]."""
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def compute_far(tp: int, fp: int) -> float:
    """False Alarm Ratio (FAR) = FP / (TP + FP). Range: [0, 1] (lower is better)."""
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(fp / denom)


def compute_fss(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
    window_size: int = 9,
) -> float:
    """
    Fractions Skill Score (FSS) over 2D spatial neighborhood (Roberts & Lean 2008).

    Args:
        pred: Predicted 2D reflectivity map (H, W).
        target: Ground truth 2D reflectivity map (H, W).
        threshold: Event binarization threshold.
        window_size: Neighborhood box size in pixels (odd integer).

    Returns:
        float score in [0, 1].
    """
    p_bin = (pred >= threshold).astype(np.float32)
    t_bin = (target >= threshold).astype(np.float32)

    p_frac = uniform_filter(p_bin, size=window_size, mode="constant", cval=0.0)
    t_frac = uniform_filter(t_bin, size=window_size, mode="constant", cval=0.0)

    fbs = np.mean((p_frac - t_frac) ** 2)
    fbs_worst = np.mean(p_frac**2 + t_frac**2)

    if fbs_worst < 1e-7:
        return 1.0 if fbs < 1e-7 else 0.0

    score = 1.0 - (fbs / fbs_worst)
    return float(np.clip(score, 0.0, 1.0))


def evaluate_forecasts(
    y_preds: np.ndarray,
    y_trues: np.ndarray,
    lead_times_min: list[int],
    step_minutes: int,
    storm_threshold: float,
) -> dict[str, dict[str, float]]:
    """
    Compute verification metrics for a batch of forecasts across specified lead times.

    Args:
        y_preds: Array shaped (N, T_out, H, W)
        y_trues: Array shaped (N, T_out, H, W)
        lead_times_min: List of lead times in minutes (e.g. [15, 30, 60...])
        step_minutes: Minutes per forecast step (e.g. 15)
        storm_threshold: Normalised reflectivity threshold

    Returns:
        Dict mapping metric_name -> {lead_time_str: score}
    """
    results: dict[str, dict[str, float]] = {
        "csi": {},
        "pod": {},
        "far": {},
        "fss": {},
        "mse": {},
        "mae": {},
    }

    N = y_preds.shape[0]

    for lead_min in lead_times_min:
        step_idx = (lead_min // step_minutes) - 1
        lead_str = str(lead_min)

        if step_idx >= y_preds.shape[1]:
            logger.warning(
                "Lead time %d min exceeds forecast length; skipping.", lead_min
            )
            continue

        p_step = y_preds[:, step_idx]  # (N, H, W)
        t_step = y_trues[:, step_idx]  # (N, H, W)

        tp, fp, fn, _ = compute_contingency(p_step, t_step, storm_threshold)
        csi_val = compute_csi(tp, fp, fn)
        pod_val = compute_pod(tp, fn)
        far_val = compute_far(tp, fp)

        # Average FSS across all test cases
        fss_scores = [
            compute_fss(p_step[i], t_step[i], storm_threshold) for i in range(N)
        ]
        fss_val = float(np.mean(fss_scores)) if fss_scores else 0.0

        mse_val = float(np.mean((p_step - t_step) ** 2))
        mae_val = float(np.mean(np.abs(p_step - t_step)))

        results["csi"][lead_str] = round(csi_val, 4)
        results["pod"][lead_str] = round(pod_val, 4)
        results["far"][lead_str] = round(far_val, 4)
        results["fss"][lead_str] = round(fss_val, 4)
        results["mse"][lead_str] = round(mse_val, 4)
        results["mae"][lead_str] = round(mae_val, 4)

    return results


def plot_skill_curves(
    ai_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
    lead_times_min: list[int],
    output_path: Path,
) -> None:
    """Generate and save skill curves comparison plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leads = [str(m) for m in lead_times_min if str(m) in ai_metrics["csi"]]
    x_vals = [int(m) for m in leads]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "VajraDrishti: AI (UNet+ConvLSTM) vs Advection Baseline Skill Comparison",
        fontsize=14,
        weight="bold",
    )

    plots_cfg = [
        ("csi", "Critical Success Index (CSI)", axes[0, 0], True),
        ("pod", "Probability of Detection (POD)", axes[0, 1], True),
        ("far", "False Alarm Ratio (FAR)", axes[1, 0], False),
        ("fss", "Fractions Skill Score (FSS)", axes[1, 1], True),
    ]

    for metric_key, title, ax, higher_better in plots_cfg:
        ai_vals = [ai_metrics[metric_key][m] for m in leads]
        base_vals = [baseline_metrics[metric_key][m] for m in leads]

        ax.plot(
            x_vals,
            ai_vals,
            "o-",
            color="#1f77b4",
            linewidth=2.2,
            label="AI (UNet+ConvLSTM)",
        )
        ax.plot(
            x_vals,
            base_vals,
            "s--",
            color="#ff7f0e",
            linewidth=2.0,
            label="Advection Baseline",
        )

        direction = "(higher is better)" if higher_better else "(lower is better)"
        ax.set_title(f"{title}\n{direction}", fontsize=11)
        ax.set_xlabel("Lead Time (minutes)")
        ax.set_ylabel(metric_key.upper())
        ax.set_xticks(x_vals)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved skill scores plot to %s", output_path)


def run_evaluation(
    config_obj: SimpleNamespace | None = None,
    checkpoint_path: str | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    limit_events: int | None = None,
) -> dict[str, object]:
    """
    Execute full evaluation suite on test dataset.
    """
    c = config_obj or cfg
    ckpt = Path(checkpoint_path or (Path(c.paths.checkpoints) / "best_model.pt"))
    data_path = Path(data_dir or c.paths.synthetic_root)
    out_dir = Path(output_dir or c.paths.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}. Please train model first."
        )

    # Load test events
    files = sorted(data_path.glob("event_*.npy"))
    if not files:
        raise FileNotFoundError(f"No events found in {data_path}")
    if limit_events is not None:
        files = files[:limit_events]

    events = [np.load(f).astype(np.float32) for f in files]
    _, _, test_ds = make_splits(events, c)

    if len(test_ds) == 0:
        raise ValueError(
            "Test dataset is empty. Check split configuration or data size."
        )

    logger.info("Evaluating on %d test samples...", len(test_ds))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # 1. Load trained AI model
    device = torch.device("cpu")
    model = UNetConvLSTM.from_config(c).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # 2. Run inference for AI model and Advection Baseline
    baseline = AdvectionBaseline(c)

    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    ai_preds: list[np.ndarray] = []
    baseline_preds: list[np.ndarray] = []

    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_np = x_t.numpy()
            y_np = y_t.numpy()

            all_inputs.append(x_np)
            all_targets.append(y_np)

            # AI prediction
            pred_t = model(x_t.to(device)).cpu().numpy()
            ai_preds.append(pred_t)

            # Baseline prediction
            for i in range(x_np.shape[0]):
                base_forecast = baseline.predict(x_np[i])
                baseline_preds.append(base_forecast)

    y_true = np.concatenate(all_targets, axis=0)  # (N, T_out, H, W)
    y_ai = np.concatenate(ai_preds, axis=0)  # (N, T_out, H, W)
    y_base = np.stack(baseline_preds, axis=0)  # (N, T_out, H, W)

    leads = [int(m) for m in c.evaluation.lead_times_min]
    step_m = int(c.time.step_minutes)
    thr = float(c.evaluation.storm_threshold)

    ai_metrics = evaluate_forecasts(y_ai, y_true, leads, step_m, thr)
    base_metrics = evaluate_forecasts(y_base, y_true, leads, step_m, thr)

    metrics_payload = {
        "storm_threshold": thr,
        "lead_times_min": leads,
        "num_test_samples": len(test_ds),
        "ai_model": ai_metrics,
        "advection_baseline": base_metrics,
    }

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved verified evaluation metrics to %s", metrics_file)

    # Plot
    plot_file = out_dir / "skill_scores.png"
    plot_skill_curves(ai_metrics, base_metrics, leads, plot_file)

    # Print summary table
    print("\n" + "=" * 70)
    print("           VajraDrishti Nowcasting Evaluation Summary")
    print("=" * 70)
    print(
        f"{'Lead (min)':<12} | {'CSI (AI vs Base)':<18} | {'POD (AI vs Base)':<18} | {'FSS (AI vs Base)':<18}"
    )
    print("-" * 70)
    for lead_min in leads:
        l_str = str(lead_min)
        if l_str in ai_metrics["csi"]:
            csi_str = (
                f"{ai_metrics['csi'][l_str]:.3f} vs {base_metrics['csi'][l_str]:.3f}"
            )
            pod_str = (
                f"{ai_metrics['pod'][l_str]:.3f} vs {base_metrics['pod'][l_str]:.3f}"
            )
            fss_str = (
                f"{ai_metrics['fss'][l_str]:.3f} vs {base_metrics['fss'][l_str]:.3f}"
            )
            print(f"{lead_min:<12} | {csi_str:<18} | {pod_str:<18} | {fss_str:<18}")
    print("=" * 70 + "\n")

    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VajraDrishti models")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--limit-events", type=int, default=None, help="Limit test events for quick run"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config) if args.config else cfg
    run_evaluation(
        config_obj=config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit_events=args.limit_events,
    )
