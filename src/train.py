"""
train.py — Training pipeline for VajraDrishti UNetConvLSTM.

Loads storm events (synthetic or SEVIR), splits them by event into
train / val / test sets, and trains the spatiotemporal nowcasting model
using the custom VajraLoss (Weighted MSE + Binary Focal Loss).

Usage:
    python -m src.train
    python -m src.train --max-epochs 5 --quick
    python -m src.train --device cpu --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import cfg, load_config
from src.data.dataset import make_splits
from src.models.losses import VajraLoss
from src.models.unet_convlstm import UNetConvLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """Resolve compute device based on configuration and availability."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def load_events(data_dir: str | Path, limit: int | None = None) -> list[np.ndarray]:
    """Load pre-generated .npy storm events from data directory."""
    path = Path(data_dir)
    files = sorted(path.glob("event_*.npy"))
    if not files:
        raise FileNotFoundError(f"No event_*.npy files found in {path.resolve()}")
    if limit is not None:
        files = files[:limit]

    logger.info("Loading %d events from %s...", len(files), path)
    events = [np.load(f).astype(np.float32) for f in files]
    logger.info("Loaded %d events. Event shape: %s", len(events), events[0].shape)
    return events


def train_one_epoch(
    model: UNetConvLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: VajraLoss,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, dict[str, float]]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    agg_metrics = {"loss_weighted_mse": 0.0, "loss_focal": 0.0}
    num_batches = len(loader)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss, metrics = loss_fn(pred, y)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        for k in agg_metrics:
            agg_metrics[k] += metrics.get(k, 0.0)

    avg_loss = total_loss / max(num_batches, 1)
    for k in agg_metrics:
        agg_metrics[k] /= max(num_batches, 1)

    return avg_loss, agg_metrics


@torch.no_grad()
def evaluate_val(
    model: UNetConvLSTM,
    loader: DataLoader,
    loss_fn: VajraLoss,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate model on validation split."""
    model.eval()
    total_loss = 0.0
    agg_metrics = {"loss_weighted_mse": 0.0, "loss_focal": 0.0}
    num_batches = len(loader)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss, metrics = loss_fn(pred, y)

        total_loss += loss.item()
        for k in agg_metrics:
            agg_metrics[k] += metrics.get(k, 0.0)

    avg_loss = total_loss / max(num_batches, 1)
    for k in agg_metrics:
        agg_metrics[k] /= max(num_batches, 1)

    return avg_loss, agg_metrics


def train_pipeline(
    config_obj: SimpleNamespace | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    data_dir: str | None = None,
    limit_events: int | None = None,
    save_dir: str | None = None,
) -> dict[str, object]:
    """
    Main training execution function.
    Can be invoked from CLI or imported into tests.
    """
    c = config_obj or cfg

    epochs = max_epochs if max_epochs is not None else int(c.training.max_epochs)
    bs = batch_size if batch_size is not None else int(c.training.batch_size)
    dev_str = device_name if device_name is not None else str(c.training.device)
    device = get_device(dev_str)

    data_path = Path(data_dir or c.paths.synthetic_root)
    checkpoint_dir = Path(save_dir or c.paths.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training pipeline on device: %s", device)
    events = load_events(data_path, limit=limit_events)
    train_ds, val_ds, test_ds = make_splits(events, c)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    logger.info(
        "Samples: train=%d, val=%d, test=%d | Batch size: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        bs,
    )

    model = UNetConvLSTM.from_config(c).to(device)
    loss_fn = VajraLoss.from_config(c).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(c.training.learning_rate),
        weight_decay=float(c.training.weight_decay),
    )

    best_val_loss = float("inf")
    best_epoch = -1
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    t0_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=float(c.training.grad_clip),
        )
        val_loss, _val_metrics = evaluate_val(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        elapsed = time.time() - t_epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch [%2d/%2d] (%.1fs) | Train Loss: %.4f (MSE: %.4f, Focal: %.4f) | Val Loss: %.4f",
            epoch,
            epochs,
            elapsed,
            train_loss,
            train_metrics["loss_weighted_mse"],
            train_metrics["loss_focal"],
            val_loss,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": {
                        "encoder_channels": [
                            int(ch) for ch in c.model.encoder_channels
                        ],
                        "convlstm_hidden": int(c.model.convlstm_hidden),
                        "output_frames": int(c.time.output_frames),
                    },
                },
                best_path,
            )
            logger.info("  --> Saved new best checkpoint to %s", best_path)

        # Periodic checkpoint
        if epoch % int(c.training.checkpoint_every) == 0:
            chk_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                chk_path,
            )

    total_time = time.time() - t0_start
    logger.info(
        "Training completed in %.1fs. Best Val Loss: %.4f at Epoch %d",
        total_time,
        best_val_loss,
        best_epoch,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_dir / "best_model.pt"),
        "history": history,
        "total_time_seconds": total_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VajraDrishti UNetConvLSTM Model"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Override max epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cpu, mps, cuda, auto)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory"
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Limit number of events for quick runs",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick run (3 epochs on 30 events)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config) if args.config else cfg

    max_epochs = 3 if args.quick and args.max_epochs is None else args.max_epochs
    limit_events = 30 if args.quick and args.limit_events is None else args.limit_events

    train_pipeline(
        config_obj=config,
        max_epochs=max_epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        data_dir=args.data_dir,
        limit_events=limit_events,
    )
