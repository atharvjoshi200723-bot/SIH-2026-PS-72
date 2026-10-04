"""
losses.py — Custom loss functions for extreme weather nowcasting.

In precipitation and lightning nowcasting, severe storm cells are rare
(< 5% of all pixels). Standard Mean Squared Error (MSE) encourages the model
to predict smooth, washed-out fields that minimise average error by never
predicting extreme values.

To overcome this, VajraDrishti uses:
1. Weighted MSE: penalises errors on severe cells (reflectivity >= 0.5) by 10×.
2. Binary Focal Loss: addresses extreme foreground/background class imbalance
   on the severe cell mask (Lin et al., 2017).
3. Combined VajraLoss: 70% Weighted MSE + 30% Focal Loss.

All hyperparameters are read from configs/config.yaml.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error giving extra weight to severe storm cells.

    Loss = mean( w(y_true) * (y_pred - y_true)^2 )
    where w(y_true) = severe_weight if y_true >= severe_threshold else 1.0
    """

    def __init__(
        self, severe_threshold: float = 0.5, severe_weight: float = 10.0
    ) -> None:
        super().__init__()
        self.severe_threshold = severe_threshold
        self.severe_weight = severe_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            target >= self.severe_threshold,
            torch.tensor(self.severe_weight, device=target.device, dtype=target.dtype),
            torch.tensor(1.0, device=target.device, dtype=target.dtype),
        )
        squared_err = (pred - target) ** 2
        return torch.mean(weights * squared_err)


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for severe storm cell classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is model's predicted probability for the ground-truth class.
    Targets are binarised: y = 1 if target >= severe_threshold else 0.
    """

    def __init__(
        self,
        severe_threshold: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.severe_threshold = severe_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Binarise target at severe_threshold
        y = (target >= self.severe_threshold).float()

        # Clamp predictions to avoid log(0) and numerical instability
        p = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # p_t: probability of true class
        p_t = y * p + (1.0 - y) * (1.0 - p)

        # alpha_t: weighting factor for positive vs negative class
        alpha_t = y * self.alpha + (1.0 - y) * (1.0 - self.alpha)

        # Focal modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Focal loss per pixel
        loss = -alpha_t * modulating_factor * torch.log(p_t)
        return torch.mean(loss)


class VajraLoss(nn.Module):
    """
    Combined loss function for VajraDrishti.

    L_total = mse_weight * L_weighted_mse + focal_weight * L_focal
    """

    def __init__(
        self,
        severe_threshold: float = 0.5,
        severe_weight: float = 10.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mse_weight: float = 0.7,
        focal_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.weighted_mse = WeightedMSELoss(
            severe_threshold=severe_threshold,
            severe_weight=severe_weight,
        )
        self.focal = BinaryFocalLoss(
            severe_threshold=severe_threshold,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

    @classmethod
    def from_config(cls, cfg: SimpleNamespace) -> VajraLoss:
        """Instantiate loss from global config namespace."""
        loss_cfg = cfg.loss
        return cls(
            severe_threshold=float(loss_cfg.severe_threshold),
            severe_weight=float(loss_cfg.severe_weight),
            focal_alpha=float(loss_cfg.focal_alpha),
            focal_gamma=float(loss_cfg.focal_gamma),
            mse_weight=float(loss_cfg.mse_weight),
            focal_weight=float(loss_cfg.focal_weight),
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            total_loss: scalar torch.Tensor for .backward()
            metrics: dict with detached float values for logging
        """
        l_mse = self.weighted_mse(pred, target)
        l_focal = self.focal(pred, target)
        total = self.mse_weight * l_mse + self.focal_weight * l_focal
        metrics = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_weighted_mse": float(l_mse.detach().cpu().item()),
            "loss_focal": float(l_focal.detach().cpu().item()),
        }
        return total, metrics
