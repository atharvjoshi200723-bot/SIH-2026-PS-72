"""
test_phase3.py — Verification tests for Phase 3 (Model & Loss functions).

Tests:
1. ConvLSTMCell shape and state updates
2. UNetConvLSTM initialization from config
3. Forward pass input/output contract: (B, 4, 128, 128) -> (B, 12, 128, 128)
4. Model output values bounded in [0, 1]
5. Weighted MSE weights severe cells (>= 0.5) more than low-reflectivity cells
6. Binary Focal Loss computes expected scalar loss
7. VajraLoss backward pass computes valid gradients
"""

import pytest
import torch

from src.config import load_config
from src.models.losses import BinaryFocalLoss, VajraLoss, WeightedMSELoss
from src.models.unet_convlstm import ConvLSTMCell, UNetConvLSTM


@pytest.fixture
def cfg():
    return load_config()


def test_convlstm_cell_forward():
    cell = ConvLSTMCell(input_dim=16, hidden_dim=32, kernel_size=3)
    B, H, W = 2, 16, 16
    x = torch.randn(B, 16, H, W)
    h0, c0 = cell.init_hidden(B, (H, W), device=torch.device("cpu"))

    h1, c1 = cell(x, (h0, c0))
    assert h1.shape == (B, 32, H, W)
    assert c1.shape == (B, 32, H, W)


def test_unet_convlstm_init(cfg):
    model = UNetConvLSTM.from_config(cfg)
    assert isinstance(model, UNetConvLSTM)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Architecture is designed to be lightweight (< 1M parameters) for quick CPU execution
    assert 100_000 < param_count < 1_000_000


def test_unet_convlstm_forward(cfg):
    model = UNetConvLSTM.from_config(cfg)
    model.eval()

    B = 2
    T_in = cfg.time.input_frames  # 4
    T_out = cfg.time.output_frames  # 12
    H = cfg.grid.height  # 128
    W = cfg.grid.width  # 128

    x = torch.rand(B, T_in, H, W)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, T_out, H, W)
    # Check outputs are in [0, 1] range due to final sigmoid
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_weighted_mse_severe_penalization():
    loss_fn = WeightedMSELoss(severe_threshold=0.5, severe_weight=10.0)

    # Prediction is zero everywhere
    pred = torch.zeros(1, 1, 10, 10)

    # Target 1: moderate reflectivity (0.3)
    target_low = torch.full((1, 1, 10, 10), 0.3)
    l_low = loss_fn(pred, target_low).item()

    # Target 2: severe reflectivity (0.8) with same error difference (0.5 vs 0.3)
    # If severe_weight was 1.0, err^2 would be (0.8)^2 / (0.3)^2 = 7.1x
    # With 10x severe_weight, ratio should be ~ 71x
    target_high = torch.full((1, 1, 10, 10), 0.8)
    l_high = loss_fn(pred, target_high).item()

    assert l_high > 10.0 * l_low


def test_binary_focal_loss():
    loss_fn = BinaryFocalLoss(severe_threshold=0.5, alpha=0.25, gamma=2.0)
    target = torch.zeros(1, 1, 10, 10)
    target[:, :, :2, :2] = 0.9  # Severe storm in small corner

    pred_good = target.clone()
    pred_bad = 1.0 - target

    l_good = loss_fn(pred_good, target).item()
    l_bad = loss_fn(pred_bad, target).item()

    assert l_bad > l_good
    assert l_good >= 0.0


def test_vajraloss_gradient_flow(cfg):
    model = UNetConvLSTM.from_config(cfg)
    loss_fn = VajraLoss.from_config(cfg)

    x = torch.rand(1, cfg.time.input_frames, cfg.grid.height, cfg.grid.width)
    y = torch.rand(1, cfg.time.output_frames, cfg.grid.height, cfg.grid.width)

    model.train()
    pred = model(x)
    loss, metrics = loss_fn(pred, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert "loss_total" in metrics
    assert "loss_weighted_mse" in metrics
    assert "loss_focal" in metrics

    loss.backward()

    # Check that gradients exist and are finite
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert not torch.isnan(p.grad).any()
    assert has_grad
