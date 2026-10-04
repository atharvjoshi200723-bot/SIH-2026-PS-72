"""
unet_convlstm.py — Spatiotemporal U-Net with ConvLSTM bottleneck.

Architecture:
1. 2D Spatial Encoder (per-frame):
   Hierarchical convolutions extract spatial representations at multiple
   scales (128x128 -> 64x64 -> 32x32) with channels [16, 32, 64].
2. ConvLSTM Bottleneck:
   Captures storm cell advection, rotation, and intensity evolution over
   the past input frames (default: 4 frames = 1 hour).
3. Recurrent Forecaster + U-Net Decoder:
   Unrolls the ConvLSTM state over the forecast horizon (default: 12 frames = 3 hours).
   Skip connections from the encoder preserve high-resolution spatial edges.
4. Output Head:
   1x1 convolution with Sigmoid activation outputs normalised reflectivity in [0, 1].

References:
- Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for
  Precipitation Nowcasting", NeurIPS 2015.
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """
    Standard 2D Convolutional LSTM cell (Shi et al., 2015).

    Replaces matrix multiplications in standard LSTM with 2D convolutions,
    preserving spatial topologies while tracking temporal transitions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # A single convolution computes all 4 gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, c_tilde, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_tilde = torch.tanh(c_tilde)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * c_tilde
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self, batch_size: int, spatial_shape: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = spatial_shape
        h_0 = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        return h_0, c_0


class ConvBlock(nn.Module):
    """Two 2D convolutions with BatchNorm, LeakyReLU, and optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetConvLSTM(nn.Module):
    """
    Spatiotemporal Nowcasting Model combining U-Net and ConvLSTM.

    Input shape:  (B, input_frames, H, W) or (B, input_frames, 1, H, W)
    Output shape: (B, output_frames, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: list[int] | None = None,
        convlstm_hidden: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_frames: int = 12,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [16, 32, 64]

        self.output_frames = output_frames
        c1, c2, c3 = encoder_channels

        # Spatial Encoder
        self.enc1 = ConvBlock(in_channels, c1, kernel_size, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc2 = ConvBlock(c1, c2, kernel_size, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        self.enc3 = ConvBlock(c2, c3, kernel_size, dropout=dropout)  # 32x32

        # ConvLSTM Bottleneck
        self.convlstm = ConvLSTMCell(
            input_dim=c3,
            hidden_dim=convlstm_hidden,
            kernel_size=kernel_size,
        )

        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(convlstm_hidden, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, kernel_size, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, kernel_size, dropout=0.0)

        # Output projection (normalized reflectivity in [0, 1])
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_config(cls, cfg: SimpleNamespace) -> UNetConvLSTM:
        """Instantiate UNetConvLSTM using values from config.yaml."""
        model_cfg = cfg.model
        time_cfg = cfg.time
        channels = [int(c) for c in model_cfg.encoder_channels]
        return cls(
            in_channels=1,
            out_channels=1,
            encoder_channels=channels,
            convlstm_hidden=int(model_cfg.convlstm_hidden),
            kernel_size=int(model_cfg.kernel_size),
            dropout=float(model_cfg.dropout),
            output_frames=int(time_cfg.output_frames),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, T_in, H, W) or (B, T_in, 1, H, W)

        Returns:
            Tensor of shape (B, T_out, H, W)
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)  # (B, T_in, 1, H, W)

        B, T_in, _C, H, W = x.shape
        device = x.device

        # Bottleneck spatial resolution is H // 4, W // 4 (e.g., 32x32)
        spatial_bottleneck = (H // 4, W // 4)
        h, c = self.convlstm.init_hidden(B, spatial_bottleneck, device=device)

        # Skip connections from the most recent input frame
        last_s1: torch.Tensor | None = None
        last_s2: torch.Tensor | None = None

        # 1. Encode past frames sequentially through ConvLSTM
        for t in range(T_in):
            xt = x[:, t]  # (B, 1, H, W)
            s1 = self.enc1(xt)  # (B, c1, 128, 128)
            p1 = self.pool1(s1)  # (B, c1, 64, 64)
            s2 = self.enc2(p1)  # (B, c2, 64, 64)
            p2 = self.pool2(s2)  # (B, c2, 32, 32)
            s3 = self.enc3(p2)  # (B, c3, 32, 32)

            h, c = self.convlstm(s3, (h, c))

            if t == T_in - 1:
                last_s1 = s1
                last_s2 = s2

        assert last_s1 is not None and last_s2 is not None

        # 2. Recurrently forecast future frames
        predictions: list[torch.Tensor] = []
        zero_input = torch.zeros(
            B, self.enc3.block[0].out_channels, *spatial_bottleneck, device=device
        )

        for _ in range(self.output_frames):
            h, c = self.convlstm(zero_input, (h, c))

            # Decode hidden state with skip connections
            u2 = self.up2(h)
            d2 = self.dec2(torch.cat([u2, last_s2], dim=1))

            u1 = self.up1(d2)
            d1 = self.dec1(torch.cat([u1, last_s1], dim=1))

            out = self.final_conv(d1)  # (B, 1, H, W)
            predictions.append(out.squeeze(1))  # (B, H, W)

        return torch.stack(predictions, dim=1)  # (B, T_out, H, W)
