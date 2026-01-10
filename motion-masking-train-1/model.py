from __future__ import annotations

import torch
import torch.nn as nn


class Tiny3DMotionNet(nn.Module):
    """Minimal 3D conv network.

    Input:  (B, T, 1, H, W)
    Output: (B, 2, H, W) where channels are (vx, vy)

    Implementation detail: we transpose to (B, 1, T, H, W) for Conv3d.
    We keep temporal size and then collapse T via a final conv that spans T.
    """

    def __init__(self, x_frames: int = 4, base_channels: int = 16):
        super().__init__()
        self.t = x_frames + 1

        c = base_channels
        self.backbone = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

        # Collapse temporal dimension to 1 using a conv spanning full T.
        self.head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=(self.t, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if c != 1:
            raise ValueError(f"Expected C=1, got C={c}")
        if t != self.t:
            raise ValueError(f"Expected T={self.t} (x_frames+1), got T={t}")

        x = x.transpose(1, 2)  # (B, 1, T, H, W)
        x = self.backbone(x)
        x = self.head(x)  # (B, 2, 1, H, W)
        x = x.squeeze(2)  # (B, 2, H, W)
        return x
