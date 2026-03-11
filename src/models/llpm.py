"""Learnable Local Preprocessing Module (LLPM).

A lightweight module (~31K params) placed BEFORE SAM's frozen ViT-H encoder.
It enhances boundary/edge features at full 1024x1024 resolution via two
branches: an edge detection branch and an enhancement branch, fused with
a learnable residual connection.

This is a novel contribution — related work (SAM-Adapter, Self-Prompt SAM)
adds adapters *inside* the encoder. LLPM operates *before* the encoder.
"""

import torch
import torch.nn as nn


class EdgeBranch(nn.Module):
    """Produces a single-channel edge attention map via sigmoid.

    Architecture: Conv(3->C,3) -> BN -> ReLU -> Conv(C->C,3) -> BN -> ReLU
                  -> Conv(C->1,1) -> Sigmoid
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancementBranch(nn.Module):
    """Produces a 3-channel enhancement residual using grouped convolutions.

    Architecture: Conv(3->C,3) -> BN -> ReLU
                  -> GroupConv(C->C,3,g=4) -> BN -> ReLU
                  -> GroupConv(C->C,3,g=4) -> BN -> ReLU
                  -> Conv(C->3,1)
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLPM(nn.Module):
    """Learnable Local Preprocessing Module.

    Fuses edge attention and enhancement via a residual connection:
        output = x + alpha * (enhancement * edge_map)

    Args:
        edge_channels: Hidden channels in EdgeBranch (default 32).
        enhance_channels: Hidden channels in EnhancementBranch (default 64).
        alpha_init: Initial value for the learnable fusion weight (default 0.1).
        mode: Operating mode for ablation studies.
            - "full": Default, both branches with learnable alpha.
            - "edge_only": Only edge branch (enhancement is identity).
            - "enhance_only": Only enhancement branch (no edge gating).
            - "no_gate": Both branches but alpha fixed to 1.0.
    """

    VALID_MODES = ("full", "edge_only", "enhance_only", "no_gate")

    def __init__(
        self,
        edge_channels: int = 32,
        enhance_channels: int = 64,
        alpha_init: float = 0.1,
        mode: str = "full",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")
        self.mode = mode
        self.edge_branch = EdgeBranch(in_channels=3, mid_channels=edge_channels)
        self.enhance_branch = EnhancementBranch(in_channels=3, mid_channels=enhance_channels)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W), float32, range [0, 255].

        Returns:
            Enhanced image tensor (B, 3, H, W), same range.
        """
        if self.mode == "edge_only":
            edge_map = self.edge_branch(x)
            return x + self.alpha * (x * edge_map)
        elif self.mode == "enhance_only":
            enhancement = self.enhance_branch(x)
            return x + self.alpha * enhancement
        elif self.mode == "no_gate":
            edge_map = self.edge_branch(x)
            enhancement = self.enhance_branch(x)
            return x + 1.0 * (enhancement * edge_map)
        else:  # "full"
            edge_map = self.edge_branch(x)
            enhancement = self.enhance_branch(x)
            return x + self.alpha * (enhancement * edge_map)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
