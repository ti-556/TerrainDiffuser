"""
Time-step sinusoidal embeddings (same formula as Transformer/Stable-Diffusion).
"""
from __future__ import annotations
import torch
from torch import nn
import math

__all__ = ["SinusoidalEmbedding"]


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int = 128, max_time: float = 10_000.0):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) scalar timesteps 0 … 1 (or any float scale).
        Returns:
            (B, d_model) sinusoid encodings.
        """
        # Ensure shape (B, 1)
        t = timesteps.float().unsqueeze(1)

        # Compute frequencies
        inv_freq = torch.exp(
            torch.arange(0, self.d_model, 2, device=t.device).float()
            * (-math.log(self.max_time) / self.d_model)
        )
        sinusoid = torch.zeros(t.size(0), self.d_model, device=t.device)
        sinusoid[:, 0::2] = torch.sin(t * inv_freq)
        sinusoid[:, 1::2] = torch.cos(t * inv_freq)
        return sinusoid