"""
UNet residual block with optional up/down sampling and attention.
"""
from __future__ import annotations
import torch
from torch import nn
from terrain_diffuser.models.diffusion.terrain_unet.utils import Conv2d
from terrain_diffuser.models.diffusion.terrain_unet.attention import AttentionBlock

__all__ = ["UnetResBlock"]


class UnetResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngroups: int,
        *,
        t_emb_dim: int,
        c_emb_dim: int,
        dropout: float = 0.1,
        feature_channels: int | None = None,
        down: bool = False,
        up: bool = False,
        attn: bool = False,
        num_heads: int = 1,
        resample_filter=(1, 1),
    ):
        super().__init__()
        if up and down:
            raise ValueError("Block cannot upsample *and* downsample")

        self.up, self.down = up, down
        self.in_channels, self.out_channels = in_channels, out_channels

        self.norm1 = nn.GroupNorm(ngroups, in_channels)
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
        )
        self.norm2 = nn.GroupNorm(ngroups, out_channels)
        self.act = nn.SiLU()

        self.emb_proj = nn.Linear(t_emb_dim, out_channels * 2)
        self.dropout = dropout
        self.conv2 = Conv2d(out_channels, out_channels, kernel=3)

        self.skip = (
            Conv2d(
                in_channels,
                out_channels,
                kernel=1,
                up=up,
                down=down,
                resample_filter=resample_filter,
            )
            if (up or down or (in_channels != out_channels))
            else None
        )

        self.attn_block = (
            AttentionBlock(
                out_channels,
                num_heads,
                ngroups,
                feature_channels,
                c_emb_dim,
            )
            if attn
            else None
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        content_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.conv2(
            torch.nn.functional.dropout(
                self.act((scale + 1) * h + shift), p=self.dropout, training=self.training
            )
        )

        out = h + (self.skip(x) if self.skip is not None else x)
        if self.attn_block is not None:
            out = self.attn_block(out, content_feat, text_emb)
        return out
