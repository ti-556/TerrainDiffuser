"""
All attention / SE helper blocks reused by UNet.
"""
from __future__ import annotations
import torch
from torch import nn, einsum
from einops import rearrange
from terrain_diffuser.models.diffusion.terrain_unet.utils import Conv2d

__all__ = [
    "SELayer",
    "ChannelAttnBlock",
    "MCABlock",
    "AttentionBlock",
]

# ------------------------  Squeeze-and-Excite  -------------------------------
class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -------------------------  Channel mixer  ----------------------------------
class ChannelAttnBlock(nn.Module):
    """
    Concat content features to x, mix with 1×1 conv, then SE attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        reduction: int = 16,
        groups: int = 32,
        non_linearity: str = "silu",
    ):
        super().__init__()
        if (in_channels) % groups:
            raise ValueError(
                f"num_channels={in_channels} is not divisible by "
                f"num_groups={groups}.  Choose different `ngroups` or "
                f"adjust feature_channels for that resolution.`"
        )
        self.se = SELayer(in_channels, reduction)
        self.act = nn.SiLU() if non_linearity == "silu" else nn.Swish()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.down = nn.Conv2d(in_channels, out_channels, 1)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, content: torch.Tensor | None) -> torch.Tensor:
        if content is not None:
            x = torch.cat([x, content], dim=1)

        h = self.conv1(self.act(self.norm1(x)))
        h = self.se(h) + x                       # SE + residual
        h = self.down(self.act(self.norm2(h)))
        return h


# ----------------------  Multi-cond attention  ------------------------------
class MCABlock(nn.Module):
    """
    Spatial queries (from image) attend to **text** keys/values, *after*
    channel-mixing with content features.
    """

    def __init__(
        self,
        emb_dim: int,
        in_channels: int,
        feature_channels: int | None,
        num_heads: int = 1,
        reduction: int = 16,
        groups: int = 32,
    ):
        super().__init__()
        self.channel_attn = ChannelAttnBlock(
            in_channels + (feature_channels or 0),
            in_channels,
            reduction=reduction,
            groups=groups,
        )
        self.num_heads = num_heads
        self.q_proj = Conv2d(in_channels, in_channels, 1)
        self.kv_proj = nn.Linear(emb_dim, in_channels * 2)
        self.out_proj = Conv2d(in_channels, in_channels, 1)

        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        content: torch.Tensor | None,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.channel_attn(x, content)

        q = self.q_proj(x)                          # (B,C,H,W)
        kv = self.kv_proj(text_emb)                 # (B,S,2C)
        k, v = rearrange(kv, "b s (n c) -> (b n) c s", n=self.num_heads).chunk(2, dim=1)
        q = rearrange(q, "b (n c) h w -> (b n) c (h w)", n=self.num_heads)

        attn = (einsum("b c q, b c k -> b q k", q, k) * (self.num_heads**-0.5)).softmax(-1)
        out = einsum("b q k, b c k -> b q c", attn, v)
        out = rearrange(out, "(b n) q c -> b (n c) q", n=self.num_heads)
        out = out.view(x.shape)                     # (B,C,H,W)
        return self.out_proj(out) + x


# ----------------------  Self-attention + MCA  ------------------------------
class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        groups: int,
        feature_ch: int | None,
        text_dim: int,
        reduction: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")

        self.norm = nn.GroupNorm(groups, in_channels)
        self.qkv = Conv2d(in_channels, in_channels * 3, 1)
        self.proj = Conv2d(in_channels, in_channels, 1)

        self.mca = MCABlock(
            text_dim,
            in_channels,
            feature_ch,
            num_heads=num_heads,
            reduction=reduction,
            groups=groups,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        content: torch.Tensor | None,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        # ---- self-attention on (H×W) tokens --------------------------
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = rearrange(qkv, "b (n c) h w -> (b n) c (h w)", n=self.num_heads).chunk(
            3, dim=1
        )
        attn = (einsum("b c q, b c k -> b q k", q, k) * (self.num_heads**-0.5)).softmax(
            -1
        )
        out = einsum("b q k, b c k -> b q c", attn, v)
        out = rearrange(out, "(b n) q c -> b (n c) q", n=self.num_heads).view(B, C, H, W)
        x = self.proj(out) + x

        # ---- cross-attention (image → text) --------------------------
        return self.mca(x, content, text_emb)
