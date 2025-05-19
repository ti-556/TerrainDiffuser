"""
Light-weight utilities: weight initialisation + a fused-resample Conv2d.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

__all__ = ["weight_init", "Conv2d"]

# ------------------------------------------------------------
# Weight initialisation helpers
# ------------------------------------------------------------
def weight_init(shape, mode: str, fan_in: int, fan_out: int):
    """
    Returns a Tensor with the requested shape/population.
    `mode` in {"xavier_uniform","xavier_normal","kaiming_uniform","kaiming_normal"}.
    """
    if mode not in {
        "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
    }:
        raise ValueError(f"Invalid init mode: {mode}")

    if mode == "xavier_uniform":
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        return torch.empty(shape).uniform_(-bound, bound)

    if mode == "xavier_normal":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.empty(shape).normal_(0.0, std)

    if mode == "kaiming_uniform":
        bound = math.sqrt(3.0 / fan_in)
        return torch.empty(shape).uniform_(-bound, bound)

    # kaiming_normal
    std = math.sqrt(1.0 / fan_in)
    return torch.empty(shape).normal_(0.0, std)


# ------------------------------------------------------------
# A StyleGAN-like convolution wrapper that folds (optional) up/
# down-sampling and custom kernel padding into one module.
# ------------------------------------------------------------
class Conv2d(torch.nn.Module):
    """
    Conv2d with optional 2× up/down resampling using a
    separable [1 1]^T[1 1] kernel (bilinear by default).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | None = 3,
        *,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: list[int] = (1, 1),
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        if up and down:
            raise ValueError("Conv2d cannot both upsample and downsample")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample

        # main conv kernel (may be None for pure resample)
        if kernel:
            w = weight_init(
                [out_channels, in_channels, kernel, kernel],
                mode=init_mode,
                fan_in=in_channels * kernel * kernel,
                fan_out=out_channels * kernel * kernel,
            )
            self.weight = torch.nn.Parameter(w * init_weight)
            self.bias = (
                torch.nn.Parameter(
                    weight_init([out_channels], init_mode, in_channels, out_channels)
                    * init_bias
                )
                if bias
                else None
            )
        else:
            self.weight, self.bias = None, None

        # build the (1D) resample filter → make 2D separable
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = (f.ger(f) / f.sum().square()).unsqueeze(0).unsqueeze(1)
        self.register_buffer("resample_filter", f if (up or down) else None)

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype) if self.weight is not None else None
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(dtype=x.dtype)
            if self.resample_filter is not None
            else None
        )

        w_pad = (w.shape[-1] // 2) if w is not None else 0
        f_pad = ((f.shape[-1] - 1) // 2) if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            # upsample → conv
            x = F.conv_transpose2d(
                x,
                f.mul(4).repeat(self.in_channels, 1, 1, 1),
                padding=max(f_pad - w_pad, 0),
                stride=2,
                groups=self.in_channels,
            )
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            # conv → downsample
            x = F.conv2d(x, w, padding=w_pad + f_pad)
            x = F.conv2d(
                x,
                f.repeat(self.out_channels, 1, 1, 1),
                stride=2,
                groups=self.out_channels,
            )
        else:
            if self.up:
                x = F.conv_transpose2d(
                    x,
                    f.mul(4).repeat(self.in_channels, 1, 1, 1),
                    padding=f_pad,
                    stride=2,
                    groups=self.in_channels,
                )
            if self.down:
                x = F.conv2d(
                    x,
                    f.repeat(self.in_channels, 1, 1, 1),
                    padding=f_pad,
                    stride=2,
                    groups=self.in_channels,
                )
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)

        if b is not None:
            x = x.add_(b.view(1, -1, 1, 1))
        return x