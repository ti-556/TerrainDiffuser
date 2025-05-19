import math, torch

# ───────────────────────────────────────────────────────────── #
class SD3Weighting:
    """γ(t) used in Stable Diffusion 3 time-sampling."""
    def __init__(self, clip: tuple[float, float] = (0.001, 0.999)):
        self.lo, self.hi = clip

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # here t already comes from a logistic mapping in your code
        return t.clamp_(self.lo, self.hi)

class CosineWeighting:
    """γ(t) = cos(π t / 2)."""
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * math.pi * t)
# ───────────────────────────────────────────────────────────── #

def make_weighting(name: str):
    name = name.lower()
    if name == "sd3":
        return SD3Weighting()
    if name == "cosine":
        return CosineWeighting()
    raise ValueError(f"Unknown weighting schedule: {name}")