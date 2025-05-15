import torch
from typing import Callable

# individual loss functions ---------------------------------------------------
def epsilon_loss(pred: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return (pred - eps).square().mean()

def flow_matching_loss(pred: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
    return (pred - vf).square().mean()

def x_prediction_loss(pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return (pred - x0).square().mean()

# simple helper ----------------------------------------------------------------
def make_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name in ("eps", "epsilon"):
        return epsilon_loss
    if name in ("flow", "fm"):
        return flow_matching_loss
    if name in ("xpred", "x_prediction"):
        return x_prediction_loss
    raise ValueError(f"Unknown loss type: {name}")