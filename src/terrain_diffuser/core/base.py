from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


class ImageEncoder(nn.Module, ABC):
    @abstractmethod
    def extract_features_at_res(
        self,
        images: torch.Tensor,           # (B, 3, H0, W0)
        resolutions: Sequence[int],     # e.g. (16, 8)
        mode: str = "all"               # 'all' | 'topk' | 'bottomk' | 'random'
    ) -> dict[str, torch.Tensor]:
        pass
    
class TextEncoder(nn.Module, ABC):
    @abstractmethod
    def tokenize(self, texts: list[str]) -> dict[int, torch.Tensor]:
        pass
    
    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        pass