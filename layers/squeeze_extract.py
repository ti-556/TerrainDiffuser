import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

class SELayer(nn.Module):
    channels: int
    reduction: int = 16
    
    @nn.compact
    def __call__(self, x):
        b, _, _, c = x.shape
        scale = jnp.mean(x, axis=[1, 2]).reshape(b, c)
        scale = nn.Dense(features=self.channels//self.reduction, use_bias=False)(scale)
        scale = nn.silu(scale)
        scale = nn.Dense(features=self.channels, use_bias=False)(scale)
        scale = nn.sigmoid(scale).reshape(b, 1, 1, c)
        return x * scale