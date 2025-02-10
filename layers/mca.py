import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from layers.squeeze_extract import SELayer
from layers.cross_attention import CrossAttention

class MCABlock(nn.Module):
    model_channels: int
    feature_channels: int
    num_heads: int
    reduction: int = 16
    num_groups: int = 16
    epsilon: float = 1e-5
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x, xf, emb):
        # Channel attention
        x = jnp.concatenate([x, xf], axis=-1)
        skip = x
        x = nn.GroupNorm(self.num_groups, epsilon=self.epsilon)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.model_channels + self.feature_channels, kernel_size=1)(x)
        x = SELayer(
            channels=self.model_channels + self.feature_channels,
            reduction=self.reduction,
        )(x) + skip
        x = nn.GroupNorm(self.num_groups, epsilon=self.epsilon)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.model_channels, kernel_size=1)(x)
        
        # Cross attention
        x = CrossAttention(
            channels=self.model_channels,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias
        )(x, emb)
        return x