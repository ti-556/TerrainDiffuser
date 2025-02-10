import jax
import jax.numpy as jnp
import flax.linen as nn

from layers.mca import MCABlock
from layers.unet_conv import UnetConv

class UnetBlock(nn.Module):
    model_channels: int
    p_dropout: int
    feature_channels: int = None
    down: bool = False
    up: bool = False
    attn: bool = False
    num_heads: int = 1,
    num_groups: int = 16
    epsilon: float = 1e-5
    use_bias: bool = False
    reduction: int = 16
    
    @nn.compact
    def __call__(self, x, time_emb, text_emb, feat_maps=None, train: bool=True):
        skip = x
        x = nn.GroupNorm(num_groups=self.num_groups, epsilon=self.epsilon)(x)
        x = nn.silu(x)
        x = UnetConv(
            features=self.model_channels, kernel_size=3, up=self.up, down=self.down, use_bias=True
        )(x)
        x = nn.GroupNorm(num_groups=self.num_groups, epsilon=self.epsilon)(x)
        scale, shift = jnp.split(nn.Dense(self.model_channels * 2)(time_emb).reshape(
            time_emb.shape[0], 1, 1, 2 * self.model_channels
        ), 2, axis=-1)
        x = nn.silu(x * (scale + 1) + shift)
        x = nn.Dropout(rate=self.p_dropout, deterministic=not train)(x)
        x = nn.Conv(features=self.model_channels, kernel_size=3)(x)
        if self.up or self.down or skip.shape[-1] != self.model_channels:
            x += UnetConv(
                features=self.model_channels, kernel_size=1, up=self.up,down=self.down, use_bias=True
            )(skip)
        else:
            x += skip
        
        if self.attn:
            x = MCABlock(
                model_channels=self.model_channels,
                feature_channels=self.feature_channels,
                num_heads=self.num_heads,
                reduction=self.reduction,
                num_groups=self.num_groups,
                epsilon=self.epsilon,
                use_bias=self.use_bias,
            )(x, feat_maps, text_emb)
        return x