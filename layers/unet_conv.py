import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Union, Sequence, Optional, Type, Any, Callable, Tuple

class UnetConv(nn.Module):
    features: int
    kernel_size: Union[int, Sequence[int]]
    up: bool = False
    down: bool = False
    use_bias: bool = True
    resample_filter: jax.Array = jnp.outer(jnp.array([1., 1.]), jnp.array([1., 1.])).reshape(2, 2, 1, 1) 

    @nn.compact
    def __call__(self, x):
        if self.up:
            x = jax.lax.conv_general_dilated(
                lhs=x,
                rhs=jnp.tile(self.resample_filter, (1, 1, 1, x.shape[-1])),
                window_strides=(1, 1),
                padding=((1, 1), (1, 1)),
                lhs_dilation=(2, 2),
                feature_group_count=x.shape[-1],
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            )
        if self.down:
            x = jax.lax.conv_general_dilated(
                lhs=x,
                rhs=jnp.tile(self.resample_filter / 4., (1, 1, 1, x.shape[-1])),
                window_strides=(2, 2),
                padding=((0, 0), (0, 0)),
                feature_group_count=x.shape[-1],
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            )
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias
        )(x)
        return x
        
        
    