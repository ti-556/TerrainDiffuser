import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

class CrossAttention(nn.Module):
    channels: int
    num_heads: int
    num_groups: int = 16
    epsilon: float = 1e-5
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x, emb):
        head_dim = self.channels // self.num_heads
        skip = x
        x = nn.GroupNorm(num_groups=self.num_groups, epsilon=self.epsilon)(x)
        q = nn.Conv(features=self.channels, kernel_size=1, use_bias=self.use_bias)(x)
        # The kv will be (bsz, 77, emb_dim) -> (bsz, 77, channels)
        kv = nn.Dense(features=self.channels * 2, use_bias=self.use_bias)(emb)
        kv = rearrange(kv, 'b s (n c) -> (b n) c s', n=self.num_heads)
        q = rearrange(q, 'b h w (n c) -> (b n) c (h w)', n=self.num_heads)
        k, v = jnp.split(kv, 2, axis=1)
        scores = jnp.einsum('bcq, bck -> bqk', q, k) * (head_dim ** -0.5)
        scores = nn.softmax(scores, axis=-1)
        x = jnp.einsum('bqk, bck -> bqc', scores, v)
        x = rearrange(x, '(b n) (h w) c -> b h w (n c)', n=self.num_heads, h=skip.shape[1], w=skip.shape[2])
        x = nn.Conv(features=self.channels, use_bias=True, kernel_size=1)(x)
        return x + skip