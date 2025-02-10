import jax
import jax.numpy as jnp
import flax.linen as nn

class SinusoidalEmbedding(nn.Module):
    d_model: int
    max_time: float
    
    def __call__(self, t: jax.Array):
        t = jnp.expand_dims(t, axis=-1)
        freqs = jnp.arange(0, self.d_model, step=2)
        freqs = jnp.exp(-freqs/self.d_model * jnp.log(self.max_time))
        embeddings = jnp.zeros((t.shape[0], self.d_model))
        embeddings = embeddings.at[:, 0::2].set(jnp.sin(t * freqs))
        embeddings = embeddings.at[:, 1::2].set(jnp.cos(t * freqs))
        return embeddings