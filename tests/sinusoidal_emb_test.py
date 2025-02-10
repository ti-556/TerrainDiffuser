import jax
import jax.numpy as jnp
import timeit

from layers.sinusoidal_emb import SinusoidalEmbedding

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, shape=(8,))
    
    model = SinusoidalEmbedding(
        d_model=128,
        max_time=10000
    )
    
    params = model.init(key, t)
    
    def forward_fn(params, t):
        return model.apply(params, t)
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, t).block_until_ready(), number=100)
    print(f"Non-jit forward pass time: {execution_time / 100:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, t).block_until_ready()
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, t).block_until_ready(), number=100)
    print(f"Jit forward pass time: {jit_execution_time / 100:.6f} seconds per run")
    
    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")

if __name__ == "__main__":
    main()