from layers.cross_attention import CrossAttention
import jax
import jax.numpy as jnp
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x = jax.random.normal(subkey1, (8, 32, 32, 64))
    emb = jax.random.normal(subkey2, (8, 77, 128))
    
    model = CrossAttention(
        channels=64,
        num_heads=4,
        num_groups=16,
        epsilon=1e-5,
        use_bias=False
    )
    params = model.init(key, x, emb)
    
    def forward_fn(params, x, emb):
        return model.apply(params, x, emb)
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, x, emb).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 100:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, x, emb).block_until_ready()
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, x, emb).block_until_ready(), number=10)
    print(f"Jit forward pass time: {jit_execution_time / 100:.6f} seconds per run")
    
    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")

if __name__ == "__main__":
    main()