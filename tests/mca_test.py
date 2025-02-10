from layers.mca import MCABlock
import jax
import jax.numpy as jnp
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    x = jax.random.normal(subkey1, (8, 32, 32, 64))
    emb = jax.random.normal(subkey2, (8, 77, 128))
    xf = jax.random.normal(subkey3, (8, 32, 32, 32))
    
    model = MCABlock(
        model_channels=64,
        feature_channels=32,
        num_heads=4,
        num_groups=16,
        epsilon=1e-5,
        use_bias=False
    )
    params = model.init(key, x, xf, emb)
    
    def forward_fn(params, x, xf, emb):
        return model.apply(params, x, xf, emb)
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, x, xf, emb).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 100:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, x, xf, emb).block_until_ready()
    execution_time = timeit.timeit(lambda: jit_forward(params, x,xf, emb).block_until_ready(), number=10)
    print(f"Jit forward pass time: {execution_time / 100:.6f} seconds per run")
    

if __name__ == "__main__":
    main()