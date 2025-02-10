from layers.terrain_diffuser import TerrainDiffuser
import jax
import jax.numpy as jnp
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)
    x = jax.random.normal(subkey1, (8, 32, 32, 64))
    emb = jax.random.normal(subkey2, (8, 77, 128))
    xf = jax.random.normal(subkey3, (8, 16, 16, 32))
    t = jax.random.uniform(subkey4, (8, ))
    
    model = TerrainDiffuser(
        out_channels = 3,
        img_resolution = 32,
        num_groups = 16,
        model_channels = 128,
        channel_mult = [2, 2, 2],
        time_emb_mult = 4,
        num_blocks = [4, 4, 4],
        cond_resolutions = {"16": 32},
        p_dropout = 0.1,
        epsilon = 1e-5,
        reduction = 16,
        use_bias = False,
        num_heads = 1,
        skip_scaling = 1/jnp.sqrt(2),
    )
    params = model.init(key, x, t, emb, xf)
    
    def forward_fn(params, x, t, emb, xf):
        return model.apply(params, x, t, emb, xf, rngs={'dropout': subkey5})
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, x, t, emb, xf).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 10:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, x, t, emb, xf).block_until_ready()
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, x, t, emb, xf).block_until_ready(), number=10)
    print(f"Jit forward pass time: {jit_execution_time / 10:.6f} seconds per run")
    
    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")

if __name__ == "__main__":
    main()