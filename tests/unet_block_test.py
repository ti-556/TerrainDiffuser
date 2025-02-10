from layers.unet_block import UnetBlock
import jax
import jax.numpy as jnp
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)
    x = jax.random.normal(subkey1, (8, 32, 32, 64))
    text_emb = jax.random.normal(subkey2, (8, 77, 128))
    xf = jax.random.normal(subkey3, (8, 32, 32, 16))
    time_emb = jax.random.uniform(subkey4, (8, 128))
    
    model = UnetBlock(
        model_channels=64,
        p_dropout=0.1,
        feature_channels=16,
        num_heads=1,
        num_groups=16,
        epsilon=1e-5,
        use_bias=False,
        reduction=16,
        attn=True,
        
    )
    params = model.init(key, x, time_emb, text_emb, xf)
    
    def forward_fn(params, x, time_emb, text_emb, xf):
        return model.apply(
            params, x, time_emb, text_emb, xf, rngs={'dropout': subkey5}
        )
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, x, time_emb, text_emb, xf).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 100:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, x, time_emb, text_emb, xf).block_until_ready()
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, x, time_emb, text_emb, xf).block_until_ready(), number=10)
    print(f"Jit forward pass time: {jit_execution_time / 100:.6f} seconds per run")
    
    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")

if __name__ == "__main__":
    main()