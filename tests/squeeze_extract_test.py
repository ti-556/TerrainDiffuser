from layers.squeeze_extract import SELayer
import jax
import jax.numpy as np
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(8, 32, 32, 64))
    
    model = SELayer(
        channels=64,
        reduction=16
    )
    params = model.init(key, x)
    
    def forward_fn(params, x):
        return model.apply(params, x)
    
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jit benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, x).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 100:.6f} seconds per run")
    
    print("Jit benchmark:")
    jit_forward(params, x).block_until_ready()
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, x).block_until_ready(), number=10)
    print(f"Jit forward pass time: {jit_execution_time / 100:.6f} seconds per run")

    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")
if __name__ == "__main__":
    main()