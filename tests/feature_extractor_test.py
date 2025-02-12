from layers.feature_extractor import VGGFeatureExtractor
import jax
import jax.numpy as jnp
import timeit

def main():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    xf = jax.random.normal(subkey3, (8, 32, 32, 3))
    
    model = VGGFeatureExtractor(
        resolutions={"32": 128, "16": 256},
        img_resolution=32
    )
    params = model.init(key, xf)
    
    def forward_fn(params, xf):
        return model.apply(params, xf)
        
    jit_forward = jax.jit(forward_fn)
    
    print("Non-jitted benchmark:")
    execution_time = timeit.timeit(lambda: forward_fn(params, xf).block_until_ready(), number=10)
    print(f"Non-jit forward pass time: {execution_time / 10:.6f} seconds per run")
    
    print("Jit benchmark:")
    x = jit_forward(params, xf).block_until_ready()
    print(x.shape)
    jit_execution_time = timeit.timeit(lambda: jit_forward(params, xf).block_until_ready(), number=10)
    print(f"Jit forward pass time: {jit_execution_time / 10:.6f} seconds per run")
    
    print(f"Jit speedup gain: {execution_time/jit_execution_time:.2f}x")

if __name__ == "__main__":
    main()