import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from noise import pnoise2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

INPUT_HEIGHTMAP_PATH = 'imageXXXXXX.png'
OUTPUT_HEIGHTMAP_PATH = 'imageXXXXXX_highres.png'

# Upscaling Parameters
SCALE_FACTOR = 64
UPSCALE_ORDER = 3
# Perlin Noise Parameters (Main Noise)
OCTAVES = 4
PERSISTENCE = 0.6
LACUNARITY = 3.0
SEED = 42
# Ridged Noise Parameters
RIDGED_WEIGHT = 0.02
# Additional small noise amplitude
NOISE_AMPLITUDE = 0.2
# Secondary Noise Pass (small-scale details)
ENABLE_SECONDARY_NOISE = True
SECONDARY_NOISE_SCALE = 0.05
SECONDARY_OCTAVES = 2
SECONDARY_PERSISTENCE = 0.5
SECONDARY_LACUNARITY = 4.0
# Masking Parameters
ELEVATION_THRESHOLD_HIGH = 0.65
ELEVATION_THRESHOLD_MED = 0.2
SLOPE_EXPONENT = 10
# Post-Processing: Gaussian Blur Parameters
GAUSSIAN_SIGMA = 1.0
LOW_REGION_GAUSSIAN_SIGMA = 1.0
LOW_REGION_ELEVATION_THRESHOLD = 0.3
# Reduce flat‐region noise boost
FLAT_NOISE_FACTOR = 0.05
# Visualization Flags
VISUALIZE = True
VISUALIZE_3D = True
# 3D Plot Parameters
Z_SCALE = 30.0
MIN_SLOPE_MASK = 0.2

def load_and_normalize_heightmap(path):
    try:
        heightmap_img = Image.open(path).convert('L')
        H = np.array(heightmap_img, dtype=np.float32)
        H_min, H_max = H.min(), H.max()
        H_norm = (H - H_min) / (H_max - H_min)
        print(f"Loaded heightmap '{path}' with shape {H.shape}.")
        return H_norm
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        exit(1)

def upscale_heightmap(H, scale_factor, order=3):
    H_upscaled = zoom(H, scale_factor, order=order)
    print(f"Upscaled heightmap to {H_upscaled.shape[1]}x{H_upscaled.shape[0]} using order={order}.")
    return H_upscaled

def calculate_slope(H):
    dH_dx = np.gradient(H, axis=1)
    dH_dy = np.gradient(H, axis=0)
    S = np.sqrt(dH_dx**2 + dH_dy**2)
    S_min, S_max = S.min(), S.max()
    S_norm = (S - S_min) / (S_max - S_min + 1e-8)
    print("Calculated normalized slope map.")
    return S_norm

def elevation_mask(E_norm, threshold_high=ELEVATION_THRESHOLD_HIGH, threshold_med=ELEVATION_THRESHOLD_MED):
    M = np.interp(E_norm, [0.0, threshold_med, threshold_high, 1.0], [0.5, 0.5, 1.0, 1.0])
    print("Created smooth elevation mask.")
    return M

def slope_mask(S_norm, exponent=SLOPE_EXPONENT, min_val=MIN_SLOPE_MASK):
    M_S = min_val + (1.0 - min_val) * np.power(S_norm, exponent)
    print(f"Created slope mask with minimum value {min_val} (exponent={exponent}).")
    return M_S

def combine_masks(M_S, M_E):
    M = 0.5 * (M_S + M_E)
    print("Combined elevation and slope masks by averaging.")
    return M

def generate_fractal_perlin(width, height, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
    noise_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            x = j / float(width)
            y = i / float(height)
            val = pnoise2(
                x * lacunarity,
                y * lacunarity,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            noise_map[i][j] = val
    noise_min, noise_max = noise_map.min(), noise_map.max()
    noise_map = (noise_map - noise_min) / (noise_max - noise_min + 1e-8)
    return noise_map

def generate_ridged_fractal(width, height, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
    noise_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            x = j / float(width)
            y = i / float(height)
            val = pnoise2(
                x * lacunarity,
                y * lacunarity,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            val = 1.0 - abs(val)
            noise_map[i][j] = val
    noise_min, noise_max = noise_map.min(), noise_map.max()
    noise_map = (noise_map - noise_min) / (noise_max - noise_min + 1e-8)
    return noise_map

def generate_combined_noise(width, height, 
                            main_octaves=6, main_persistence=0.5, main_lacunarity=2.0, main_seed=0,
                            ridged_octaves=6, ridged_persistence=0.5, ridged_lacunarity=2.0, ridged_seed=1,
                            ridged_weight=0.3):
    standard_noise = generate_fractal_perlin(
        width, height,
        octaves=main_octaves,
        persistence=main_persistence,
        lacunarity=main_lacunarity,
        seed=main_seed
    )
    ridged_noise = generate_ridged_fractal(
        width, height,
        octaves=ridged_octaves,
        persistence=ridged_persistence,
        lacunarity=ridged_lacunarity,
        seed=ridged_seed
    )
    combined = (1.0 - ridged_weight) * standard_noise + ridged_weight * ridged_noise
    cmin, cmax = combined.min(), combined.max()
    combined = (combined - cmin) / (cmax - cmin + 1e-8)
    return combined

def add_secondary_noise(base_map, scale=0.5, octaves=4, persistence=0.6, lacunarity=2.5, seed=100):
    height, width = base_map.shape
    secondary = generate_fractal_perlin(
        width, height,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed
    )
    combined = base_map + scale * secondary
    cmin, cmax = combined.min(), combined.max()
    combined = (combined - cmin) / (cmax - cmin + 1e-8)
    return combined

def apply_noise_to_heightmap(H_upscaled, noise_map, mask):
    H_final = H_upscaled + NOISE_AMPLITUDE * mask * noise_map
    fmin, fmax = H_final.min(), H_final.max()
    H_final = (H_final - fmin) / (fmax - fmin + 1e-8)
    print("Applied masked noise with a small amplitude to heightmap.")
    return H_final

def post_process(H, sigma=GAUSSIAN_SIGMA, low_sigma=LOW_REGION_GAUSSIAN_SIGMA,
                 elevation_threshold=LOW_REGION_ELEVATION_THRESHOLD):
    heavy_blur = gaussian_filter(H, sigma=low_sigma)
    light_blur = gaussian_filter(H, sigma=sigma)
    mask = np.clip((elevation_threshold - H) / elevation_threshold, 0, 1)
    H_blurred = mask * heavy_blur + (1 - mask) * light_blur
    H_blurred = (H_blurred - H_blurred.min()) / (H_blurred.max() - H_blurred.min() + 1e-8)
    print(f"Applied variable Gaussian blur: sigma={sigma}, low_sigma={low_sigma}.")
    return H_blurred

def save_heightmap(H, path):
    H_uint8 = (H * 255).astype(np.uint8)
    heightmap_img = Image.fromarray(H_uint8, mode='L')
    heightmap_img.save(path)
    print(f"Saved heightmap to '{path}'.")

def visualize_heightmaps(H_original, H_upscaled, H_final):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = 'terrain'
    axes[0].imshow(H_original, cmap=cmap)
    axes[0].set_title('Original (32x32)')
    axes[0].axis('off')
    axes[1].imshow(H_upscaled, cmap=cmap)
    axes[1].set_title(f'Upscaled ({H_upscaled.shape[1]}x{H_upscaled.shape[0]})')
    axes[1].axis('off')
    axes[2].imshow(H_final, cmap=cmap)
    axes[2].set_title(f'Noised ({H_upscaled.shape[1]}x{H_upscaled.shape[0]})')
    axes[2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Displayed 2D comparison of heightmaps.")

def visualize_subdivided_noised_3d(H_upscaled, H_final, z_scale=30.0, stride=10):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    y_size_u, x_size_u = H_upscaled.shape
    Xu, Yu = np.meshgrid(np.arange(x_size_u), np.arange(y_size_u))
    Z_up = H_upscaled * z_scale
    Xu_sub = Xu[::stride, ::stride]
    Yu_sub = Yu[::stride, ::stride]
    Z_up_sub = Z_up[::stride, ::stride]
    ax1.plot_surface(Xu_sub, Yu_sub, Z_up_sub, rstride=1, cstride=1,
                     cmap='terrain', linewidth=0, antialiased=True)
    ax1.set_title(f"Upscaled Terrain ({x_size_u}x{y_size_u})")
    ax1.set_zlim(0, z_scale)
    ax2 = fig.add_subplot(122, projection='3d')
    y_size_f, x_size_f = H_final.shape
    Xf, Yf = np.meshgrid(np.arange(x_size_f), np.arange(y_size_f))
    Z_final = H_final * z_scale
    Xf_sub = Xf[::stride, ::stride]
    Yf_sub = Yf[::stride, ::stride]
    Zf_sub = Z_final[::stride, ::stride]
    ax2.plot_surface(Xf_sub, Yf_sub, Zf_sub, rstride=1, cstride=1,
                     cmap='terrain', linewidth=0, antialiased=True)
    ax2.set_title(f"Noised Terrain ({x_size_f}x{y_size_f})")
    ax2.set_zlim(0, z_scale)
    plt.tight_layout()
    plt.show()
    print("Displayed 3D surface plots of subdivided and noised terrains.")

def main():
    # load and Normalize Heightmap
    H_norm = load_and_normalize_heightmap(INPUT_HEIGHTMAP_PATH)
    # upscale Heightmap
    H_upscaled = upscale_heightmap(H_norm, SCALE_FACTOR, order=UPSCALE_ORDER)
    # Calculate Slope
    S_norm = calculate_slope(H_upscaled)
    # Define Elevation and Slope Masks
    M_E = elevation_mask(H_upscaled, threshold_high=ELEVATION_THRESHOLD_HIGH,
                         threshold_med=ELEVATION_THRESHOLD_MED)
    M_S = slope_mask(S_norm, exponent=SLOPE_EXPONENT)
    M = combine_masks(M_S, M_E)
    # Reduce extra noise in flat areas
    M = M + FLAT_NOISE_FACTOR * (1 - S_norm)
    M = np.clip(M, 0, 1)
    # generate Combined Noise
    height, width = H_upscaled.shape
    combined_noise = generate_combined_noise(
        width, height,
        main_octaves=OCTAVES,
        main_persistence=PERSISTENCE,
        main_lacunarity=LACUNARITY,
        main_seed=SEED,
        ridged_octaves=OCTAVES,
        ridged_persistence=PERSISTENCE,
        ridged_lacunarity=LACUNARITY,
        ridged_seed=SEED + 1,
        ridged_weight=RIDGED_WEIGHT
    )
    # Add secondary noise pass for extra micro-details (optional)
    if ENABLE_SECONDARY_NOISE:
        combined_noise = add_secondary_noise(
            combined_noise,
            scale=SECONDARY_NOISE_SCALE,
            octaves=SECONDARY_OCTAVES,
            persistence=SECONDARY_PERSISTENCE,
            lacunarity=SECONDARY_LACUNARITY,
            seed=SEED + 10
        )
    # apply Masked Noise (with small amplitude) to Heightmap
    H_final = apply_noise_to_heightmap(H_upscaled, combined_noise, M)
    # Optional Post-Processing (Variable Gaussian Blur)
    H_final = post_process(H_final)
    # Save Final Heightmap
    save_heightmap(H_final, OUTPUT_HEIGHTMAP_PATH)
    # 2D Visualization
    if VISUALIZE:
        visualize_heightmaps(H_norm, H_upscaled, H_final)
    # 3D Visualization (upscaled vs. noised comparison)
    if VISUALIZE_3D:
        visualize_subdivided_noised_3d(H_upscaled, H_final, z_scale=Z_SCALE)

if __name__ == "__main__":
    main()
