import os
import glob
import rasterio
import numpy as np
from PIL import Image

input_dir = r'/earthengineexport' # tiff file dir
output_dir = r'/outputdir' 

def rescale_to_8bit(array, min_val=None, max_val=None):
    """
    Rescales a floating-point or integer numpy array to 8-bit (0-255).
    If min_val and max_val are not provided, they will be derived from the array.
    """
    if min_val is None:
        min_val = np.nanmin(array)
    if max_val is None:
        max_val = np.nanmax(array)
    
    if max_val == min_val:
        return np.zeros_like(array, dtype=np.uint8)
    
    scaled = (array - min_val) / (max_val - min_val)
    scaled = np.clip(scaled, 0, 1) * 255
    return scaled.astype(np.uint8)


tif_files = glob.glob(os.path.join(input_dir, '*.tif'))

for tif_path in tif_files:
    filename = os.path.basename(tif_path)
    out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
    
    with rasterio.open(tif_path) as src:
        if 'DEM' in filename.upper():
            dem_data = src.read(1)
            dem_scaled = rescale_to_8bit(dem_data)
            dem_img = Image.fromarray(dem_scaled, mode='L')
            dem_img.save(out_path, format='PNG')
            print(f"Converted DEM: {tif_path} -> {out_path}")

        elif 'S2' in filename.upper():
            blue = src.read(1)
            green = src.read(2)
            red = src.read(3)
            
            min_val = 0
            max_val = 3000
            
            red_8bit = rescale_to_8bit(red, min_val, max_val)
            green_8bit = rescale_to_8bit(green, min_val, max_val)
            blue_8bit = rescale_to_8bit(blue, min_val, max_val)
            
            rgb_img = np.dstack((red_8bit, green_8bit, blue_8bit))
            rgb_pil = Image.fromarray(rgb_img, mode='RGB')
            rgb_pil.save(out_path, format='PNG')
            print(f"Converted Sentinel-2: {tif_path} -> {out_path}")

        else:
            print(f"Skipping {tif_path}: filename does not indicate DEM or S2.")
