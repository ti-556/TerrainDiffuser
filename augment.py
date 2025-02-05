import os
import glob
import random
from PIL import Image

def augment_pair(dem_img, sat_img):

    do_flip_lr = (random.random() > 0.5)
    do_flip_tb = (random.random() > 0.5)
    angle = random.choice([0, 90, 180, 270])

    if do_flip_lr:
        dem_img = dem_img.transpose(Image.FLIP_LEFT_RIGHT)
    if do_flip_tb:
        dem_img = dem_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if angle == 90:
        dem_img = dem_img.transpose(Image.ROTATE_90)
    elif angle == 180:
        dem_img = dem_img.transpose(Image.ROTATE_180)
    elif angle == 270:
        dem_img = dem_img.transpose(Image.ROTATE_270)

    if do_flip_lr:
        sat_img = sat_img.transpose(Image.FLIP_LEFT_RIGHT)
    if do_flip_tb:
        sat_img = sat_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if angle == 90:
        sat_img = sat_img.transpose(Image.ROTATE_90)
    elif angle == 180:
        sat_img = sat_img.transpose(Image.ROTATE_180)
    elif angle == 270:
        sat_img = sat_img.transpose(Image.ROTATE_270)

    return dem_img, sat_img

def resize_image(img, size, resample=Image.BILINEAR):

    return img.resize((size, size), resample=resample)

def ensure_dir_exists(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def random_crop(dem_img, sat_img, crop_size):

    width, height = dem_img.size
    if width < crop_size or height < crop_size:
        raise ValueError("Crop size must be smaller than the image size.")

    x_max = width - crop_size
    y_max = height - crop_size
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)

    dem_cropped = dem_img.crop((x, y, x + crop_size, y + crop_size))
    sat_cropped = sat_img.crop((x, y, x + crop_size, y + crop_size))

    return dem_cropped, sat_cropped

def prepare_data(
    input_dir="ImageExports",
    output_dir=".",
    total_samples=100,  
    dem512_dir="dem512directory",
    dem32_dir="dem32directory",
    sat512_dir="sat512directory",
    sat32_dir="sat32directory",
    crop_size=512,      
    resize_sizes={'512': 512, '32': 32}  
):
    
    dem512_path = os.path.join(output_dir, dem512_dir)
    dem32_path  = os.path.join(output_dir, dem32_dir)
    sat512_path = os.path.join(output_dir, sat512_dir)
    sat32_path  = os.path.join(output_dir, sat32_dir)
    
    ensure_dir_exists(dem512_path)
    ensure_dir_exists(dem32_path)
    ensure_dir_exists(sat512_path)
    ensure_dir_exists(sat32_path)
    
    dem_files = sorted(glob.glob(os.path.join(input_dir, "*DEM.png")))
    sat_files = sorted(glob.glob(os.path.join(input_dir, "*S2.png")))

    pairs = list(zip(dem_files, sat_files))
    if not pairs:
        print("No DEM/S2 pairs found in the input directory.")
        return

    num_pairs = len(pairs)
    
    image_pairs_in_memory = []
    for dem_path, sat_path in pairs:

        dem_img = Image.open(dem_path).convert("RGB")
        sat_img = Image.open(sat_path).convert("RGB")
        image_pairs_in_memory.append((dem_img, sat_img))
    
    for i in range(total_samples):

        dem_img, sat_img = random.choice(image_pairs_in_memory)
        
        file_id_str = f"{i+1:06d}"
        
        try:
            dem_cropped, sat_cropped = random_crop(dem_img, sat_img, crop_size)
        except ValueError as ve:
            print(f"Sample {i+1}: {ve}")
            continue 

        dem_aug, sat_aug = augment_pair(dem_cropped, sat_cropped)
        
        dem_resized = {}
        sat_resized = {}
        for key, size in resize_sizes.items():
            dem_resized[key] = resize_image(dem_aug, size)
            sat_resized[key] = resize_image(sat_aug, size)
        
        dem512_outfile = os.path.join(dem512_path, f"image{file_id_str}.png")
        dem32_outfile  = os.path.join(dem32_path,  f"image{file_id_str}.png")
        sat512_outfile = os.path.join(sat512_path, f"image{file_id_str}.png")
        sat32_outfile  = os.path.join(sat32_path,  f"image{file_id_str}.png")
        
        dem_resized['512'].save(dem512_outfile, compress_level=1)
        dem_resized['32'].save(dem32_outfile, compress_level=1)
        sat_resized['512'].save(sat512_outfile, compress_level=1)
        sat_resized['32'].save(sat32_outfile, compress_level=1)

        if (i+1) % 100 == 0:
            print(f"Generated {i+1} / {total_samples} samples.")

    print(f"Data preparation complete! Generated {total_samples} samples.")


#data cropping and augmenting
if __name__ == "__main__":
    prepare_data(
        input_dir=r"/home/mnmlk/projects/ess/dataset2images",
        output_dir=".",
        total_samples=50000,
        dem512_dir=r"/home/mnmlk/projects/ess/dataset2/dem512",
        dem32_dir=r"/home/mnmlk/projects/ess/dataset2/dem32",
        sat512_dir=r"/home/mnmlk/projects/ess/dataset2/sat512",
        sat32_dir=r"/home/mnmlk/projects/ess/dataset2/sat32",
        crop_size=512,           
        resize_sizes={'512': 512, '32': 32}
    )
