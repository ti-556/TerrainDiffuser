import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import jax.numpy as jnp
import torchvision.transforms as transforms

class MapDataset(Dataset):
    def __init__(self, dem_dir, sat_dir, vec_dir, resolution=32):
        self.dem_dir = dem_dir
        self.dem_filenames = sorted([file for file in os.listdir(dem_dir) if file.lower().endswith(".png")])
        self.sat_dir = sat_dir
        self.sat_filenames = sorted([file for file in os.listdir(sat_dir) if file.lower().endswith(".png")])

        print(f"dem number images: {len(self.dem_filenames)}, sat number images: {len(self.sat_filenames)}")
        assert len(self.dem_filenames) == len(self.sat_filenames), "sat data and dem data do not match."

        self.vecs = torch.load(vec_dir)
        self.text_embedding_map = {
            os.path.basename(path): idx for idx, path in enumerate(self.vecs["image_paths"])
        }
        self.transform = transforms.Resize((resolution, resolution))

    def __len__(self):
        return len(self.dem_filenames)

    def __getitem__(self, idx):
        id_str = f"{idx+1:06d}"
        image_filename = f"image{id_str}.png"
        text_label = self.vecs["text_embeddings"][self.text_embedding_map[image_filename]]

        dem_filename = os.path.join(self.dem_dir, self.dem_filenames[idx])
        sat_filename = os.path.join(self.sat_dir, self.sat_filenames[idx])
        dem_image = Image.open(dem_filename).convert("L")
        sat_image = Image.open(sat_filename).convert("RGB")

        dem_image = self.transform(dem_image)
        sat_image = self.transform(sat_image)
        
        dem_image = jnp.array(np.repeat(np.expand_dims(np.array(dem_image), axis=-1), 3, axis=-1))
        sat_image = jnp.array(np.array(sat_image))
        
        return dem_image, sat_image, jnp.array(text_label.cpu().numpy())