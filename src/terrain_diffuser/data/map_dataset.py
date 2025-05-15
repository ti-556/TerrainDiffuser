
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py

class MapDataset(Dataset):
    """
    DEM (1-channel) & SAT (RGB) 32×32 images
    + text embeddings stored in an HDF5 file:
        ├─ "image_ids"       Dataset of shape (N,) dtype string
        └─ "embeddings"      Dataset of shape (N, S, D) dtype float32
    """
    def __init__(self, dem_dir: str, sat_dir: str, h5_file: str,
                transform: transforms.Compose = transforms.Compose([ 
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * 2 - 1)
                ])
    ):
        super().__init__()
        self.dem_dir   = dem_dir
        self.sat_dir   = sat_dir
        self.transform = transform

        # list image files
        self.dem_files = sorted([f for f in os.listdir(dem_dir)
                                 if f.lower().endswith(".png")])
        self.sat_files = sorted([f for f in os.listdir(sat_dir)
                                 if f.lower().endswith(".png")])
        assert len(self.dem_files) == len(self.sat_files), \
            "DEM and SAT directories must contain the same number of images"

        print(f"Found {len(self.dem_files)} DEM/SAT image pairs")

        # open HDF5 file for embeddings
        self.h5 = h5py.File(h5_file, "r")
        # datasets
        self.ids_ds  = self.h5["image_ids"]       # shape (N,)
        self.emb_ds  = self.h5["embeddings"]      # shape (N, S, D)

        # build a lookup from filename → index in HDF5
        # HDF5 strings come back as bytes, so decode
        self.id_map = {
            self.ids_ds[i].decode("utf-8") if isinstance(self.ids_ds[i], bytes)
            else self.ids_ds[i]: i
            for i in range(len(self.ids_ds))
        }

    def __len__(self):
        return len(self.dem_files)

    def __getitem__(self, idx):
        # load images
        dem_path = os.path.join(self.dem_dir, self.dem_files[idx])
        sat_path = os.path.join(self.sat_dir, self.sat_files[idx])
        dem_img  = Image.open(dem_path).convert("L")
        sat_img  = Image.open(sat_path).convert("RGB")

        if self.transform:
            dem_img = self.transform(dem_img)
            sat_img = self.transform(sat_img)

        # look up embedding by filename
        key = self.dem_files[idx]  # e.g. "image_0001234.png"
        emb_idx = self.id_map[key]
        emb_np  = self.emb_ds[emb_idx]            # NumPy array (S, D)
        text_embed = torch.from_numpy(emb_np)     # convert to Tensor

        return dem_img, sat_img, text_embed

    def __del__(self):
        # ensure the HDF5 file is closed when dataset is freed
        try:
            self.h5.close()
        except Exception:
            pass
