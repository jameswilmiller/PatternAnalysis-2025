import numpy as np
import nibabel as nib
from pathlib import Path 
import os
from tqdm import tqdm
import torch
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
#use these paths when using rangpur


#BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
#TRAIN_DIR = os.path.join(BASE_DIR, "keras_slices_train")
#VAL_DIR = os.path.join(BASE_DIR, "keras_slices_validate")
#TEST_DIR = os.path.join(BASE_DIR, "keras_slices_test")


CURR = Path(__file__).resolve().parent
ROOT = CURR.parents[1] / "data" / "keras_slices_data"

TRAIN_DIR = ROOT / "keras_slices_train"
VAL_DIR = ROOT / "keras_slices_validate"
TEST_DIR = ROOT / "keras_slices_test"

def list_nifti(folder: Path):
    return sorted(str(p) for p in folder.rglob("*") if p.name.lower().endswith(".nii.gz"))

class KerasSlicesDataset(Dataset):
    def __init__(self, image_paths, size=(256, 256), norm=True):
        self.image_paths = image_paths
        self.size = size
        self.norm = norm

    def __len__(self):
        return len(self.image_paths)
    
    def load_2d(self, path):
        arr = nib.load(path).get_fdata(caching='unchanged')
        if len(arr.shape) == 3:
            arr = arr [:, :, 0]
        arr = arr.astype(np.float32)
        return arr
    

    def __getitem__(self, idx):
        arr = self.load_2d(self.image_paths[idx])

        #convert to tensor
        x = torch.from_numpy(arr).unsqueeze(0)

        #now resize
        h, w = x.shape[-2], x.shape[-1]
        if (h, w) != self.size:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        #normalise
        if self.norm:
            vmin = torch.min(x)
            vmax = torch.max(x)
            if torch.isfinite(vmin) and torch.isfinite(vmax) and (vmax > vmin):
                x = (x - vmin) / (vmax - vmin)
            else:
                x = torch.zeros_like(x)
        return x

train = list_nifti(TRAIN_DIR)


