import numpy as np
import nibabel as nib
from pathlib import Path 
import os
from tqdm import tqdm
import torch
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from modules import *
#use these paths when using rangpur






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





class KerasSlicesDataLoader():

    def __init__(self, p: Parameters = None):
        self.p = p if p is not None else Parameters()

        


        train_paths = list_nifti(self.p.train_dir)
        val_paths = list_nifti(self.p.val_dir)
        test_paths = list_nifti(self.p.test_dir)

        self.train_data = KerasSlicesDataset(train_paths, size=self.p.img_size, norm=self.p.normalise)
        self.val_data = KerasSlicesDataset(val_paths, size=self.p.img_size, norm=self.p.normalise)
        self.test_data = KerasSlicesDataset(test_paths, size=self.p.img_size, norm=self.p.normalise)


    def get_train(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.p.batch_size, 
            shuffle=True
            )
        return train_loader
    
    def get_validation(self):
        validation_loader = DataLoader(
            self.val_data, 
            batch_size=self.p.batch_size, 
            shuffle=False
            )
        return validation_loader
    
    def get_test(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size = self.p.batch_size,
            shuffle=False
            )
        return test_loader


        