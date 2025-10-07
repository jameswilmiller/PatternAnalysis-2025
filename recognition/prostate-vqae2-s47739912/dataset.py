import numpy as np
import nibabel as nib
import glob, os, re, random
from typing import List, Optional
from torch.utils.data import Dataset 
import torch 
#plan#
#I want to be able to load the dataset with location of data as command line arg so rangpur cluster can be used 
BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
TRAIN_DIR = os.path.join(BASE_DIR, "keras_slices_train")
VAL_DIR = os.path.join(BASE_DIR, "keras_slices_validate")
TEST_DIR = os.path.join(BASE_DIR, "keras_slices_test")

BATCH_SIZE = 32
CROP_SIZE = 128
CROP_PER_IMG = 512
AUG_TRAIN = True
AUG_VAL = False

def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata(caching='unchanged')
    if arr.dim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)
#swapped to min max to work with ssm (z scoring makes intensities unbounded distoring ssm score)
def minmax(x):
    x_min = x.min()
    x_max = x.max()
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return np.zeros_like(x, dtype=np.float32)
    s = x_max - x_min
    if s < 1e-12: 
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / s


def random_crop(img, size):
    h, w = img.shape
    if h < size or w  < size:
        padh, padw = max(0, size -h), max(0, size - w)
        img = np.pad(img, ((padh//2, padh - padh // 2 ), (padw//2, padw - padw//2)), mode='reflect')
        h, w = img.shape
    i = random.randint(0, h - size)
    j = random.randint(0, w - size)
    return img[i:i+size, j:j+size]


class SlicesData(Dataset):
    """
    find and load all .nii.gz files recursively under given directory
    """
    def __init__(self, dir: str, crop_size: int, crop_per_img: int, augment: bool ):
        self.dir = dir
        self.crop_size = crop_size
        self.crop_per_img = crop_per_img
        self.augment = augment

        if not os.path.isdir(dir):
            raise FileNotFoundError(f"Directory not found {dir}")
        
        self.paths = sorted(glob.glob(os.path.join(dir, "**", "*.nii.gz"), recursive=True))

        if len(self.paths) == 0:
            raise FileNotFoundError(f"no .nii.gz files found in {dir}")
        
        self.num_files = len(self.paths)
        self.length = self.nfiles * crop_per_img
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        path = self.paths[random.randrange(self.n_files)]
        #load using nilabel
        img = load_nii(path)
        #normalise
        img = minmax(img)

        #crop
        p = random_crop(img, self.crop_size)

        #if augment is true (i.e training data) augment
        if self.augment:
            #50% chance to flip along axis 0
            if random.random() < 0.5: p = np.flip(p, 0)
            #50% chacne to flip along axis 1
            if random.random() < 0.5: p = np.flip(p, 1)
           
        #convert to tensor
        p = np.expand_dims(p.astype(np.float32), 0)
        return torch.from_numpy(p)
    


