import numpy as np
import nibabel as nib
import glob, os, re, random
from typing import List, Optional

#plan#
#I want to be able to load the dataset with location of data as command line arg so rangpur cluster can be used 


def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata(caching='unchanged')
    if arr.dim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)

def z_score(x):
    m, s = x.mean(), x.std() + 1e-6
    return (x - m) / s

def random_crop(img, size):
    h, w = img.shape
    if h < size or w  < size:
        padh, padw = max(0, size -h), max(0, size - w)
        img = np.pad(img, ((padh//2, padh - padh // 2 ), (padw//2, padw - padw//2)), mode='reflect')
        h, w = img.shape
    i = random.randint(0, h - size)
    j = random.randint(0, w - size)
    return img[i:i+size, j:j+size]


