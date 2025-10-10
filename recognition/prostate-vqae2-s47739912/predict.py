import os
from pathlib import Path
import torch
from modules import *
from train import *
from dataset import *
p = Parameters(profile="local")
device = p.device
if device.type != 'cuda':
    print("Warning CUDA not found. using CPU")



        

 







