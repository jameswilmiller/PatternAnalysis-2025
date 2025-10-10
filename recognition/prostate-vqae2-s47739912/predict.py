import os
from pathlib import Path
import torch
from modules import *
from dataset import *
p = Parameters(profile="local")
device = p.device
if device.type != 'cuda':
    print("Warning CUDA not found. using CPU")


def main(top_n, path):
    #data
    loaders = KerasSlicesDataLoader(p)
    test_loader = loaders.get_test()

    #model
    model = VQVAE(
        embedding_dim=p.embedding_dim,
        num_embeddings=p.num_embeddings,
        beta=p.beta
    ).to(device)
    s = torch.load(path, map_location=device)
    model.load_state_dict(s)
    model.eval()

 







