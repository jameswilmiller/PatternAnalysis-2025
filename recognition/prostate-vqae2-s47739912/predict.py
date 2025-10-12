import os
from pathlib import Path
import torch
from modules import *
from train import *
from dataset import *


p = Parameters(profile="local")
device = p.device




@torch.no_grad()
def eval_test(model, test_loader, device, ssim_metric):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0

    for batch in tqdm(test_loader, leave=False):
        batch = batch.to(device)
        recon, codebook_loss, commitment_loss, indices = model(batch)
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
        total_loss += loss.item()
        total_ssim += ssim_metric(recon, batch).item()
    
    n = len(test_loader)
    return total_loss / n, total_ssim / n


def main():
    mod_path = "best_vqvae.pt"

    p = Parameters(profile="local")
    device = p.device
    if device.type != 'cuda':
        print("Warning CUDA not found. using CPU")
 
    loaders = KerasSlicesDataLoader(p)
    test_loader = loaders.get_test()

    #model
    model = VQVAE(
        embedding_dim=p.embedding_dim,
        num_embeddings=p.num_embeddings,
        beta=p.beta
        ).to(device)
    
    state = torch.load(mod_path, map_location=device)
    model.load_state_dict(state)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #eval
    avg_loss, avg_ssim = eval_test(model, test_loader, device, ssim_metric)

    print(f"avg test loss: {avg_loss}")
    print(f"avg test ssim: {avg_ssim}")


if __name__ == "__main__":
    main()
