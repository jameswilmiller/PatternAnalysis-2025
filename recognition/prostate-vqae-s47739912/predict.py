import os
from pathlib import Path
import torch
from modules import *
from train import *
from dataset import *


p = Parameters(profile="local")
device = p.device



@torch.no_grad()
def sample_images(p):
    device = p.device
    os.makedirs("samples", exist_ok=True)

    #load the vqvae
    vqvae = VQVAE(
        embedding_dim=p.embedding_dim,
        num_embeddings=p.num_embeddings,
        beta=p.beta
        ).to(device)
    vq_state = torch.load("best_vqvae.pt", map_location=device)
    vqvae.load_state_dict(vq_state)
    vqvae.eval()

    #infer latent height width
    dummy = torch.zeros(1, 1, *p.img_size, device=device)
    code = vqvae.encode(dummy)
    D, H, W = code.shape[1:]

    pixelcnn = PixelCNN(init_channel=p.embedding_dim,
                        channels=256,
                        out_channel=p.num_embeddings,
                        num_resid=10).to(device)
    pix_state = torch.load("logs/best_pixelcnn.pt", map_location=device)
    pixelcnn.load_state_dict(pix_state)
    pixelcnn.eval()

    #sample indices and decode using vqvae
    idx, z = sample(vqvae, pixelcnn, B=16, D=D, H=H, W=W,
                    t=0.8, device=device)
    imgs = decode_z(vqvae, z)
    grid = make_grid(imgs, nrow=4, padding=2, pad_value=0.5)
    save_image(grid, os.path.join("samples", "pixel_cnn_samples.png"))

@torch.no_grad()
def eval_test(model, test_loader, device, ssim_metric):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0

    for batch in tqdm(test_loader, leave=False):
        batch = batch.to(device)
        recon, codebook_loss, commitment_loss, indices = model(batch)
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss, ssim_weight=0.15)
        total_loss += loss.item()
        total_ssim += ssim_metric(recon, batch).item()
    
    n = len(test_loader)
    return total_loss / n, total_ssim / n

@torch.no_grad()
def plot_original_recon(model, loader, device, out_dir="vis", max_images=8):
    """
    make 2 row grid originals on top and reconstructions on the bottom
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader)).to(device)
    recon, _, _, _ = model(batch)

    #pick up to the images num selected
    n = min(max_images, batch.size(0))
    originals = batch[:n].clamp(0, 1)
    recons = recon[:n].clamp(0, 1)

    #stack into 2 row grid
    grid = make_grid(torch.cat([originals, recons], dim=0), nrow=n, padding=2, pad_value=0.5)
    save_path = os.path.join(out_dir, "original_recon_grid.png")
    save_image(grid, save_path)

def plot_latent_quant(model, loader, device, out_dir="vis", channel=0):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    #get a single image
    batch = next(iter(loader)).to(device)
    x = batch[0:1]

    #get the latent pre VQ
    code = model.encode(x)

    quantised, _, _, indices = model.reparameterise(code)

    #latent channel heatmap
    latent_channel = code[0, channel].detach().cpu().float()

    plt.figure(figsize=(6,5))
    plt.imshow(latent_channel.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("latent pre vector quantised channel")
    latent_path = os.path.join(out_dir, "latent_channel.png")
    plt.savefig(latent_path)
    plt.close()

    #quantised channel heatmap
    quantised_channel = quantised[0, channel].detach().cpu().float()

    plt.figure(figsize=(6,5))
    plt.imshow(quantised_channel.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("post vector quantised latent channel")
    latent_path = os.path.join(out_dir, "quantised_channel.png")
    plt.savefig(latent_path)
    plt.close()

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
    plot_original_recon(model, test_loader, device, out_dir="vis", max_images=8)
    plot_latent_quant(model, test_loader, device, out_dir="vis", channel=0)
    print(f"avg test loss: {avg_loss}")
    print(f"avg test ssim: {avg_ssim}")
    sample_images(p)



if __name__ == "__main__":
    main()
    