import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from dataset import *
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from itertools import zip_longest
from pytorch_msssim import ssim

@torch.no_grad()
def decode_z(vqvae, z):
    return vqvae.decode(z).clamp(0,1)

@torch.no_grad()
def get_quantized(vqvae, x):
    vqvae.eval()
    code = vqvae.encode(x)
    z, _, _, _ = vqvae.reparameterise(code)
    return z


@torch.no_grad()
def sample(vqvae, pixelcnn, B, D, H, W, t, device="cuda"):
    pixelcnn.eval()
    vqvae.eval()

    z_in = torch.zeros(B, D, H, W, device=device)
    idx = torch.zeros(B, H, W, dtype=torch.long, device=device)
    emb_tab = vqvae.vq.embedding

    for y in range(H):
        for x in range(W):
            l = pixelcnn(z_in)[:, :, y, x] / max(t, 1e-6)
            probs = l.softmax(dim=1)
            ix = torch.multinomial(probs, num_samples=1).squeeze(1)
            idx[:,y,x] = ix
            z_in[:,:,y,x] = emb_tab(ix)
    return idx, z_in

def train_pixelcnn(p=None, epochs=None, save_dir="logs"):
    """
    train pixelcnn on VQVAE quantised latents saves training curves and best_pixelcnn
    """
    os.makedirs(save_dir, exist_ok=True)
    if p is None:
        p = Parameters(profile="local")
    device = p.device
    
    #data 
    loaders = KerasSlicesDataLoader(p)
    train_loader = loaders.get_train()
    val_loader = loaders.get_validation()


    vqvae = VQVAE(embedding_dim=p.embedding_dim, num_embeddings=p.num_embeddings, beta=p.beta).to(device)
    state = torch.load("best_vqvae.pt", map_location=device)
    vqvae.load_state_dict(state)
    vqvae.eval()

    pixelcnn = PixelCNN(init_channel=p.embedding_dim,
                        channels=256,
                        out_channel=p.num_embeddings,
                        num_resid=10).to(device)
    opt = optim.Adam(pixelcnn.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    epochs = epochs or 100
    best_val = float("inf")
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        pixelcnn.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[PixelCNN] Epoch {epoch} / 100 (train)", leave=False)
        for imgs in pbar:
            imgs = imgs.to(device)
            with torch.no_grad():
                code = vqvae.encode(imgs)
                _, _, _, indices = vqvae.reparameterise(code)
                z_in = vqvae.vq.embedding(indices)
                z_in = z_in.permute(0, 3, 1, 2).contiguous()
                
            l = pixelcnn(z_in)
            loss = ce_loss(l, indices.long())

            

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)
        train_losses.append(train_loss)

        #validate
        pixelcnn.eval()
        running = 0.0
        n_batches = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[PixelCNN] Epoch {epoch} / 100 (validation)", leave=False)
            for imgs in pbar:
                imgs = imgs.to(device)
                code = vqvae.encode(imgs)
                _, _, _, indices = vqvae.reparameterise(code)
                z_in = vqvae.vq.embedding(indices).permute(0,3,1,2).contiguous()
                l = pixelcnn(z_in)
                loss = ce_loss(l, indices.long())
                running += loss.item()
                n_batches += 1
        val_loss = running / max(1, n_batches)
        val_losses.append(val_loss)

        #save
        if val_loss < best_val:
            best_val = val_loss
            torch.save(pixelcnn.state_dict(), os.path.join(save_dir, "best_pixelcnn.pt"))

    torch.save(pixelcnn.state_dict(), os.path.join(save_dir, "final_pixelcnn.pt"))

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("Cross entropy")
    plt.legend()
    plt.title("PixelCNN on VQ VAE latents")
    plt.savefig(os.path.join(save_dir, "pixelcnn_loss.png"))
    plt.close()
@torch.no_grad()
def show_visualisation(model, p, out_dir,split, n):
    loaders = KerasSlicesDataLoader(p)
    loader = loaders.get_validation() if split == "validation" else (
        loaders.get_train() if split == "train" else loaders.get_test()
    )
    out = Path(out_dir)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    (out / "indices").mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader)).to(p.device)
    model.eval()
    code = model.encode(batch)
    quant, x, y, indices = model.reparameterise(code)
    recon = model.decode(quant).clamp(0,1)

    nrow = min(n, batch.size(0))
    save_image(make_grid(batch[:n], nrow=nrow, padding=2, pad_value=0.5), out / "originals.png")
    save_image(make_grid(recon[:n], nrow=nrow, padding=2, pad_value=0.5), out / "recons.png")

    #upsscale latents and indices

    h, w = code.shape[-2:]
    target = (h * 8, w * 8)
    
    for i in range(min(n, code.size(0))):
        #latent mean over channels
        z = code[i].mean(0, keepdim=True).unsqueeze(0)
        z_up = F.interpolate(z, size=target, mode="bilinear", 
                             align_corners=False).squeeze().cpu().numpy()
        plt.imsave(str(out / "latents" / f"latent{i}.png"), z_up, cmap = "viridis")

        index = indices[i].view(1,1,h,w).float()
        index_up = F.interpolate(index, size=target,
                                  mode="nearest").squeeze().cpu().numpy()
        plt.imsave(str(out / "indices" / f"indices{i}.png"), index_up, cmap="viridis")



def loss_function(recon, target, codebook_loss, commitment_loss, ssim_weight):
    recon_loss = F.mse_loss(recon, target)
    ssim_loss = 1 - ssim(recon, target, data_range=1.0)
    loss = recon_loss + ssim_weight * ssim_loss
    return  loss + codebook_loss + commitment_loss, recon_loss


@torch.no_grad()
def evaluate(model, val_loader, device, ssim_metric, epoch, epochs):
    """
    Evaluate the model on the validation loader
    returns avg_val_loss and avg_val_ssim
    """
    model.eval()
    val_loss = 0.0
    val_ssim = 0.0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} training", leave=False):
        batch = batch.to(device)
        recon, codebook_loss, commitment_loss, indices= model(batch)
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss, ssim_weight = 0.15)
        val_loss += loss.item()
        val_ssim += ssim_metric(recon, batch).item()

    avg_v_loss = val_loss / len(val_loader)
    avg_v_ssim = val_ssim / len(val_loader)
    return avg_v_loss, avg_v_ssim


def train_epoch(model, train_loader, optimiser, device, ssim_metric, epoch, epochs):
    """
    trains the model for a single epoch
    """
    model.train()
    t_loss = 0.0
    t_ssim = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} training", leave=False):
        optimiser.zero_grad()
        batch = batch.to(device)

        #forward
        recon, codebook_loss, commitment_loss, indices = model(batch)

        #loss + step
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss, ssim_weight=0.15)
        loss.backward()
        optimiser.step()

        t_loss += loss.item()
        t_ssim += ssim_metric(recon, batch).item()

    avg_t_loss = t_loss / len(train_loader)
    avg_t_ssim = t_ssim / len(train_loader)

    return avg_t_loss, avg_t_ssim


def save_metrics(train_loss, 
                 val_losses, 
                 train_ssim, 
                 val_ssims, 
                 best_epoch, 
                 best_v_loss,
                 epochs, 
                 out_path="training_metrics.txt"
                 ):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Training Metrics\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best validation loss {best_v_loss}\n\n")

        f.write("epoch\ttrain_loss\tval_loss\ttrain_ssim\tval_ssim\n")
        for epoch, tl, tv, ts, vs in zip_longest(
            range(1, epochs + 1),
            train_loss,
            val_losses,
            train_ssim,
            val_ssims,
            fillvalue=float("nan")
        ):
            f.write(f"{epoch}\t{tl}\t{tv}\t{ts}\t{vs}\n")
    


def plot_curves(train_loss, val_losses, train_ssim, val_ssims, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    #loss curve
    plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    losspath = os.path.join(out_dir, "loss.png")
    plt.savefig(losspath, bbox_inches="tight", dpi=150)
    plt.close()

    #ssim curve
    plt.figure()
    plt.plot(train_ssim, label="train SSIM")
    plt.plot(val_ssims, label="val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("VQ-VAE SSIM")
    plt.legend()
    ssim_path = os.path.join(out_dir, "ssim.png")
    plt.savefig(ssim_path, bbox_inches="tight", dpi=150)
    plt.close()

def main():
    #build params
    p = Parameters(profile="")

    device = p.device
    if device.type != 'cuda':
        print("Warning: CUDA not found. using cpu")
   
    # prep data
    loaders = KerasSlicesDataLoader(p)
    train_loader = loaders.get_train()
    val_loader = loaders.get_validation()

    

    #model
    model = VQVAE(
        embedding_dim=p.embedding_dim,
        num_embeddings=p.num_embeddings,
        beta=p.beta
        ).to(device)
        
    optimiser = optim.Adam(model.parameters(), lr=p.learning_rate)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


    train_loss, val_losses = [], []
    train_ssim, val_ssims = [], []

    best_v_loss = float("inf")
    best_epoch = -1
    for epoch in range(p.epochs):
        #train
        avg_t_loss, avg_t_ssim = train_epoch(
            model, train_loader, optimiser, device, ssim_metric, epoch, p.epochs
        )

        #validate
        avg_v_loss, avg_v_ssim = evaluate(
            model, val_loader, device, ssim_metric, epoch, p.epochs
        )

        train_loss.append(avg_t_loss)
        train_ssim.append(avg_t_ssim)
        val_losses.append(avg_v_loss)
        val_ssims.append(avg_v_ssim)

       

        #save best 
        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_vqvae.pt")

    torch.save(model.state_dict(), "final_vqvae.pt")
    #plot
    plot_curves(train_loss, val_losses, train_ssim, val_ssims, out_dir="logs")
    save_metrics(train_loss, val_losses, train_ssim, val_ssims, best_epoch, best_v_loss, epochs=p.epochs, out_path="logs/training_metrics.txt")
    show_visualisation(model, p, out_dir="logs", split="validation", n=10)

if __name__ == "__main__":
    #main()
    p = Parameters(profile = "local")
    train_pixelcnn(p=p, epochs=100, save_dir="logs")