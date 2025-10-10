import torch
from modules import *
from dataset import *
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from itertools import zip_longest





def loss_function(recon, target, codebook_loss, commitment_loss):
    recon_loss = F.mse_loss(recon, target)
    return  recon_loss + codebook_loss + commitment_loss, recon_loss

#train
@torch.no_grad()
def evaluate(model, val_loader, device, ssim_metric):
    """
    Evaluate the model on the validation loader
    returns avg_val_loss and avg_val_ssim
    """
    model.eval()
    val_loss = 0.0
    val_ssim = 0.0

    for batch in tqdm(val_loader, leave=False):
        batch = batch.to(device)
        recon, codebook_loss, commitment_loss, indices= model(batch)
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
        val_loss += loss.item()
        val_ssim += ssim_metric(recon, batch).item()

    avg_v_loss = val_loss / len(val_loader)
    avg_v_ssim = val_ssim / len(val_loader)
    return avg_v_loss, avg_v_ssim


def train_epoch(model, train_loader, optimiser, device, ssim_metric):
    """
    trains the model for a single epoch
    """
    model.train()
    t_loss = 0.0
    t_ssim = 0.0

    for batch in tqdm(train_loader, leave=False):
        optimiser.zero_grad()
        batch = batch.to(device)

        #forward
        recon, codebook_loss, commitment_loss, indices = model(batch)

        #loss + step
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
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
            f.write(f"{epoch}\t{tl}\t{vl}\t{ts}\t{vs}\n")
    


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
    p = Parameters(profile="rangpur")

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
            model, train_loader, optimiser, device, ssim_metric
        )

        #validate
        avg_v_loss, avg_v_ssim = evaluate(
            model, val_loader, device, ssim_metric
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
    #plot
    plot_curves(train_loss, val_losses, train_ssim, val_ssims, out_dir="logs")
    save_metrics(train_loss, val_losses, train_ssim, val_ssims, best_epoch, best_v_loss, out_path="logs/training_metrics.txt")


if __name__ == "__main__":
    main()