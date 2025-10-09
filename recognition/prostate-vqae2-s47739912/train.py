import torch
from modules import *
from dataset import *
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid






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
        recon, codebook_loss, commitment_loss = model(batch)
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
        val_loss += loss.item()
        val_ssim = ssim_metric(recon, batch).item()

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
        recon, codebook_loss, commitment_loss = model(batch)

        #loss + step
        loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
        loss.backward()
        optimiser.step()

        t_loss += loss.item()
        t_ssim += ssim_metric(recon, batch).item()

    avg_t_loss = t_loss / len(train_loader)
    avg_t_ssim = t_ssim / len(train_loader)

    return avg_t_loss, avg_t_ssim
def main():
    #build params
    p = Parameters(profile="local")

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

        print(
            f"Epoch {epoch+1}/{p.epochs} | "
            f"train_loss {avg_t_loss:.4f}  val_loss {avg_val_loss:.4f} | "
            f"train_ssim {avg_t_ssim:.4f}  val_ssim {avg_val_ssim:.4f}"
        )

        #save best 
        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            torch.save(model.state_dict(), "best_vqae.pt")
    #plot
    
       
if __name__ == "__main__":
    main()