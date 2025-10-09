import torch
from modules import *
from dataset import *
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")



def loss_function(recon, target, codebook_loss, commitment_loss, recon_weight=1.0):
    recon_loss = F.mse_loss(recon, target)
    return recon_weight * recon_loss + codebook_loss + commitment_loss, recon_loss

#train

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("warning: CUDA not found. Using cpu")

    #params
    BATCH_SIZE = 32
    

    loaders = KerasSlicesDataLoader(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        size=(256,256),
        norm=True
    )
    train_loader = loaders.get_train()
    val_loader = loaders.get_validation()
    test_loader = loaders.get_test()

    #model
    model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, beta=beta).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


    train_loss, val_losses = [], []
    train_ssim, val_ssims = [], []


    for epoch in range(epochs):
        model.train()
        t_loss = 0
        t_ssim = 0

        for batch in tqdm(train_loader):
            batch = batch.to(device)
            
            #forward
            recon, codebook_loss, commitment_loss = model(batch)

            #loss
            loss, recon_loss = loss_function(recon, batch, codebook_loss, commitment_loss)
            loss.backward()
            optimiser.step()


            t_loss += loss.item()
            t_ssim += ssim_metric(recon, batch).item()

        avg_t_loss = t_loss / len(train_loader)
        avg_t_ssim = t_ssim / len(train_loader)

        train_loss.append(avg_t_loss)
        train_ssim.append(avg_t_ssim)


        #validate
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
      

        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                recon, codebook_loss, commitment_loss = model(batch)
                loss, _ = loss_function(recon, batch, codebook_loss, commitment_loss)
                val_loss += loss.item()
                val_ssim += ssim_metric(recon, batch).item()
                last_val_batch = (batch, recon)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ssims.append(avg_val_ssim)

if __name__ == "__main__":
    main()