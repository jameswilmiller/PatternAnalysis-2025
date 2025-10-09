import torch
from modules import *
from dataset import *
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

#

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
    epochs = 10
    learning_rate = 1e-4
    embedding_dim = 64
    num_embeddings = 512
    beta = 0.25

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


    
