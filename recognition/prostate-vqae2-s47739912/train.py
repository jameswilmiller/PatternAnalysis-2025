import torch
from modules import *
from dataset import *

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

p = Parameters()

dataLoader = KerasSlicesDataLoader(TRAIN_DIR, VAL_DIR, TEST_DIR, size=(256,256), norm=True)
train_loader = dataLoader.get_train()
validation_loader = dataLoader.get_validation()
test_loader = dataLoader.get_test()

vae = CNNVAE(latent_dim=64).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=p.learning_rate)

#training conf
epochs = 10
beta = 1.0

print("training VAE...")


for epoch in range(epochs):
    vae.train()
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        #forward
        recon_images, mu, logvar = vae(images)

        #loss
        loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar, 0.5)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
        
    avg_loss =total_loss / len(train_loader.dataset)
    avg_bce = total_bce / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)

