import torch
from modules import *
from dataset import *


device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
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


