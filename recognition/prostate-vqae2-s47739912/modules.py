import torch
import torch.nn as nn
import torch.nn.functional as f

class Parameters():
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.embedding_dimension = 64


class CNNVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  #256 -> 128
            
            nn.conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) #128 -> 64

            nn.conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) #64 -> 32

            nn.conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.maxPool2d(2), #32 -> 16
            
            nn.conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), #16 -> 8

            nn.Flatten(),
        )
        #latent space params
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim) #mean
        self.log_var = nn.Linear(256 * 8 * 8, latent_dim) #log variance

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU,
            nn.UnFlatten(1, (256, 8, 8)) 

            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1), #8 > 16
            nn.BatchNorm2d(256), 
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), #16 > 32
            nn.BatchNorm2d(128),
            nn.RelU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), #32 > 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #64 > 128
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), #128 > 256
            nn.BatchNorm2d(32),
            nn.Sigmoid(), #output in [0, 1] for img reconstruction
        )
