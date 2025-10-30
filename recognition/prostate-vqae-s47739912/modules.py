import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class Parameters():
    def __init__(self, profile="local"):
        self.profile = profile #change to "rangpur if using rangpur"
        #data roots
        rangpur_base_dir = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")       

        #this is the directory for other PC i have been training on
        #local_base_dir = Path("C:/Users/DadsDDR5/OneDrive/Documents/james stuff/PatternAnalysis-2025/data/keras_slices_data")

        local_base_dir = Path("C:/Users/itbmi/OneDrive/Documents/UQ-Work/pattern recognition/report/PatternAnalysis-2025/data/keras_slices_data")
 
        if profile == "rangpur":
            self.base_dir = rangpur_base_dir
        else:
            self.base_dir = local_base_dir

        self.train_dir = self.base_dir / "keras_slices_train"
        self.val_dir = self.base_dir / "keras_slices_validate"
        self.test_dir = self.base_dir / "keras_slices_test"

        #data
        self.img_size = (256, 256)
        self.normalise = True
        
        #model
        self.embedding_dim = 64
        self.num_embeddings = 512
        self.beta = 0.25

        #training
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 100
        self.recon_weight = 1.0

        #misc
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
      


class VectorQuantiser(nn.Module):
    def __init__(self, num_embedding, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost

        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embedding, 1.0 / self.num_embedding)

    def forward(self, x):
        #where x is batch size , channels, height, width
        #originals
        batch, chan, height, width = x.shape

        x_perm = x.permute(0, 2, 3,1).contiguous()
        flattened_x = x_perm.view(-1, self.embedding_dim)

        #squared L2 dist to codebook
        codebook = self.embedding.weight
        dists = (
            flattened_x.pow(2).sum(1, keepdim=True)
            - 2 * flattened_x @ codebook.t()
            + codebook.pow(2).sum(1)
        )

        #nearest code index 
        indices = torch.argmin(dists, dim=1)
        nearest = self.embedding(indices)

        #reshape back
        quantised_permuted = nearest.view(batch, height, width, chan)
        quantised = quantised_permuted.permute(0, 3, 1, 2).contiguous()

        #loss
        codebook_loss = F.mse_loss(quantised_permuted.detach(), x_perm)
        commitment_loss = self.beta * F.mse_loss(x_perm.detach(), quantised_permuted)

        quantised_straight = x + (quantised - x).detach()

        unflattened_indices = indices.view(batch, height, width)
        return quantised_straight, codebook_loss, commitment_loss, unflattened_indices
        

class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, beta=0.5):
        super().__init__()
       
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  #256 -> 128
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), #128 -> 64

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), #64 -> 32
        )
        self.to_embed = nn.Conv2d(128, embedding_dim, kernel_size = 1)

        self.vq = VectorQuantiser(num_embedding=num_embeddings, embedding_dim = embedding_dim, commitment_cost=beta)

        #decoder
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(embedding_dim, 128, 3, stride=2, padding=1, output_padding=1), #16 > 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), #32 > 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #64 > 128
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, padding=1),
              #128 > 256
            nn.Sigmoid(), #output in [0, 1] for img reconstruction
        )
        
    def encode(self, x):
        return self.to_embed(self.encoder(x))
    
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterise(self, code):
        return self.vq(code)
    
    def forward(self, x):
        code = self.encode(x)
        quantised, codebook_loss, commitment_loss, indices = self.reparameterise(code)
        recon = self.decode(quantised)
        return recon, codebook_loss, commitment_loss, indices
    

class CNNConv2d(nn.Conv2d):
    def __init__(self, mask_type, initial_channels, out_channels, kernel_size, **kwargs):
        super().__init__(initial_channels, out_channels, kernel_size, **kwargs)
        #build mask

        c = kernel_size // 2 #centers the index
        mask = torch.ones_like(self.weight)
        #block all rows below the center
        mask[:, :, c+1:, :] = 0

        mask[:, :, c, c+1:] = 0
        if mask_type == 'A':
            mask[:, :, c, c] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class PixelCNNresBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CNNConv2d("B", channels // 2, channels // 2, kernel_size =3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return x + out
    
class PixelCNN(nn.Module):
    def __init__(self, init_channel = 128, channels = 128, out_channel=128, num_resid=5):
        super().__init__()
        #intial layer (masked A)
        self.s = nn.Sequential(
            CNNConv2d("A", init_channel, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[PixelCNNresBlock(channels) for _ in range(num_resid)]
        )

        self.head = nn.Sequential(
           CNNConv2d("B", channels, channels, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           CNNConv2d("B", channels, channels, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv2d(channels, out_channel, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.s(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x
    
