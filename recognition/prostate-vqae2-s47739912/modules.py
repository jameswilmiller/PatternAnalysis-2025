import torch
import torch.nn as nn
import torch.nn.functional as F

class Parameters():
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.embedding_dimension = 64
        
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

        x = x.permute(0, 2, 3,1).contiguous()
        flattened_x = x.view(-1, self.embedding_dim)

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
        quantised = nearest.view(batch, height, width, chan)
        quantised = quantised.permute(0, 3, 1, 2).contiguous()

        #loss
        codebook_loss = F.mse_loss(quantised, x.detach())
        commitment_loss = self.beta * F.mse_loss(x, quantised.detach())

        quantised = x + (quantised - x).detach()

        return quantised, codebook_loss, commitment_loss 
        


    
        





class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, beta=0.5):
        super().__init__()
        initial_channels = 1
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

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), #32 -> 16
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), #16 -> 8
        )
        self.to_embed = nn.Conv2d(256, embedding_dim, kernel_size = 1)

        self.vq = VectorQuantiser(num_embedding=num_embeddings, embedding_dim = embedding_dim, commitment_cost=beta)

        #decoder
        self.decoder = nn.Sequential(
            

            nn.ConvTranspose2d(embedding_dim, 256, 3, stride=2, padding=1, output_padding=1), #8 > 16
            nn.BatchNorm2d(256), 
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), #16 > 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), #32 > 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #64 > 128
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
              #128 > 256
            nn.Sigmoid(), #output in [0, 1] for img reconstruction
        )
        
    def encode(self, x):
        encoder_feat = self.encoder(x)
        code_feat = self.to_embed(encoder_feat)
        return code_feat
    
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterise(self, code):
        quantised, codebook_loss, commitment_loss = self.vq(code)
        return quantised, codebook_loss, commitment_loss
    def forward(self, x):
        code = self.encode(x)
        quantised, codebook_loss, commitment_loss = self.reparameterise(code)
        recon = self.decode(quantised)
        return recon, codebook_loss, commitment_loss
    
