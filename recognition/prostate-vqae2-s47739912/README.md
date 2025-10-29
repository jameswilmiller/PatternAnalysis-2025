# VQVAE on HipMRI data slices


## Overview
This project implements a Vector quantised variational autoencoder (VQ-VAE) to learn compact, discrete representations
of 2D Hip MRI slices, and a pixelCNN that models the prior over VQ-VAE latents. New images are generated autoregressively sampling
a latent tensor with PixelCNN and decoding it with the VQ-VAE. The following project aims to use a pixelCNN to generate reasonably clear
images using the VQ-VAE which was trained to reconstruct with a target of SSIM > 0.60.

## VQ-VAE
A Vector Quantised Variational Autoencoder (VQ-VAE) is a neural network that learns to compress an image into a smaller, discrete
representation before reconstructing it. The encoder maps the image into a continuous latent space, which then gets quantised by 
replacing each latent vector with the nearest entry from a learned codebook of embedding vectors. This encourages the model to learn
meaningful features. The decoder then deconstructs the image from the codes producing realistic outputs that preserve important structural
details.

(image will go here)

### Architecture
The VQ-VAE uses three convolutional blocks with batch normalisation, ReLU activations and pooling to reduce the image size from 256x256
to a 32x32 latent space with 64 channels. A 1x1 convolution maps the features to the embedding dimension which is then quantised using a 512
entry codebook that replaces each latent vector with its nearest learned embedding. The decoder mirrors the encoder using transposed convolutions
and batch normalisation to upsample the quantised features back to the original image size. A sigmoid output is used to produce normalised reconstructions
between 0 and 1.


## Pixel CNN
A pixelCNN is an autoregressive model that learns to generate data one element at a time predicting each pixel based on previously generated ones.
when used with a VQ-VAE, the pixelCNN is trained on the latent space rather than a raw image, this causes it to learn the relationships between discrete
codes the VQ-VAE has produced. The pixelCNN is then used as a prior model where during image generation the pixelCNN samples a new grid of latent codes
which are decoded using the VQ-VAE into a new realistic looking image

(image will go here)

### Architecture
The pixelCNN learns spacial dependencies between VQ-VAE embeddings. It begins with a masked convolution(type A) to enforce
autoregressive ordering, followed by 5 residual blocks combining 1x1 and 3x3 convolutions. The final layers project the output to match
the VQ-VAE 64 dimensional latent channels. The model is trained using MSE loss, the model predicts and samples latent grids that the VQ-VAE
can decode into new imagess.

## Data and Preprocessing
The dataset consists of 2D prostate MRI slices stored as NIfTI (.nii.gz) files. each slices was loaded, converted to grayscale and
resized to 256x256 to maintain consistency. Pixel values were then min-max normalised to 0-1. The data was given pre split for training,
validation and testing so no further splitting needed to be done before training.

## Training Performance







