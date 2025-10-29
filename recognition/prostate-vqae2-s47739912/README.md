# VQVAE on HipMRI data slices

## Overview
The following project aims to create a generative model using VQVAE on processed 2D slices of hipMRI that achieves a decoded output with "reasonably clear"
images with an SSIM of 0.6 or above. To achieve this first a Vectorised variational auto encoder is trained to achieved reconstructions of a reasonably high ssim, 
then a pixelCNN is trained to create vector quantised vectors / indices which can be decoded into new images.
## Model

### VQVAE
VQVAE (Vectorised Quantised Variational Auto Encoder) is a type of variational autoencoder which use an extra step of vector quantisation to obtain a discrete
latent representation. The architecture for a VQVAE can be seen below:



### PixelCNN
