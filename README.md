# Restoring Quality Images by Using Denoising Autoencoders

## Overview
This project aims to restore the quality of CIFAR-10 images by removing synthetic Gaussian noise. We implement five different models:

- **Convolutional Denoising Autoencoder**
- **Variational Autoencoder (VAE)**
- **MLP-based Autoencoder**
- **KNN-based Denoising Filter**
- **UNet-based Denoising Model**

Each model takes noisy images as input and outputs cleaned images. The performance is evaluated using metrics like MSE, PSNR, and SSIM.

## Repository Structure
