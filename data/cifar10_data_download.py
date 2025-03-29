#!/usr/bin/env python3
import os
from torchvision import datasets, transforms

def download_and_save_cifar10(data_dir='cifar10_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Downloading CIFAR-10 dataset using PyTorch...")
    # Convert images to tensors
    transform = transforms.ToTensor()
    
    # Download training and test datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    print(f"Training dataset downloaded with {len(train_dataset)} samples.")
    print(f"Test dataset downloaded with {len(test_dataset)} samples.")

if __name__ == '__main__':
    download_and_save_cifar10()
