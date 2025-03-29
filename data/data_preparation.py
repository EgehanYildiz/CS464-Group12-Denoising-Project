#!/usr/bin/env python3
import os
import numpy as np
from torchvision import datasets, transforms

def add_gaussian_noise(images, sigma=0.1):
    noise = np.random.normal(0, sigma, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

def prepare_data(data_dir='cifar10_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    transform = transforms.ToTensor()
    
    print("Loading CIFAR-10 training and test datasets from cifar10_data...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    
    x_train = np.array([np.transpose(np.array(img), (1, 2, 0)) for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])
    
    x_test = np.array([np.transpose(np.array(img), (1, 2, 0)) for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])
    
    sigma = 0.1  # Standard deviation for Gaussian noise.
    x_train_noisy = add_gaussian_noise(x_train, sigma)
    x_test_noisy = add_gaussian_noise(x_test, sigma)
    
    train_output = os.path.join(data_dir, 'cifar10_train_noisy.npz')
    test_output = os.path.join(data_dir, 'cifar10_test_noisy.npz')
    
    np.savez_compressed(train_output, x_train_clean=x_train, x_train_noisy=x_train_noisy, y_train=y_train)
    np.savez_compressed(test_output, x_test_clean=x_test, x_test_noisy=x_test_noisy, y_test=y_test)
    
    print(f"Noisy training data saved to: {train_output}")
    print(f"Noisy test data saved to: {test_output}")

if __name__ == '__main__':
    prepare_data()
