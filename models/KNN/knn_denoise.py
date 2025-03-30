#!/usr/bin/env python3
import os
import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import datasets, transforms
import torch

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(images, sigma=0.1):
    """
    Add Gaussian noise to images.
    
    Args:
        images (numpy.array): Array of images normalized in [0,1].
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy.array: Noisy images, clipped to [0,1].
    """
    noise = np.random.normal(0, sigma, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

def knn_denoise_image(image, K=5, patch_size=3, window_size=7):
    """
    Apply KNN-based denoising to a single image using GPU acceleration.

    Args:
        image (np.array): Noisy image with shape (H, W, C) normalized in [0,1].
        K (int): Number of nearest neighbor patches to average.
        patch_size (int): Size of the patch (odd number).
        window_size (int): Size of the search window (odd number).

    Returns:
        np.array: Denoised image with the same shape as input.
    """
    H, W, C = image.shape
    pad = patch_size // 2
    pad_w = window_size // 2

    # Convert NumPy array to PyTorch tensor and move to GPU
    image_tensor = torch.from_numpy(image).to(device)

    # Pad the image with reflection
    padded = torch.nn.functional.pad(
        image_tensor.permute(2, 0, 1)[None, ...], 
        (pad, pad, pad, pad), 
        mode='reflect'
    )[0].permute(1, 2, 0)

    # Extract all patches
    patches = torch.as_strided(
        padded,
        size=(H, W, patch_size, patch_size, C),
        stride=(padded.stride(0), padded.stride(1), padded.stride(0), padded.stride(1), padded.stride(2))
    )

    # Define search window offsets
    num_di = 2 * pad_w + 1
    num_dj = 2 * pad_w + 1
    di, dj = torch.meshgrid(
        torch.arange(-pad_w, pad_w + 1, device=device),
        torch.arange(-pad_w, pad_w + 1, device=device),
        indexing='ij'
    )

    # Compute candidate indices
    i_indices = torch.arange(H, device=device).view(-1, 1, 1, 1)
    j_indices = torch.arange(W, device=device).view(1, -1, 1, 1)
    i_candidates = i_indices + di[None, None, :, :]
    j_candidates = j_indices + dj[None, None, :, :]
    i_candidates_clamped = torch.clamp(i_candidates, 0, H - 1)
    j_candidates_clamped = torch.clamp(j_candidates, 0, W - 1)

    # Extract candidate patches
    candidate_patches = patches[i_candidates_clamped, j_candidates_clamped]

    # Compute distances
    ref_patches = patches[:, :, None, None, :, :, :]
    diff = candidate_patches - ref_patches
    distances = (diff ** 2).sum(dim=[-3, -2, -1])

    # Find K nearest neighbors
    num_candidates = num_di * num_dj
    distances_flat = distances.view(H, W, num_candidates)
    _, topk_indices = torch.topk(distances_flat, K, dim=2, largest=False)

    # Get indices of K nearest patches
    i_candidates_flat = i_candidates_clamped.view(H, W, num_candidates)
    j_candidates_flat = j_candidates_clamped.view(H, W, num_candidates)
    ii = torch.gather(i_candidates_flat, dim=2, index=topk_indices)
    jj = torch.gather(j_candidates_flat, dim=2, index=topk_indices)

    # Compute denoised image
    center_pixels = image_tensor[ii, jj, :]
    denoised = center_pixels.mean(dim=2)

    return denoised.cpu().numpy()

def denoise_dataset_for_k(npz_file, K, patch_size=3, window_size=7):
    """
    Apply KNN denoising to the entire test dataset from the NPZ file.
    
    Args:
        npz_file (str): Path to the input NPZ file (e.g., cifar10_test_noisy.npz).
        K (int): Number of nearest neighbors.
        patch_size (int): Size of patch.
        window_size (int): Size of search window.
    
    Returns:
        denoised_images (np.array): Array of denoised images.
        x_clean (np.array): Array of clean images from the NPZ file.
    """
    data = np.load(npz_file)
    x_noisy = data['x_test_noisy']
    x_clean = data['x_test_clean'] if 'x_test_clean' in data else None
    num_images = x_noisy.shape[0]
    denoised_images = []
    
    print(f"Processing {num_images} images for KNN denoising with K={K}...")
    for idx in range(num_images):
        img = x_noisy[idx]
        denoised = knn_denoise_image(img, K=K, patch_size=patch_size, window_size=window_size)
        denoised_images.append(denoised)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_images} images")
    denoised_images = np.array(denoised_images)
    return denoised_images, x_clean

def compute_metrics(clean_images, denoised_images):
    """
    Compute average MSE, PSNR, and SSIM between clean and denoised images.
    
    Args:
        clean_images (np.array): Clean images array of shape (N, H, W, C).
        denoised_images (np.array): Denoised images array of same shape.
    
    Returns:
        mse (float), psnr (float), ssim (float): Average metrics over all images.
    """
    mse_values = []
    ssim_values = []
    N = clean_images.shape[0]
    
    for i in range(N):
        mse = np.mean((clean_images[i] - denoised_images[i])**2)
        mse_values.append(mse)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        ssim_val = compare_ssim(clean_images[i], denoised_images[i], multichannel=True)
        ssim_values.append(ssim_val)
    
    avg_mse = np.mean(mse_values)
    avg_psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else float('inf')
    avg_ssim = np.mean(ssim_values)
    return avg_mse, avg_psnr, avg_ssim

def run_experiments():
    # Input NPZ file produced by data_preparation.py
    input_npz = os.path.join('cifar10_data', 'cifar10_test_noisy.npz')
    output_dir = 'cifar10_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define K values to experiment with
    k_values = [3, 5, 7, 10, 15]
    patch_size = 3
    window_size = 7

    results = {}

    for K in k_values:
        print(f"\n=== Running KNN denoising with K = {K} ===")
        denoised_images, x_clean = denoise_dataset_for_k(input_npz, K, patch_size, window_size)
        output_npz = os.path.join(output_dir, f'cifar10_test_denoised_knn_K{K}.npz')
        if x_clean is not None:
            np.savez_compressed(output_npz, x_test_denoised=denoised_images, x_test_clean=x_clean)
        else:
            np.savez_compressed(output_npz, x_test_denoised=denoised_images)
        print(f"Denoised dataset saved to: {output_npz}")

        if x_clean is not None:
            mse, psnr, ssim = compute_metrics(x_clean, denoised_images)
            results[K] = {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim}
            print(f"Results for K={K}: MSE = {mse:.6f}, PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
        else:
            results[K] = None

    # Save experiment results in a training log
    log_file = os.path.join('models', 'KNN', 'training_log.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write("KNN Denoising Experiment Results\n")
        f.write("================================\n\n")
        for K in k_values:
            if results[K] is not None:
                f.write(f"K = {K}:\n")
                f.write(f"  MSE:  {results[K]['MSE']:.6f}\n")
                f.write(f"  PSNR: {results[K]['PSNR']:.2f} dB\n")
                f.write(f"  SSIM: {results[K]['SSIM']:.4f}\n\n")
            else:
                f.write(f"K = {K}: No clean images available for evaluation.\n\n")
    print(f"\nExperiment log saved to: {log_file}")

if __name__ == '__main__':
    run_experiments()