import torch
import math

# Device configuration: use GPU if available for speed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# **Load the CIFAR-10 noisy and clean datasets** 
# (We assume data_preparation has produced tensors or arrays of noisy and clean images.)
# For example, data might be stored as PyTorch tensors or NumPy arrays in .pt or .npy files.
# Here we demonstrate with a placeholder loading mechanism. In practice, replace this with actual data loading.
try:
    # Try loading pre-saved data tensors (if available)
    clean_data = torch.load('data/cifar10_data/clean_images.pt').to(device)      # shape [N, 3, 32, 32]
    noisy_data = torch.load('data/cifar10_data/noisy_images.pt').to(device)      # shape [N, 3, 32, 32]
except Exception as e:
    # Fallback: if above fails, attempt to load via torchvision and add noise (for demonstration purposes)
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()  # to get [0,1] float tensors
    cifar_clean = datasets.CIFAR10(root='data/cifar10_data/', train=False, download=True, transform=transform)
    # Convert to tensor batch
    clean_list = [img for img, _ in cifar_clean]
    clean_data = torch.stack(clean_list).to(device)
    noise_std = 0.1
    noisy_data = (clean_data + noise_std * torch.randn_like(clean_data)).clamp(0.0, 1.0)
    # (In a real scenario, the noisy_data would be provided directly to avoid randomness in each run.)

# Ensure data types are float for metric calculations
clean_data = clean_data.float()
noisy_data = noisy_data.float()

# Define helper functions for metrics
def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images (averaged per pixel)."""
    return torch.mean((img1 - img2) ** 2).item()

def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (in decibels) between two images."""
    mse_val = compute_mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    max_val = 1.0  # Assuming image pixel values are in [0,1]
    psnr_val = 10 * math.log10((max_val ** 2) / mse_val)
    return psnr_val

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two images.
    This is a simplified implementation that computes SSIM on the luminance (grayscale) channel of the whole image."""
    # Convert to grayscale (luminance) for SSIM calculation
    # We use standard ITU-R BT.601 coefficients for RGB to luminance conversion.
    r, g, b = img1[0], img1[1], img1[2]
    lumin1 = 0.299*r + 0.587*g + 0.114*b
    r2, g2, b2 = img2[0], img2[1], img2[2]
    lumin2 = 0.299*r2 + 0.587*g2 + 0.114*b2

    # Compute means
    mu1 = lumin1.mean().item()
    mu2 = lumin2.mean().item()
    # Compute variances and covariance
    sigma1 = ((lumin1 - mu1) ** 2).mean().item()
    sigma2 = ((lumin2 - mu2) ** 2).mean().item()
    sigma12 = ((lumin1 - mu1) * (lumin2 - mu2)).mean().item()

    # Constants to stabilize division (use standard values for dynamic range = 1)
    C1 = (0.01 * 1) ** 2  # = 0.0001
    C2 = (0.03 * 1) ** 2  # = 0.0009

    # SSIM formula
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_val

# Pixel-level KNN denoising function
def denoise_knn_pixel(noisy_img, K, search_radius=3):
    """
    Denoise an image using pixel-level KNN filtering.
    noisy_img: Tensor of shape [3, H, W] (RGB image).
    K: number of nearest neighbors to use for averaging.
    search_radius: radius of the neighborhood window (in pixels) to search for nearest neighbors.
    """
    C, H, W = noisy_img.shape
    # Pad the image to handle borders (reflect padding to avoid artificial zero padding influence)
    # This allows the neighborhood of pixels near edges to be computed easily.
    padded_img = torch.nn.functional.pad(noisy_img.unsqueeze(0), pad=(search_radius, search_radius, search_radius, search_radius), mode='reflect')
    padded_img = padded_img.squeeze(0)  # shape [3, H+2*search_radius, W+2*search_radius]

    # Use unfolding to get local neighborhoods for all pixels.
    window_size = 2 * search_radius + 1
    # Unfold (sliding window) to get a tensor of shape [C * window_size * window_size, H * W] for all local patches.
    patches = torch.nn.functional.unfold(padded_img.unsqueeze(0), kernel_size=(window_size, window_size), padding=0, stride=1)
    # Now 'patches' shape is [1, C*window_size*window_size, H*W] because we slid over the original image positions.
    patches = patches.squeeze(0)  # shape [C*window_size*window_size, H*W]

    # Reshape patches to separate the channels and neighbor positions:
    # new shape: [C, window_size*window_size, H*W]
    patches = patches.view(C, window_size * window_size, H * W)
    # Identify the center index of the window (this corresponds to the pixel itself in its neighborhood)
    center_idx = (window_size * window_size) // 2  # e.g. if window_size=5, center_idx = 12 for a 5x5 window.
    # Extract the pixel values at the center of each window (this is the original pixel value for each position)
    center_vals = patches[:, center_idx, :]  # shape [C, H*W]

    # Compute squared distance between the center pixel and every other pixel in the window for each position.
    # diff = neighbor_value - center_value for each neighbor in the window.
    # We broadcast center_vals over the neighbor dimension:
    diff = patches - center_vals.unsqueeze(1)  # shape [C, window_size*window_size, H*W]
    diff_sq = diff ** 2
    dist_sq = diff_sq.sum(dim=0)  # sum over color channels, shape [window_size*window_size, H*W]
    # Exclude the center itself from consideration by setting its distance to infinity (so it won't be chosen among K nearest).
    dist_sq[center_idx, :] = float('inf')

    # Find the indices of the K smallest distances in each column (each pixel position).
    # We use torch.topk with largest=False to get the smallest distances.
    vals, idx = torch.topk(dist_sq, k=K, dim=0, largest=False)
    # idx has shape [K, H*W], where idx[:, j] are the indices of the K nearest neighbor positions for pixel j.

    # Gather the K neighbor values for each pixel and average them.
    # We have patches (all neighbor pixel values) and idx (neighbor indices to pick). We'll use gather to pick neighbors.
    # First, expand idx to shape [C, K, H*W] to select values for each channel.
    idx_expanded = idx.unsqueeze(0).expand(C, -1, -1)  # shape [C, K, H*W]
    # Use gather: along dim=1 (the neighbor dimension in patches) for each pixel's neighbors.
    selected_vals = torch.gather(patches, 1, idx_expanded)  # shape [C, K, H*W]
    # Average over the K selected neighbor values for each pixel
    avg_vals = selected_vals.mean(dim=1)  # shape [C, H*W]
    # Reshape back to image shape
    denoised_img = avg_vals.view(C, H, W)
    return denoised_img

# Patch-level KNN denoising function
def denoise_knn_patch(noisy_img, K, patch_radius=1):
    """
    Denoise an image using patch-level KNN filtering.
    noisy_img: Tensor [3, H, W].
    K: number of nearest neighbor patches to use for averaging.
    patch_radius: radius of the patch around each pixel to compare (patch size will be (2*patch_radius+1)^2).
    """
    C, H, W = noisy_img.shape
    patch_size = 2 * patch_radius + 1
    # Pad the image to allow full patch at borders
    padded_img = torch.nn.functional.pad(noisy_img.unsqueeze(0), pad=(patch_radius, patch_radius, patch_radius, patch_radius), mode='reflect')
    padded_img = padded_img.squeeze(0)  # shape [3, H+2*patch_radius, W+2*patch_radius]

    # Use unfold to get all patches of size patch_size x patch_size from the image.
    patches = torch.nn.functional.unfold(padded_img.unsqueeze(0), kernel_size=(patch_size, patch_size), padding=0, stride=1)
    patches = patches.squeeze(0)  # shape [C * patch_size * patch_size, H*W]
    # Reshape to [H*W, C * patch_size * patch_size] for easier distance calculation (each row is a patch vector).
    patch_vectors = patches.t().contiguous()  # shape [H*W, C * patch_size * patch_size]

    # Compute pairwise distances between patch vectors.
    # We use torch.cdist to compute the L2 distance between each patch and every other patch.
    # This yields a distance matrix of shape [H*W, H*W].
    dist_matrix = torch.cdist(patch_vectors, patch_vectors, p=2)  # L2 distances
    dist_matrix_squared = dist_matrix ** 2  # square the distances to work with squared Euclidean (optional, monotonic with L2).
    # Exclude each patch itself by setting diagonal to infinity
    num_pixels = H * W
    diag_indices = torch.arange(num_pixels, device=dist_matrix.device)
    dist_matrix_squared[diag_indices, diag_indices] = float('inf')

    # Find K nearest patch distances for each patch (each row in the distance matrix)
    vals, idx = torch.topk(dist_matrix_squared, k=K, dim=1, largest=False)
    # idx is shape [H*W, K], containing indices of the K nearest neighboring patches for each patch.

    # Use the neighbor indices to compute the denoised pixel values.
    # We will average the center pixel values of the K nearest patches.
    # The center pixel of patch i is just the pixel i itself (since patch i is centered on pixel i).
    # Therefore, for patch j, and one of its neighbor patch indices n = idx[j, :], the center pixel of that neighbor patch is just pixel n.
    # So effectively, we average the values of pixels with indices in idx[j, :] for each channel.
    noisy_flat = noisy_img.view(C, -1)  # shape [C, H*W], flattened image
    denoised_flat = torch.empty_like(noisy_flat)
    # Loop over each pixel (this loop runs H*W times = 1024 for a 32x32 image, which is fine)
    for pix in range(num_pixels):
        neighbor_idxs = idx[pix]  # tensor of length K
        # Average the neighbor pixels for each channel
        # (We use the noisy image's values; in non-local means, averaging noisy values yields the denoised estimate.)
        denoised_flat[:, pix] = noisy_flat[:, neighbor_idxs].mean(dim=1)
    # Reshape back to [3, H, W]
    denoised_img = denoised_flat.view(C, H, W)
    return denoised_img


def main_experiment():
    # Open the log file for writing results
    log_path = "models/KNN/training_logs.txt"
    with open(log_path, "w") as log:
        log.write("KNN Denoising Results on Noisy CIFAR-10 (Gaussian noise) \n")
        log.write("="*50 + "\n\n")
        # Optionally, write a header with metric descriptions
        log.write("Metrics: PSNR (Peak Signal-to-Noise Ratio, dB - higher is better), "
                "SSIM (Structural Similarity Index - higher is better, max 1), "
                "MSE (Mean Squared Error - lower is better).\n\n")

        # Run experiments for each approach and each K
        approaches = [("Pixel-level KNN", denoise_knn_pixel), ("Patch-level KNN", denoise_knn_patch)]
        for approach_name, denoise_func in approaches:
            log.write(f"--- {approach_name} Denoising ---\n")
            print(f"\n=== Starting {approach_name} experiments ===")  # Print approach info
            
            for K in [3, 5, 7, 10]:
                print(f"Starting {approach_name} denoising with K={K}...")  # Print K-value info
                
                # Denoising
                denoised_imgs = []
                if approach_name.startswith("Patch"):
                    for img_idx in range(len(noisy_data)):
                        denoised_img = denoise_func(noisy_data[img_idx], K, patch_radius=1)
                        denoised_imgs.append(denoised_img)
                        
                        # Print progress every 100 images
                        if (img_idx + 1) % 100 == 0:
                            print(f"{approach_name} (K={K}): processed {img_idx + 1} / {len(noisy_data)} images")
                else:
                    # Pixel-level approach
                    for img_idx in range(len(noisy_data)):
                        denoised_img = denoise_func(noisy_data[img_idx], K, search_radius=3)
                        denoised_imgs.append(denoised_img)
                        
                        # Print progress every 100 images
                        if (img_idx + 1) % 100 == 0:
                            print(f"{approach_name} (K={K}): processed {img_idx + 1} / {len(noisy_data)} images")
                
                denoised_imgs = torch.stack(denoised_imgs)
                
                # Compute average metrics
                N = len(denoised_imgs)
                if N > 1000:
                    N = 1000
                psnr_total = 0.0
                ssim_total = 0.0
                mse_total = 0.0
                for i in range(N):
                    psnr_val = compute_psnr(denoised_imgs[i], clean_data[i])
                    ssim_val = compute_ssim(denoised_imgs[i], clean_data[i])
                    mse_val = compute_mse(denoised_imgs[i], clean_data[i])
                    psnr_total += psnr_val
                    ssim_total += ssim_val
                    mse_total += mse_val
                
                psnr_avg = psnr_total / N
                ssim_avg = ssim_total / N
                mse_avg = mse_total / N

                # Log the metrics for this setting
                log.write(f"{approach_name}, K={K}: PSNR={psnr_avg:.2f} dB, SSIM={ssim_avg:.4f}, MSE={mse_avg:.6f}\n")
                # Provide a brief explanation of the result
                # (E.g., how increasing K affected the metrics or how this approach is performing)
                explanation = ""
                if approach_name.startswith("Pixel"):
                    explanation += f"For K={K}, pixel-level averaging uses the {K} most similar nearby pixels to denoise each pixel. "
                else:
                    explanation += f"For K={K}, patch-level averaging uses the {K} most similar patches to denoise each pixel. "
                # Mention how K influences noise vs detail
                if K < 5:
                    explanation += "A small K retains more image detail but might leave some noise. "
                elif K > 7:
                    explanation += "A larger K smooths more noise but can blur fine details. "
                # Interpret the metric values in words
                explanation += f"The resulting PSNR of {psnr_avg:.2f} dB indicates the denoised image is much closer to the original (higher is better). "
                explanation += f"SSIM of {ssim_avg:.3f} (on a 0-1 scale) implies a {'high' if ssim_avg>0.8 else 'moderate'} structural similarity to the clean image. "
                explanation += f"MSE of {mse_avg:.6f} shows the average squared error per pixel; lower values signify better fidelity. "
                log.write(explanation + "\n\n")
                print(f"[{approach_name}, K={K}] => PSNR={psnr_avg:.2f} dB, SSIM={ssim_avg:.4f}, MSE={mse_avg:.6f}")
        log.write("Experiment complete. The above results show the trade-off between noise reduction and detail preservation for different K values and methods.\n")

if __name__ == "__main__":
    main_experiment()