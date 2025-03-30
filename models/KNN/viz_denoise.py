import os
import torch
import matplotlib.pyplot as plt

# Import your denoise functions from knn_denoise.py
from knn_denoise import denoise_knn_pixel, denoise_knn_patch

def load_cifar_data(device='cpu'):
    """
    Attempt to load clean_images.pt / noisy_images.pt.
    Otherwise, fallback to CIFAR-10 test set and add random Gaussian noise.
    """
    try:
        clean_data = torch.load('data/cifar10_data/clean_images.pt').to(device)
        noisy_data = torch.load('data/cifar10_data/noisy_images.pt').to(device)
        print("Loaded data from .pt files.")
    except:
        print("Falling back to CIFAR-10 test set + random Gaussian noise.")
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()
        cifar_test = datasets.CIFAR10(root='data/cifar10_data/',
                                      train=False, download=True,
                                      transform=transform)
        clean_list = [img for img, _ in cifar_test]
        clean_data = torch.stack(clean_list)
        noise_std = 0.1
        noisy_data = (clean_data + noise_std * torch.randn_like(clean_data)).clamp(0.0, 1.0)
    
    return clean_data.float(), noisy_data.float()

def visualize_20examples_per_experiment(clean_data, noisy_data, device='cpu'):
    """
    For each of the 8 experiments (Pixel/ Patch, K=3/5/7/10),
    create a separate JPG showing 20 images in a 20x3 grid:
       (Clean / Noisy / Denoised).
    => 8 output files (one for each experiment).
    """
    clean_data = clean_data.to(device)
    noisy_data = noisy_data.to(device)

    # 8 scenario definitions
    approach_scenarios = [
        ("pixel", denoise_knn_pixel, 3),
        ("pixel", denoise_knn_pixel, 5),
        ("pixel", denoise_knn_pixel, 7),
        ("pixel", denoise_knn_pixel, 10),
        ("patch", denoise_knn_patch, 3),
        ("patch", denoise_knn_patch, 5),
        ("patch", denoise_knn_patch, 7),
        ("patch", denoise_knn_patch, 10),
    ]

    # Make sure we have a folder for storing visuals
    viz_folder = "models/KNN/visualizations"
    os.makedirs(viz_folder, exist_ok=True)

    N_EXAMPLES = 20
    # e.g. first 20 images
    indices = list(range(N_EXAMPLES))

    for (approach_name, denoise_func, K) in approach_scenarios:
        fig, axes = plt.subplots(nrows=N_EXAMPLES, ncols=3, figsize=(9, N_EXAMPLES*1.5))
        fig.suptitle(f"{approach_name.capitalize()}-level, K={K}", fontsize=14)

        for row, idx in enumerate(indices):
            # Denoise
            if denoise_func is denoise_knn_pixel:
                denoised_img = denoise_func(noisy_data[idx], K, search_radius=3)
            else:
                denoised_img = denoise_func(noisy_data[idx], K, patch_radius=1)

            # Convert to numpy
            clean_np = clean_data[idx].cpu().permute(1,2,0).numpy()
            noisy_np = noisy_data[idx].cpu().permute(1,2,0).numpy()
            den_np   = denoised_img.cpu().permute(1,2,0).numpy()

            # Plot each row as: Clean / Noisy / Denoised
            axes[row, 0].imshow(clean_np)
            if row == 0:
                axes[row, 0].set_title("Clean")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(noisy_np)
            if row == 0:
                axes[row, 1].set_title("Noisy")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(den_np)
            if row == 0:
                axes[row, 2].set_title("Denoised")
            axes[row, 2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_name = f"{approach_name}_K{K}_20examples.jpg"
        out_path = os.path.join(viz_folder, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

def all_experiments_side_by_side(clean_data, noisy_data, device='cpu'):
    """
    Produce ONE large figure with 20 rows x 8 columns.
    Rows = 20 images from the dataset (indices 0..19).
    Columns = 8 experiments:
      (Pixel-level K=3,5,7,10) + (Patch-level K=3,5,7,10).
    Only show the denoised results side by side for each row.
    """
    clean_data = clean_data.to(device)
    noisy_data = noisy_data.to(device)

    approach_scenarios = [
        ("Pix K=3",   denoise_knn_pixel,  3),
        ("Pix K=5",   denoise_knn_pixel,  5),
        ("Pix K=7",   denoise_knn_pixel,  7),
        ("Pix K=10",  denoise_knn_pixel, 10),
        ("Patch K=3", denoise_knn_patch,  3),
        ("Patch K=5", denoise_knn_patch,  5),
        ("Patch K=7", denoise_knn_patch,  7),
        ("Patch K=10",denoise_knn_patch, 10),
    ]

    N_ROWS = 20
    indices = list(range(N_ROWS))

    fig, axes = plt.subplots(nrows=N_ROWS, ncols=8, figsize=(16, 2*N_ROWS))

    for row, idx in enumerate(indices):
        for col, (title, denoise_func, K) in enumerate(approach_scenarios):
            if denoise_func is denoise_knn_pixel:
                denoised_img = denoise_func(noisy_data[idx], K, search_radius=3)
            else:
                denoised_img = denoise_func(noisy_data[idx], K, patch_radius=1)

            den_np = denoised_img.cpu().permute(1,2,0).numpy()

            ax = axes[row, col]
            ax.imshow(den_np)
            ax.axis("off")

            if row == 0:
                ax.set_title(title, fontsize=10)

            if col == 0:
                ax.set_ylabel(f"idx={idx}", rotation=90, labelpad=5, fontsize=9)

    plt.tight_layout()
    out_folder = "models/KNN/visualizations"
    os.makedirs(out_folder, exist_ok=True)

    out_path = os.path.join(out_folder, "all_experiments_side_by_side.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    clean_data, noisy_data = load_cifar_data(device=device)

    # 1) Generate 8 experiment-specific JPGs (20x3 each)
    visualize_20examples_per_experiment(clean_data, noisy_data, device=device)

    # 2) Generate one big side-by-side JPG (20x8)
    all_experiments_side_by_side(clean_data, noisy_data, device=device)

if __name__ == "__main__":
    main()
