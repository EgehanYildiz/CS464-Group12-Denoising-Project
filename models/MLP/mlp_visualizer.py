import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

from mlp_autoencoder import MLP_Autoencoder

def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if it's a full checkpoint or just model state
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
    else:
        model_state = checkpoint  # Assume it's just the model weights
        
    model = MLP_Autoencoder()
    model.load_state_dict(model_state)
    return model.to(device)

def load_clean_and_noisy_images(device='cpu'):
    transform = transforms.ToTensor()
    cifar_test = datasets.CIFAR10(root='data/cifar10_data/', train=False, download=True, transform=transform)
    
    clean_images = []
    for i in range(20):
        img, _ = cifar_test[i]
        clean_images.append(img)
    clean_images = torch.stack(clean_images).to(device)

    # Apply Gaussian noise
    noise_std = 0.1
    noisy_images = (clean_images + noise_std * torch.randn_like(clean_images)).clamp(0.0, 1.0)

    return clean_images, noisy_images

def visualize_denoised_examples(model, clean_images, noisy_images, save_path):
    model.eval()
    with torch.no_grad():
        # Ensure proper input shape (batch_size, 3072)
        inputs = noisy_images.view(noisy_images.size(0), -1)
        outputs = model(inputs).view(-1, 3, 32, 32)
        
        print("Output statistics:")
        print("Mean:", outputs.mean().item())
        print("Std:", outputs.std().item())
        print("Min:", outputs.min().item())
        print("Max:", outputs.max().item())

    fig, axes = plt.subplots(nrows=20, ncols=3, figsize=(9, 30))
    fig.suptitle("MLP Denoising Results", fontsize=16, y=0.99)
    
    for i in range(20):
        # Convert tensors to HWC format for matplotlib
        clean = clean_images[i].cpu().permute(1, 2, 0).numpy()
        noisy = noisy_images[i].cpu().permute(1, 2, 0).numpy()
        denoised = outputs[i].cpu().permute(1, 2, 0).numpy()
        
        # Clip to valid range [0,1] just in case
        denoised = np.clip(denoised, 0, 1)

        axes[i, 0].imshow(clean)
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title("Clean")

        axes[i, 1].imshow(noisy)
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title("Noisy")

        axes[i, 2].imshow(denoised)
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title("Denoised")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try loading different checkpoint formats
    checkpoint_paths = [
        'models/MLP/checkpoints/checkpoint_epoch_17.pth',
        'models/MLP/checkpoints/final_model.pth'
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"\nLoading model from: {path}")
            try:
                model = load_checkpoint(path, device)
                clean_images, noisy_images = load_clean_and_noisy_images(device)
                save_path = f'models/MLP/visualizations/denoised_{os.path.basename(path).split(".")[0]}.jpg'
                visualize_denoised_examples(model, clean_images, noisy_images, save_path)
                break  # Stop after first successful load
            except Exception as e:
                print(f"Failed to load {path}: {str(e)}")
        else:
            print(f"Checkpoint not found: {path}")
    else:
        print("Error: No valid checkpoints found!")

if __name__ == "__main__":
    main()