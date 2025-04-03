import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from vae import LazyVAE, show_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NoisyCIFARDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.noisy = torch.tensor(data['x_test_noisy']).permute(0, 3, 1, 2).float()
        self.clean = torch.tensor(data['x_test_clean']).permute(0, 3, 1, 2).float()

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# Load test data
test_dataset = NoisyCIFARDataset("cifar10_data/cifar10_test_noisy.npz")  # use actual test set!
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = LazyVAE(latent_dim=128).to(device)
model.load_state_dict(torch.load("models/VAE/vae_denoiser.pth"))
model.eval()

# Get a batch and evaluate
with torch.no_grad():
    sample_noisy, sample_clean = next(iter(test_loader))
    sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)
    recon, _, _ = model(sample_noisy)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnr, ssim = [], []
    for r, c in zip(recon, sample_clean):
        r_np = r.detach().cpu().numpy().transpose(1,2,0)
        c_np = c.detach().cpu().numpy().transpose(1,2,0)
        psnr.append(peak_signal_noise_ratio(c_np, r_np, data_range=1))
        ssim.append(structural_similarity(c_np, r_np, channel_axis=2, data_range=1))

    print(f"Test PSNR: {np.mean(psnr):.2f} | SSIM: {np.mean(ssim):.4f}")

    # Show and save images
    show_images(sample_noisy, recon, sample_clean, n=5)
    plt.savefig("models/VAE/test_output.png")
    plt.close()
