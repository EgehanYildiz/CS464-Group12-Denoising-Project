import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

print("Script started running...")

# Dataset
class NoisyCIFARDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.noisy = torch.tensor(data['x_train_noisy']).permute(0, 3, 1, 2).float()
        self.clean = torch.tensor(data['x_train_clean']).permute(0, 3, 1, 2).float()

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# VAE model
class LazyVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(LazyVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Loss
def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div

# Evaluation
def compute_psnr_ssim_batch(recon_batch, target_batch):
    psnr_list, ssim_list = [], []
    for recon, target in zip(recon_batch, target_batch):
        r = recon.detach().cpu().numpy().transpose(1,2,0) 
        t = target.detach().cpu().numpy().transpose(1,2,0)
        psnr_list.append(peak_signal_noise_ratio(t, r, data_range=1))
        ssim_list.append(structural_similarity(t, r, channel_axis=2, data_range=1))
    return np.mean(psnr_list), np.mean(ssim_list)

# Visualization
def show_images(noisy, recon, clean, n=5):
    noisy = noisy.detach().cpu().numpy().transpose(0,2,3,1)
    recon = recon.detach().cpu().numpy().transpose(0,2,3,1)
    clean = clean.detach().cpu().numpy().transpose(0,2,3,1)

    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(noisy[i])
        plt.axis("off")
        plt.title("Noisy")
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(recon[i])
        plt.axis("off")
        plt.title("Denoised")
        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(clean[i])
        plt.axis("off")
        plt.title("Clean")
    plt.tight_layout()

# Training Only Runs If File Is Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NoisyCIFARDataset("cifar10_data/cifar10_train_noisy.npz")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = LazyVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    beta_schedule = [1e-4, 1e-3, 1e-2]
    epochs_per_stage = [20, 20, 60]

    for stage, (beta, num_epochs) in enumerate(zip(beta_schedule, epochs_per_stage)):
        print(f"\nStage {stage + 1} — Using beta = {beta}")
        
        for epoch in range(1, num_epochs + 1):
            global_epoch = sum(epochs_per_stage[:stage]) + epoch
            model.train()
            total_loss = 0

            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(noisy)
                loss = vae_loss(recon, clean, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {global_epoch}/100 | Loss: {total_loss / len(train_loader.dataset):.4f}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                sample_noisy, sample_clean = next(iter(train_loader))
                sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)
                recon, _, _ = model(sample_noisy)
                psnr, ssim = compute_psnr_ssim_batch(recon, sample_clean)
                print(f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

                if global_epoch % 10 == 0 or global_epoch == 1:
                    output_dir = "models/VAE/recon_outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, f"recon_epoch{global_epoch}.png")
                    show_images(sample_noisy, recon, sample_clean, n=5)
                    plt.savefig(filename)
                    plt.close()
                    print(f"Saved recon output to {filename}")

        # After stage
        stage_out = os.path.join("models/VAE/recon_outputs", f"recon_stage{stage+1}.png")
        show_images(sample_noisy, recon, sample_clean, n=5)
        plt.savefig(stage_out)
        plt.close()
        print(f"Saved recon output to {stage_out}")

    torch.save(model.state_dict(), "models/VAE/vae_denoiser.pth")
    print("Model saved")

    model.eval()
    with torch.no_grad():
        sample_noisy, sample_clean = next(iter(train_loader))
        sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)
        recon, _, _ = model(sample_noisy)
        print("Noisy max:", sample_noisy.max().item(), "min:", sample_noisy.min().item())
        print("Recon max:", recon.max().item(), "min:", recon.min().item())
        print("Clean max:", sample_clean.max().item(), "min:", sample_clean.min().item())
