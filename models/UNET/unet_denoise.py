import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# ------------------------ UNet Model ------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 32)   # 32x32 → 32 channels
        self.pool1 = nn.MaxPool2d(2)              # → 16x16
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)              # → 8x8

        self.bottleneck = conv_block(64, 128)

        self.up2 = up_block(128, 64)              # 8x8 → 16x16
        self.dec2 = conv_block(128, 64)
        self.up1 = up_block(64, 32)               # 16x16 → 32x32
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ------------------------ Data Loading ------------------------
def load_data(data_path, batch_size=64):
    data = np.load(data_path)
    x_noisy = torch.tensor(data['x_train_noisy'], dtype=torch.float32).permute(0, 3, 1, 2)
    x_clean = torch.tensor(data['x_train_clean'], dtype=torch.float32).permute(0, 3, 1, 2)
    dataset = TensorDataset(x_noisy, x_clean)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_test_samples(path):
    data = np.load(path)
    x_clean = torch.tensor(data['x_test_clean'], dtype=torch.float32).permute(0, 3, 1, 2)
    x_noisy = torch.tensor(data['x_test_noisy'], dtype=torch.float32).permute(0, 3, 1, 2)
    return x_noisy, x_clean

# ------------------------ Training & Evaluation ------------------------
def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    with open("models/UNET/training_log.txt", "a") as f:
        for epoch in range(epochs):
            total_loss = 0
            for x_noisy, x_clean in dataloader:
                optimizer.zero_grad()
                output = model(x_noisy)
                loss = criterion(output, x_clean)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, MSE: {avg_loss:.6f}")
            f.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}\n")
            f.flush()

# ------------------------ Evaluation Metrics ------------------------
def evaluate_model(model, x_noisy, x_clean, num_samples=10):
    model.eval()
    with torch.no_grad():
        output = model(x_noisy[:num_samples])
    denoised = output.clamp(0, 1).cpu().numpy()
    clean = x_clean[:num_samples].cpu().numpy()
    noisy = x_noisy[:num_samples].cpu().numpy()

    avg_psnr = np.mean([psnr(clean[i].transpose(1, 2, 0), denoised[i].transpose(1, 2, 0)) for i in range(num_samples)])
    avg_ssim = np.mean([
    ssim(clean[i].transpose(1, 2, 0), denoised[i].transpose(1, 2, 0), win_size=7, channel_axis=-1, data_range=1.0)
            for i in range(num_samples)
        ])


    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save comparison image
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples * 2))
    for i in range(num_samples):
        for ax, img, title in zip(axes[i], [clean[i], noisy[i], denoised[i]], ['Original', 'Noisy', 'Denoised']):
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(title)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig("comparison.png")
    print("Comparison image saved as 'comparison.png'.")

# ------------------------ Main ------------------------
if __name__ == "__main__":
    train_loader = load_data("cifar10_data/cifar10_train_noisy.npz", batch_size=64)
    x_noisy_test, x_clean_test = load_test_samples("cifar10_data/cifar10_test_noisy.npz")

    model = UNet()
    train_model(model, train_loader, epochs=10)
    evaluate_model(model, x_noisy_test, x_clean_test)
