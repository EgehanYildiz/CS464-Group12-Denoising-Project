import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# UNet Model
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),  # Add dropout to prevent overfitting
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = double_conv(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        
    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        
        return self.final(c1)

# Custom Dataset
class NoisyCIFAR10(Dataset):
    def __init__(self, clean_images, noisy_images, transform=None):
        self.clean = clean_images
        self.noisy = noisy_images
        self.transform = transform
    
    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        clean_img = self.clean[idx]
        noisy_img = self.noisy[idx]
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        
        return noisy_img, clean_img

# Load Noisy Data
def load_data(batch_size=64):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

    
    two_up = Path(__file__).parent.parent.parent

    # Construct your full path
    clean_path = two_up / "cifar10_data" / "cifar10_test_noisy.npz"
    noisy_path = two_up / "cifar10_data" / "cifar10_train_noisy.npz"

    clean_data = np.load(clean_path)
    noisy_data = np.load(noisy_path)
    
    clean_images = clean_data["x_test_clean"].astype(np.float32)  # Corrected key
    noisy_images = noisy_data["x_train_noisy"].astype(np.float32)  # Corrected key
    
    dataset = NoisyCIFAR10(clean_images, noisy_images, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Train the model
def train_unet(epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataloader = load_data(batch_size)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    return model

# Run Training
if __name__ == "__main__":
    trained_model = train_unet()
