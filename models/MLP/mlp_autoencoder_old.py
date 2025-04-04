import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ========================== Load CIFAR-10 from Local ==========================
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    images = data_dict[b'data']  # Shape: (10000, 3072)
    images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # Normalize
    labels = np.array(data_dict[b'labels'])
    return images, labels

# Load all batches
data_dir = "cifar10_data/cifar-10-batches-py"
train_files = [f"{data_dir}/data_batch_{i}" for i in range(1, 6)]
test_file = f"{data_dir}/test_batch"

train_images, train_labels = zip(*[load_cifar10_batch(f) for f in train_files])
train_images = np.vstack(train_images)
train_labels = np.hstack(train_labels)

test_images, test_labels = load_cifar10_batch(test_file)

# Custom dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(images)  # Keep as (N, 3, 32, 32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return img.view(-1), img.view(-1)  # Return flattened for model

# Create DataLoaders
train_dataset = CIFAR10Dataset(train_images)
test_dataset = CIFAR10Dataset(test_images)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("✅ CIFAR-10 Loaded Successfully!")

# ========================== Improved MLP Autoencoder ==========================
class MLP_Autoencoder(nn.Module):
    def __init__(self):
        super(MLP_Autoencoder, self).__init__()

        # Encoder with BatchNorm and LeakyReLU
        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),  # Larger latent space
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder with residual connection
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 32*32*3),
            nn.Sigmoid()
        )
        
        # Learned residual mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        # Ensure correct input shape (batch_size, 3072)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_flat = x.view(x.size(0), -1)
        
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return self.alpha * decoded + (1 - self.alpha) * x_flat

# ========================== Composite Loss Function ==========================
class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        
        # Calculate SSIM in batches
        input_img = input.view(-1, 3, 32, 32)
        target_img = target.view(-1, 3, 32, 32)
        
        input_np = input_img.permute(0,2,3,1).cpu().detach().numpy()
        target_np = target_img.permute(0,2,3,1).cpu().detach().numpy()
        
        ssim_loss = 0
        for i in range(input_np.shape[0]):
            ssim_loss += 1 - ssim(target_np[i], input_np[i], 
                                 channel_axis=-1, 
                                 data_range=1.0,
                                 win_size=3)
        
        return 0.7*mse_loss + 0.3*(ssim_loss/input_np.shape[0])

# ========================== Training Setup ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Autoencoder().to(device)
criterion = CompositeLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Checkpoint setup
checkpoint_dir = "models/MLP/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(checkpoint_dir, "training_log.txt")

# Load checkpoint if exists
start_epoch = 0
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip().split(", ")
            start_epoch = int(last_line[0]) + 1
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{int(last_line[0])}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"🔄 Resuming from epoch {start_epoch} (Loss: {last_line[1]})")

# ========================== Training Loop ==========================
num_epochs = 18

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Add synthetic noise
        noisy_images = images + 0.1*torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0, 1)
        
        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    scheduler.step(train_loss)
    
    # Save checkpoint
    checkpoint_data = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': train_loss
    }
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    # Keep only last 3 checkpoints
    if epoch >= 3:
        old_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch-3}.pth")
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
    
    # Log progress
    with open(log_file, "a") as f:
        f.write(f"{epoch}, {train_loss:.6f}\n")
    
    print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

# Save final model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
print("✅ Training Complete!")

# ========================== Evaluation ==========================
def visualize_results(model, loader, device, num_examples=5):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        clean = images[0][:num_examples].to(device)
        noisy = clean + 0.2*torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0, 1)
        denoised = model(noisy.view(num_examples, -1)).view(num_examples, 3, 32, 32)
        
        # Plot results
        plt.figure(figsize=(15, 5))
        for i in range(num_examples):
            plt.subplot(3, num_examples, i+1)
            plt.imshow(clean[i].cpu().permute(1, 2, 0))
            plt.title("Clean")
            plt.axis('off')
            
            plt.subplot(3, num_examples, num_examples+i+1)
            plt.imshow(noisy[i].cpu().permute(1, 2, 0))
            plt.title("Noisy")
            plt.axis('off')
            
            plt.subplot(3, num_examples, 2*num_examples+i+1)
            plt.imshow(denoised[i].cpu().permute(1, 2, 0))
            plt.title("Denoised")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("denoising_results.jpg")
        plt.show()

model.eval()
mse_scores = []
psnr_scores = []
ssim_scores = []

with torch.no_grad():
    for images, targets in test_loader:
        clean = targets.to(device)
        noisy = images.to(device)
        outputs = model(noisy)
        
        # Convert to numpy
        clean_np = clean.view(-1, 3, 32, 32).cpu().numpy().transpose(0, 2, 3, 1)
        output_np = outputs.view(-1, 3, 32, 32).cpu().numpy().transpose(0, 2, 3, 1)
        
        # Compute metrics
        for i in range(clean_np.shape[0]):
            mse_scores.append(np.mean((clean_np[i] - output_np[i])**2))
            psnr_scores.append(psnr(clean_np[i], output_np[i], data_range=1))
            ssim_scores.append(ssim(clean_np[i], output_np[i], 
                              channel_axis=-1, data_range=1, win_size=3))

print("\n📊 Final Evaluation:")
print(f"PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"SSIM: {np.mean(ssim_scores):.4f}")
print(f"MSE: {np.mean(mse_scores):.6f}")

# Visualize sample results
#visualize_results(model, test_loader, device)