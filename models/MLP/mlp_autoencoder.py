import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
        self.images = torch.tensor(images).view(-1, 32*32*3)  # Flatten

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images[idx]  # Autoencoder input = output

# Create DataLoaders
train_dataset = CIFAR10Dataset(train_images)
test_dataset = CIFAR10Dataset(test_images)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("✅ CIFAR-10 Loaded Successfully!")

# ========================== Define MLP Autoencoder ==========================
class MLP_Autoencoder(nn.Module):
    def __init__(self):
        super(MLP_Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)  # Latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32*32*3),  # Output size matches input
            nn.Sigmoid()  # Normalize output (0-1 range)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ========================== Checkpointing Setup ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_file = "models/MLP/training_log.txt"

# Load checkpoint if it exists
start_epoch = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip().split(", ")
            start_epoch = int(last_line[0]) + 1  # Resume from next epoch
            checkpoint_path = f"checkpoint_epoch_{last_line[0]}.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"🔄 Resuming from epoch {start_epoch} (Loss: {last_line[1]})")

# ========================== Training Loop with Logging ==========================
num_epochs = 50
log_file = "models/MLP/training_log.txt"

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Save training progress
    with open(checkpoint_file, "a") as f:
        f.write(f"{epoch}, {train_loss:.6f}\n")
    
    # Log training progress
    with open(log_file, "a") as log:
        log.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f}\n")
        log.flush()

    # Save model checkpoint
    checkpoint_data = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint_data, f"checkpoint_epoch_{epoch}.pth")

    print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f} (Saved Checkpoint)")

print("✅ Training Complete!")


# ========================== Evaluation ==========================
model.eval()
mse_scores = []
psnr_scores = []
ssim_scores = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)

        # Convert tensors to NumPy for evaluation
        images_np = images.cpu().numpy().reshape(-1, 32, 32, 3)
        outputs_np = outputs.cpu().numpy().reshape(-1, 32, 32, 3)

        # Compute MSE, PSNR, SSIM for each image in batch
        for i in range(images_np.shape[0]):
            mse_val = np.mean((images_np[i] - outputs_np[i])**2)
            psnr_val = psnr(images_np[i], outputs_np[i], data_range=1)
            ssim_val = ssim(images_np[i], outputs_np[i], channel_axis=-1, data_range=1, win_size=5)



            mse_scores.append(mse_val)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

# Print Final Results
print(f"\nEvaluation Results:")
print(f"MSE: {np.mean(mse_scores):.6f}")
print(f"PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"SSIM: {np.mean(ssim_scores):.4f}")

