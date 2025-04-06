#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from tqdm import tqdm
import logging

# Set up logging
os.makedirs('models/MLP', exist_ok=True)
logging.basicConfig(
    filename='models/MLP/training_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

class MLPAutoencoder(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[1024, 512, 256, 128]):
        """
        MLP-based autoencoder for CIFAR-10 images (32x32x3 = 3072 pixels)
        
        Args:
            input_size: Flattened input dimension (32*32*3 = 3072)
            hidden_sizes: List of hidden layer sizes (encoder path)
        """
        super(MLPAutoencoder, self).__init__()
        
        # Create encoder layers with decreasing sizes
        encoder_layers = []
        prev_size = input_size
        
        for h_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, h_size))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_size = h_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers with increasing sizes
        decoder_layers = []
        hidden_sizes_reversed = list(reversed(hidden_sizes))
        
        for i in range(len(hidden_sizes_reversed) - 1):
            decoder_layers.append(nn.Linear(hidden_sizes_reversed[i], hidden_sizes_reversed[i+1]))
            decoder_layers.append(nn.ReLU(inplace=True))
        
        # Final decoder layer to original input size
        decoder_layers.append(nn.Linear(hidden_sizes_reversed[-1], input_size))
        decoder_layers.append(nn.Sigmoid())  # Output in range [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reshape back to image dimensions (B, C, H, W)
        decoded = decoded.view(batch_size, 3, 32, 32)
        
        return decoded

def load_data(data_dir='cifar10_data'):
    """Load the prepared CIFAR-10 data with noise"""
    try:
        train_data = np.load(os.path.join(data_dir, 'cifar10_train_noisy.npz'))
        test_data = np.load(os.path.join(data_dir, 'cifar10_test_noisy.npz'))
        
        x_train_clean = train_data['x_train_clean']
        x_train_noisy = train_data['x_train_noisy']
        x_test_clean = test_data['x_test_clean']
        x_test_noisy = test_data['x_test_noisy']
        
        # Convert to PyTorch tensors and permute to NCHW format
        x_train_clean = torch.tensor(x_train_clean, dtype=torch.float32).permute(0, 3, 1, 2)
        x_train_noisy = torch.tensor(x_train_noisy, dtype=torch.float32).permute(0, 3, 1, 2)
        x_test_clean = torch.tensor(x_test_clean, dtype=torch.float32).permute(0, 3, 1, 2)
        x_test_noisy = torch.tensor(x_test_noisy, dtype=torch.float32).permute(0, 3, 1, 2)
        
        logging.info(f"Loaded data shapes: Train: {x_train_clean.shape}, Test: {x_test_clean.shape}")
        
        return x_train_clean, x_train_noisy, x_test_clean, x_test_noisy
    
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        print("Error: Could not find prepared data files.")
        print("Please run the data preparation script first.")
        exit(1)

def create_dataloaders(x_train_clean, x_train_noisy, x_test_clean, x_test_noisy, batch_size=64, val_split=0.1):
    """Create training, validation, and test data loaders"""
    # Split training data into train and validation
    val_size = int(val_split * len(x_train_clean))
    indices = torch.randperm(len(x_train_clean))
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    x_val_clean = x_train_clean[val_indices]
    x_val_noisy = x_train_noisy[val_indices]
    x_train_clean = x_train_clean[train_indices]
    x_train_noisy = x_train_noisy[train_indices]
    
    # Create datasets
    train_dataset = TensorDataset(x_train_noisy, x_train_clean)
    val_dataset = TensorDataset(x_val_noisy, x_val_clean)
    test_dataset = TensorDataset(x_test_noisy, x_test_clean)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logging.info(f"Created dataloaders - Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, save_path='models/MLP'):
    """Train the MLP autoencoder"""
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Start training
    logging.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move tensors to the configured device
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log progress (less frequently to avoid clutter)
            if (batch_idx + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                batch_loss = criterion(outputs, clean_imgs)
                val_loss += batch_loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, Time: {epoch_time:.2f}s')
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            logging.info(f'Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.6f}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()
    
    return train_losses, val_losses

def evaluate(model, test_loader, save_path='models/MLP/results'):
    """Evaluate the model and save sample results"""
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    # Metrics
    test_loss = 0.0
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    # Original noisy image metrics (for comparison)
    mse_noisy_values = []
    psnr_noisy_values = []
    ssim_noisy_values = []
    
    # For visualization
    samples = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for idx, (noisy_imgs, clean_imgs) in enumerate(test_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            test_loss += loss.item()
            
            # Convert to numpy for metric calculation
            clean_np = clean_imgs.cpu().permute(0, 2, 3, 1).numpy()
            noisy_np = noisy_imgs.cpu().permute(0, 2, 3, 1).numpy()
            output_np = outputs.cpu().permute(0, 2, 3, 1).numpy()
            
            # Save first batch for visualization
            if idx == 0:
                samples = (clean_np[:10], noisy_np[:10], output_np[:10])
            
            # Calculate metrics for each image
            for i in range(len(clean_np)):
                # MSE
                mse = np.mean((clean_np[i] - output_np[i])**2)
                mse_noisy = np.mean((clean_np[i] - noisy_np[i])**2)
                mse_values.append(mse)
                mse_noisy_values.append(mse_noisy)
                
                # PSNR
                psnr = peak_signal_noise_ratio(clean_np[i], output_np[i], data_range=1.0)
                psnr_noisy = peak_signal_noise_ratio(clean_np[i], noisy_np[i], data_range=1.0)
                psnr_values.append(psnr)
                psnr_noisy_values.append(psnr_noisy)
                
                # SSIM with proper parameters to handle small image size
                try:
                    # For newer versions of scikit-image
                    ssim = structural_similarity(clean_np[i], output_np[i], channel_axis=2, data_range=1.0, win_size=3)
                    ssim_noisy = structural_similarity(clean_np[i], noisy_np[i], channel_axis=2, data_range=1.0, win_size=3)
                except TypeError:
                    # Fallback for older versions of scikit-image
                    ssim = structural_similarity(clean_np[i], output_np[i], multichannel=True, data_range=1.0, win_size=3)
                    ssim_noisy = structural_similarity(clean_np[i], noisy_np[i], multichannel=True, data_range=1.0, win_size=3)
                
                ssim_values.append(ssim)
                ssim_noisy_values.append(ssim_noisy)
                
            # Process fewer batches to save time when using Colab
            if idx >= 4:  
                break
    
    # Calculate average metrics
    avg_test_loss = test_loss / min(5, len(test_loader))
    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    avg_mse_noisy = np.mean(mse_noisy_values)
    avg_psnr_noisy = np.mean(psnr_noisy_values)
    avg_ssim_noisy = np.mean(ssim_noisy_values)
    
    # Save evaluation metrics
    evaluation_results = {
        'test_loss': avg_test_loss,
        'mse': {'noisy': avg_mse_noisy, 'denoised': avg_mse},
        'psnr': {'noisy': avg_psnr_noisy, 'denoised': avg_psnr},
        'ssim': {'noisy': avg_ssim_noisy, 'denoised': avg_ssim}
    }
    
    # Log and save evaluation results
    with open(os.path.join(save_path, 'evaluation_metrics.txt'), 'w') as f:
        f.write("MLP Autoencoder Evaluation Results\n")
        f.write("=====================================\n\n")
        f.write(f"Test Loss: {avg_test_loss:.6f}\n\n")
        
        f.write(f"Mean Squared Error (MSE):\n")
        f.write(f"  Noisy images: {avg_mse_noisy:.6f}\n")
        f.write(f"  Denoised images: {avg_mse:.6f}\n")
        f.write(f"  Improvement: {(1 - avg_mse/avg_mse_noisy) * 100:.2f}%\n\n")
        
        f.write(f"Peak Signal-to-Noise Ratio (PSNR):\n")
        f.write(f"  Noisy images: {avg_psnr_noisy:.2f} dB\n")
        f.write(f"  Denoised images: {avg_psnr:.2f} dB\n")
        f.write(f"  Improvement: {avg_psnr - avg_psnr_noisy:.2f} dB\n\n")
        
        f.write(f"Structural Similarity Index (SSIM):\n")
        f.write(f"  Noisy images: {avg_ssim_noisy:.4f}\n")
        f.write(f"  Denoised images: {avg_ssim:.4f}\n")
        f.write(f"  Improvement: {(avg_ssim - avg_ssim_noisy) * 100:.2f}%\n")
    
    logging.info(f"Evaluation results - MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    # Visualize sample results
    clean_samples, noisy_samples, denoised_samples = samples
    num_samples = min(5, len(clean_samples))  # Limit to 5 samples
    
    # Create figure with column headers
    plt.figure(figsize=(15, 5*num_samples + 1))
    
    # Add column labels at the top
    plt.subplot(num_samples + 1, 3, 1)
    plt.text(0.5, 0.5, "CLEAN", fontsize=16, fontweight='bold', ha='center')
    plt.axis('off')
    
    plt.subplot(num_samples + 1, 3, 2)
    plt.text(0.5, 0.5, "NOISY", fontsize=16, fontweight='bold', ha='center')
    plt.axis('off')
    
    plt.subplot(num_samples + 1, 3, 3)
    plt.text(0.5, 0.5, "DENOISED", fontsize=16, fontweight='bold', ha='center')
    plt.axis('off')
    
    # Display sample images
    for i in range(num_samples):
        # Clean image
        plt.subplot(num_samples + 1, 3, (i+1)*3 + 1)
        plt.imshow(clean_samples[i])
        plt.title("Clean")
        plt.axis('off')
        
        # Noisy image
        plt.subplot(num_samples + 1, 3, (i+1)*3 + 2)
        plt.imshow(noisy_samples[i])
        plt.title(f"PSNR: {peak_signal_noise_ratio(clean_samples[i], noisy_samples[i], data_range=1.0):.2f} dB")
        plt.axis('off')
        
        # Denoised image
        plt.subplot(num_samples + 1, 3, (i+1)*3 + 3)
        plt.imshow(denoised_samples[i])
        plt.title(f"PSNR: {peak_signal_noise_ratio(clean_samples[i], denoised_samples[i], data_range=1.0):.2f} dB")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sample_results.png'))
    plt.close()
    
    return evaluation_results

def main():
    # Configuration
    data_dir = 'cifar10_data'
    save_dir = 'models/MLP'
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Start logging
    logging.info("Starting MLP autoencoder training script")
    logging.info(f"Configuration: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")
    
    try:
        # Load data
        logging.info("Loading data...")
        x_train_clean, x_train_noisy, x_test_clean, x_test_noisy = load_data(data_dir)
        
        # Create dataloaders
        logging.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            x_train_clean, x_train_noisy, x_test_clean, x_test_noisy,
            batch_size=batch_size
        )
        
        # Initialize model, loss function, and optimizer
        logging.info("Initializing MLP model...")
        # Define the MLP autoencoder with appropriate hidden sizes
        model = MLPAutoencoder(
            input_size=3*32*32,  # 3072 flattened pixels
            hidden_sizes=[1024, 512, 256, 128]  # Progressively smaller layers
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Print model architecture
        model_summary = str(model)
        logging.info(f"Model architecture:\n{model_summary}")
        
        # Train the model
        logging.info("Starting model training...")
        train_losses, val_losses = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            save_path=save_dir
        )
        
        # Load best model for evaluation
        logging.info("Loading best model for evaluation...")
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate the model
        logging.info("Evaluating model...")
        evaluation_results = evaluate(
            model,
            test_loader,
            save_path=os.path.join(save_dir, 'results')
        )
        
        logging.info("Training and evaluation completed successfully!")
        print(f"Training and evaluation completed. Results saved to {save_dir}/results/")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()