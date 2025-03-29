# Restoring Quality Images Using Denoising Models

This repository contains the implementation of various denoising models applied to the CIFAR-10 dataset, which has been artificially corrupted with Gaussian noise. The project, developed as part of the CS464 course by Group 12, aims to evaluate and compare the effectiveness of different approaches in restoring the original image quality. Five distinct methods are explored: Convolutional Autoencoder, Variational Autoencoder (VAE), MLP-based Autoencoder, K-Nearest Neighbors (KNN) filtering, and UNet-based model.

## Project Scope

The primary research question is: *Can different deep learning-based autoencoder models and traditional methods effectively remove Gaussian noise from CIFAR-10 images, and if so, which model architecture performs best?* To answer this, the project involves:

- **Data Preparation:** Adding Gaussian noise to CIFAR-10 images to create noisy-clean pairs for supervised learning.
- **Model Development:** Implementing five denoising approaches, each handled by a team member.
- **Evaluation:** Assessing model performance using metrics like Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
- **Comparison:** Analyzing the strengths and weaknesses of each method to determine the most effective approach.

## Repository Structure

- **`data/`**
  - `cifar10_data_download.py`: Script to download the CIFAR-10 dataset.
  - `data_preparation.py`: Script to add Gaussian noise to the images.
  - `data_info.txt`: Details about the dataset and noise parameters.
- **`models/`**
  - `conv_autoencoder/`
    - `conv_autoencoder.py`: Convolutional Autoencoder implementation.
    - `training_log.txt`: Training logs.
  - `VAE/`
    - `vae.py`: Variational Autoencoder implementation.
    - `training_log.txt`: Training logs.
  - `MLP/`
    - `mlp_autoencoder.py`: MLP-based Autoencoder implementation.
    - `training_log.txt`: Training logs.
  - `KNN/`
    - `knn_denoise.py`: KNN-based denoising algorithm (no training required).
  - `UNet/`
    - `unet_denoise.py`: UNet-based denoising model.
    - `training_log.txt`: Training logs.
- **`docs/`**
  - `progress_report.pdf`: Mid-term progress report.
  - `final_report.pdf`: Final project report.
  - `individual_notebooks/`: Jupyter notebooks for individual analyses (e.g., `egehan.ipynb`, `emir.ipynb`, etc.).
- **`requirements.txt`**: List of Python dependencies.
- **`.gitignore`**: Specifies files/folders to exclude from version control (e.g., `venv/`, `*.pyc`).

## Setup Instructions

To get started, follow these steps to set up the project environment:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/username/CS464-Group12-Denoising-Project.git
   cd CS464-Group12-Denoising-Project
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This installs required libraries such as NumPy, TensorFlow/PyTorch (depending on model implementations), Matplotlib, and Scikit-learn.

## Data Preparation

Prepare the dataset by downloading CIFAR-10 and adding Gaussian noise:

1. **Download CIFAR-10:**
   ```bash
   python data/cifar10_data_download.py
   ```
   This saves the dataset locally.

2. **Add Gaussian Noise:**
   ```bash
   python data/data_preparation.py
   ```
   This generates noisy versions of the images with a fixed noise level (e.g., σ = 25 for pixel values in [0, 255]). Check `data_info.txt` for specific parameters.

The resulting noisy-clean image pairs are used for training and testing the models.

## Running the Models

Each model resides in its own subdirectory under `models/`. Navigate to the respective directory and run the main script to train (if applicable) and evaluate the model. Examples:

- **Convolutional Autoencoder:**
  ```bash
  cd models/conv_autoencoder
  python conv_autoencoder.py
  ```

- **Variational Autoencoder (VAE):**
  ```bash
  cd models/VAE
  python vae.py
  ```

- **MLP-based Autoencoder:**
  ```bash
  cd models/MLP
  python mlp_autoencoder.py
  ```

- **KNN-based Denoising:**
  ```bash
  cd models/KNN
  python knn_denoise.py
  ```
  *Note:* KNN is a traditional method and does not require training; it directly processes the noisy images.

- **UNet-based Denoising:**
  ```bash
  cd models/UNet
  python unet_denoise.py
  ```

Each script handles model training (where applicable) and evaluation, saving outputs and logs accordingly. For deep learning models (e.g., UNet), a GPU is recommended for faster training.

## Viewing Results

- **Training Logs:** Check `training_log.txt` in each model’s directory for training progress.
- **Denoised Images and Metrics:** Outputs (e.g., denoised images, MSE/PSNR/SSIM values) are saved either in a `results/` directory or within each model’s folder, depending on the script implementation.

## Individual Notebooks

Explore the `docs/individual_notebooks/` directory for Jupyter notebooks (e.g., `egehan.ipynb`, `emir.ipynb`) prepared by team members. These contain detailed experiments, analyses, and visualizations specific to each model.

## Evaluation Metrics

Model performance is assessed using:
- **Mean Squared Error (MSE):** Measures pixel-level error between denoised and original images.
- **Peak Signal-to-Noise Ratio (PSNR):** Quantifies image quality in decibels; higher is better.
- **Structural Similarity Index (SSIM):** Evaluates perceptual similarity; ranges from 0 to 1 (1 = identical).

These metrics are computed for all models to enable a fair comparison.

## Team Contributions

- **Egehan Yıldız:** KNN-Based Denoising, Data Preparation, GitHub Repo Structure
- **Emir Ensar Sevil:** Variational Autoencoder
- **Ali Deniz Sözer:** MLP-based Autoencoder
- **Aybars Buğra Aksoy:** Convolutional Autoencoder
- **Eren Aslan:** UNet-based Denoising