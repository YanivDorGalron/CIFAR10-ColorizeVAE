import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os

# Hyperparameters
batch_size = 8
epochs = 10
lr = 1e-5
save_interval = 2  # Save model every few epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale 
])

transform_rgb = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_rgb)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_rgb)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Pretrained VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)

# Training Loop
os.makedirs("./checkpoints", exist_ok=True)
for epoch in trange(epochs, desc="Training Progress"):
    vae.train()
    total_loss = 0.0
    for rgb_imgs, _ in tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False):
        gray_imgs = transform(rgb_imgs)  # Convert RGB to grayscale 
        gray_imgs, rgb_imgs = gray_imgs.to(device), rgb_imgs.to(device)
        
        # Encode Gray images
        posterior = vae.encode(gray_imgs).latent_dist
        latents = posterior.sample()
        
        # Decode to RGB
        recon_rgb = vae.decode(latents).sample
        
        # Reconstruction Loss (MSE Loss)
        loss = F.mse_loss(recon_rgb, rgb_imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model every few epochs
    if (epoch + 1) % save_interval == 0:
        vae.save_pretrained(f"./checkpoints/vae_epoch_{epoch+1}")
    
    # Evaluation
    vae.eval()
    test_total_loss = 0.0
    with torch.no_grad():
        for rgb_imgs, _ in tqdm(test_loader, desc=f'Test Epoch {epoch}', leave=False):
            gray_imgs = transform(rgb_imgs)  # Convert RGB to grayscale 
            gray_imgs, rgb_imgs = gray_imgs.to(device), rgb_imgs.to(device)
            
            # Encode gray images
            posterior = vae.encode(gray_imgs).latent_dist
            latents = posterior.sample()
            
            # Decode to RGB
            recon_rgb = vae.decode(latents).sample
            
            # Reconstruction Loss (MSE Loss)
            loss = F.mse_loss(recon_rgb, rgb_imgs)
            test_total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_total_loss/len(test_loader):.4f}")

# Save Final Model
vae.save_pretrained("./checkpoints/vae_final")