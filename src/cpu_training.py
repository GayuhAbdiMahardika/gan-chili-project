"""
Script untuk training GAN dengan optimasi CPU
Versi yang lebih cepat untuk komputer tanpa CUDA
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import dari file utama
from gan_data_augmentation import Generator, Discriminator, ChiliDataset, weights_init

def train_gan_cpu_optimized(data_dir, class_name, target_images=200, epochs=50):
    """
    Training GAN yang dioptimasi untuk CPU
    
    Args:
        data_dir: Path ke folder data kelas
        class_name: Nama kelas
        target_images: Target jumlah gambar
        epochs: Jumlah epoch (dikurangi untuk CPU)
    """
    
    print(f"\n=== CPU-Optimized GAN Training untuk {class_name} ===")
    
    # Hyperparameters yang dioptimasi untuk CPU
    IMG_SIZE = 64
    BATCH_SIZE = 8  # Dikurangi untuk CPU
    LEARNING_RATE = 0.0003  # Sedikit lebih tinggi
    BETA1 = 0.5
    NZ = 100
    NGF = 32  # Dikurangi untuk mengurangi kompleksitas
    NDF = 32  # Dikurangi untuk mengurangi kompleksitas
    NC = 3
    
    device = torch.device("cpu")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = ChiliDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Gambar asli: {len(dataset)}")
    print(f"Target: {target_images}")
    print(f"Perlu generate: {target_images - len(dataset)}")
    print(f"Training epochs: {epochs}")
    print(f"Device: {device}")
    
    # Initialize networks dengan ukuran lebih kecil
    netG = Generator(NZ, NGF, NC).to(device)
    netD = Discriminator(NC, NDF).to(device)
    
    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # Fixed noise untuk monitoring
    fixed_noise = torch.randn(16, NZ, 1, 1, device=device)
    
    # Training
    print("Starting CPU training...")
    start_time = time.time()
    
    G_losses = []
    D_losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator
            netD.zero_grad()
            
            real_batch = data.to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            
            output = netD(real_batch).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update Generator
            netG.zero_grad()
            label.fill_(1.)
            
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
        
        # Save losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        epoch_time = time.time() - epoch_start
        
        # Print progress setiap 10 epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            
            print(f'Epoch [{epoch:3d}/{epochs}] Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} '
                  f'Time: {epoch_time:.1f}s ETA: {eta/60:.1f}m')
            
            # Save sample
            if epoch % 20 == 0:
                with torch.no_grad():
                    fake_sample = netG(fixed_noise).detach().cpu()
                    sample_dir = f"cpu_samples/{class_name}"
                    os.makedirs(sample_dir, exist_ok=True)
                    vutils.save_image(fake_sample, f"{sample_dir}/epoch_{epoch}.png", 
                                    normalize=True, nrow=4)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    
    # Save model
    model_dir = f"cpu_models/{class_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(netG.state_dict(), f"{model_dir}/generator.pth")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title(f"Training Losses - {class_name}")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"cpu_losses_{class_name.replace(' ', '_')}.png")
    plt.close()
    
    return netG

def generate_images_cpu(generator, class_name, num_images, output_dir):
    """Generate gambar menggunakan CPU"""
    print(f"Generating {num_images} images untuk {class_name}...")
    
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    NZ = 100
    device = torch.device("cpu")
    
    with torch.no_grad():
        for i in tqdm(range(num_images), desc="Generating"):
            noise = torch.randn(1, NZ, 1, 1, device=device)
            fake_image = generator(noise)
            
            # Convert to PIL and save
            img = fake_image[0].cpu()
            img = (img + 1) / 2.0
            img = transforms.ToPILImage()(img)
            
            img_name = f"generated_{class_name.replace(' ', '_')}_{i:03d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            img.save(img_path)
    
    print(f"Generated {num_images} images saved to {output_dir}")

def quick_demo():
    """Demo cepat dengan 1 kelas dan epoch minimal"""
    print("=== QUICK DEMO - CPU GAN Training ===")
    print("Training 1 kelas dengan 20 epoch untuk demo cepat...")
    
    class_name = "healthy"
    data_dir = r"c:\Riset Infromatika\Python V3\Dataset Original\train\healthy"
    
    # Train dengan epoch minimal
    generator = train_gan_cpu_optimized(data_dir, class_name, target_images=90, epochs=20)
    
    # Generate beberapa gambar
    demo_dir = "quick_demo_output"
    os.makedirs(demo_dir, exist_ok=True)
    generate_images_cpu(generator, class_name, 10, demo_dir)
    
    print(f"\nDemo selesai! Cek folder '{demo_dir}' untuk hasil.")
    print("Untuk training lengkap, gunakan: python main.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        print("CPU-Optimized GAN Training")
        print("Usage:")
        print("  python cpu_training.py demo          # Quick demo")
        print("  python main.py                       # Full training")
