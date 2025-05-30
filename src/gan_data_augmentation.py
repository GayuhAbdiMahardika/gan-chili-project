"""
GAN untuk Augmentasi Data Penyakit Tanaman Cabai
Menggunakan DCGAN (Deep Convolutional GAN) untuk generate gambar baru
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 0.0002
BETA1 = 0.5
NZ = 100  # Size of latent vector
NGF = 64  # Generator feature map size
NDF = 64  # Discriminator feature map size
NC = 3    # Number of channels

class ChiliDataset(Dataset):
    """Custom dataset untuk memuat gambar cabai"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Load semua gambar dari folder
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class Generator(nn.Module):
    """Generator network untuk DCGAN"""
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent vector nz
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output state size: (nc) x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """Discriminator network untuk DCGAN"""
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input state size: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output state size: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    """Initialize weights untuk network"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_gan(data_dir, class_name, target_images=200):
    """
    Train GAN untuk kelas tertentu
    
    Args:
        data_dir: Path ke folder data kelas
        class_name: Nama kelas (healthy, leaf curl, dll)
        target_images: Target jumlah gambar setelah augmentasi
    """
    print(f"\n=== Training GAN untuk kelas: {class_name} ===")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = ChiliDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"Jumlah gambar asli: {len(dataset)}")
    print(f"Target gambar: {target_images}")
    print(f"Perlu generate: {target_images - len(dataset)} gambar")
    
    # Initialize networks
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
    
    # Fixed noise untuk monitoring progress
    fixed_noise = torch.randn(16, NZ, 1, 1, device=device)
    
    # Training loop
    print("Starting Training...")
    
    G_losses = []
    D_losses = []
    
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            
            # Train with real batch
            real_batch = data.to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            
            output = netD(real_batch).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake batch
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
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
        
        # Print statistics
        if epoch % 50 == 0:
            print(f'[{epoch}/{NUM_EPOCHS}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            # Save sample images
            with torch.no_grad():
                fake_sample = netG(fixed_noise).detach().cpu()
                sample_dir = f"gan_samples/{class_name}"
                os.makedirs(sample_dir, exist_ok=True)
                vutils.save_image(fake_sample, f"{sample_dir}/epoch_{epoch}.png", normalize=True, nrow=4)
    
    # Save model
    model_dir = f"gan_models/{class_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(netG.state_dict(), f"{model_dir}/generator.pth")
    torch.save(netD.state_dict(), f"{model_dir}/discriminator.pth")
    
    print(f"Model saved to {model_dir}")
    
    return netG, netD, G_losses, D_losses

def generate_images(generator, class_name, num_images, output_dir):
    """
    Generate gambar baru menggunakan trained generator
    
    Args:
        generator: Trained generator model
        class_name: Nama kelas
        num_images: Jumlah gambar yang akan di-generate
        output_dir: Directory output untuk menyimpan gambar
    """
    print(f"\nGenerating {num_images} images untuk kelas {class_name}...")
    
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dalam batch
    batch_size = 32
    generated_count = 0
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            current_batch_size = min(batch_size, num_images - i)
            
            # Generate noise
            noise = torch.randn(current_batch_size, NZ, 1, 1, device=device)
            
            # Generate fake images
            fake_images = generator(noise)
            
            # Save images
            for j in range(current_batch_size):
                # Denormalize image
                img = fake_images[j].cpu()
                img = (img + 1) / 2.0  # Convert dari [-1,1] ke [0,1]
                
                # Convert to PIL Image
                img_pil = transforms.ToPILImage()(img)
                
                # Save image
                img_name = f"generated_{class_name}_{generated_count:03d}.jpg"
                img_path = os.path.join(output_dir, img_name)
                img_pil.save(img_path)
                
                generated_count += 1
    
    print(f"Generated {generated_count} images saved to {output_dir}")

def main():
    """Main function untuk menjalankan augmentasi data"""
    
    # Path ke dataset
    train_dir = r"c:\Riset Infromatika\Python V3\Dataset Original\train"
    
    # Daftar kelas penyakit
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # Target jumlah gambar per kelas
    target_images_per_class = 200
    
    # Buat folder output untuk gambar yang di-generate
    output_base_dir = r"c:\Riset Infromatika\Python V3\Dataset Augmented\train"
    
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} tidak ditemukan!")
            continue
            
        # Hitung jumlah gambar yang sudah ada
        existing_images = len([f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if existing_images >= target_images_per_class:
            print(f"Kelas {class_name} sudah memiliki {existing_images} gambar (target: {target_images_per_class})")
            continue
        
        # Train GAN untuk kelas ini
        generator, discriminator, g_losses, d_losses = train_gan(
            class_dir, class_name, target_images_per_class
        )
        
        # Generate gambar tambahan
        num_to_generate = target_images_per_class - existing_images
        output_dir = os.path.join(output_base_dir, class_name)
        
        generate_images(generator, class_name, num_to_generate, output_dir)
        
        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.title(f"Generator and Discriminator Loss During Training - {class_name}")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"gan_losses_{class_name}.png")
        plt.close()
        
        print(f"Selesai untuk kelas {class_name}")
        print("-" * 50)

if __name__ == "__main__":
    main()
