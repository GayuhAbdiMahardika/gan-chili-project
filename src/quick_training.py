"""
Quick Training Script - Maksimal 40 menit untuk satu kelas
Optimasi khusus untuk training cepat dengan hasil yang baik
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import GAN classes
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gan_data_augmentation import Generator, Discriminator

def create_quick_config():
    """Konfigurasi optimized untuk training 40 menit"""
    return {
        'img_size': 64,
        'batch_size': 16,  # Lebih kecil untuk CPU
        'num_epochs': 100,  # Reduced epochs
        'lr': 0.0003,  # Slightly higher LR
        'beta1': 0.5,
        'nz': 100,
        'ngf': 64,  # Reduced features
        'ndf': 64,  # Reduced features
        'target_images': 50,  # Target lebih sedikit
        'save_interval': 25,  # Save setiap 25 epoch
        'device': 'cpu'
    }

def setup_quick_dataset(class_name, config):
    """Setup dataset untuk satu kelas dengan transformasi cepat"""
    
    dataset_path = r"c:\Riset Infromatika\Python V3\GAN_Project\Dataset Original\train"
    class_path = os.path.join(dataset_path, class_name)
    
    # Transformasi sederhana dan cepat
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create temporary dataset structure with proper ImageFolder structure
    temp_dataset_path = "temp_dataset"
    temp_class_path = os.path.join(temp_dataset_path, class_name)  # Use original class name
    
    os.makedirs(temp_class_path, exist_ok=True)
    
    # Copy images to temp structure
    import shutil
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        src = os.path.join(class_path, img_file)
        dst = os.path.join(temp_class_path, img_file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    dataset = ImageFolder(temp_dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=0)  # No multiprocessing
    
    return dataloader, len(image_files)

def train_quick_gan(class_name, max_minutes=40):
    """Training GAN cepat untuk satu kelas"""
    
    print(f"🚀 QUICK TRAINING - {class_name.upper()}")
    print(f"⏰ Target waktu: {max_minutes} menit")
    print("="*50)
    
    config = create_quick_config()
    device = torch.device(config['device'])
    
    # Setup dataset
    print("📂 Setting up dataset...")
    dataloader, original_count = setup_quick_dataset(class_name, config)
    print(f"   Original images: {original_count}")
    print(f"   Target generate: {config['target_images']}")
    
    # Initialize models dengan arsitektur yang lebih ringan
    print("🧠 Initializing models...")
    netG = Generator(config['nz'], config['ngf'], 3).to(device)
    netD = Discriminator(3, config['ndf']).to(device)
    
    # Loss function dan optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    
    # Training setup
    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.randn(16, config['nz'], 1, 1, device=device)
    
    # Tracking
    start_time = time.time()
    losses_g = []
    losses_d = []
    
    print("🎯 Starting quick training...")
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Check time limit
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > max_minutes:
            print(f"\n⏰ Time limit reached: {elapsed_minutes:.1f} minutes")
            break
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        
        for i, (data, _) in enumerate(dataloader):
            batch_count += 1
            
            # Update Discriminator
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            
            # Real images
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Fake images
            noise = torch.randn(batch_size, config['nz'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()
        
        # Average losses
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        losses_d.append(avg_d_loss)
        losses_g.append(avg_g_loss)
        
        # Progress update
        elapsed_time = time.time() - epoch_start
        total_elapsed = (time.time() - start_time) / 60
        remaining_time = max_minutes - total_elapsed
        
        if epoch % 10 == 0 or epoch == config['num_epochs'] - 1:
            print(f"Epoch [{epoch:3d}/{config['num_epochs']}] "
                  f"D_loss: {avg_d_loss:.3f} G_loss: {avg_g_loss:.3f} "
                  f"Time: {elapsed_time:.1f}s Remaining: {remaining_time:.1f}m")
        
        # Save sample setiap interval
        if epoch % config['save_interval'] == 0:
            save_sample(netG, fixed_noise, epoch, class_name, config)
    
    # Final time check
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed in {total_time:.1f} minutes")
    
    # Save final model
    save_model(netG, netD, class_name, config)
    
    # Generate final images
    print(f"🎨 Generating {config['target_images']} final images...")
    generate_final_images(netG, class_name, config)
    
    # Plot losses
    plot_quick_losses(losses_d, losses_g, class_name, total_time)
    
    # Cleanup
    cleanup_temp_files()
    
    print(f"\n🎉 Quick training completed!")
    print(f"   Time used: {total_time:.1f}/{max_minutes} minutes")
    print(f"   Generated: {config['target_images']} images")
    print(f"   Saved to: quick_results_{class_name}/")

def save_sample(generator, fixed_noise, epoch, class_name, config):
    """Save sample images during training"""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
        # Denormalize
        fake = (fake + 1) / 2
        fake = torch.clamp(fake, 0, 1)
        
        # Create output directory
        sample_dir = f"quick_results_{class_name}/samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save grid
        import torchvision.utils as vutils
        vutils.save_image(fake, f"{sample_dir}/epoch_{epoch}.png", 
                         nrow=4, normalize=False)
    generator.train()

def save_model(generator, discriminator, class_name, config):
    """Save trained models"""
    model_dir = f"quick_results_{class_name}/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(generator.state_dict(), f"{model_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"{model_dir}/discriminator.pth")

def generate_final_images(generator, class_name, config):
    """Generate final images for augmentation"""
    generator.eval()
    output_dir = f"quick_results_{class_name}/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    num_generate = config['target_images']
    batch_size = 8  # Generate in small batches
    
    count = 0
    with torch.no_grad():
        for i in range(0, num_generate, batch_size):
            current_batch = min(batch_size, num_generate - i)
            noise = torch.randn(current_batch, config['nz'], 1, 1, 
                              device=torch.device(config['device']))
            
            fake_images = generator(noise)
            # Denormalize
            fake_images = (fake_images + 1) / 2
            fake_images = torch.clamp(fake_images, 0, 1)
            
            # Save individual images
            for j in range(current_batch):
                img_tensor = fake_images[j]
                img_pil = transforms.ToPILImage()(img_tensor)
                img_pil.save(f"{output_dir}/generated_{class_name}_{count:03d}.jpg")
                count += 1

def plot_quick_losses(losses_d, losses_g, class_name, total_time):
    """Plot training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses_d, label='Discriminator Loss', alpha=0.7)
    plt.plot(losses_g, label='Generator Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Quick Training Losses - {class_name.title()}\n'
              f'Training Time: {total_time:.1f} minutes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = f"quick_results_{class_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/losses.png", dpi=300, bbox_inches='tight')
    plt.close()

def cleanup_temp_files():
    """Clean up temporary files"""
    import shutil
    if os.path.exists("temp_dataset"):
        shutil.rmtree("temp_dataset")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick GAN Training')
    parser.add_argument('--class', dest='class_name', default='healthy',
                       choices=['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish'],
                       help='Class to train')
    parser.add_argument('--time', type=int, default=40,
                       help='Maximum training time in minutes')
    
    args = parser.parse_args()
    
    print("🚀 QUICK GAN TRAINING")
    print("="*50)
    print(f"Class: {args.class_name}")
    print(f"Max time: {args.time} minutes")
    print(f"Device: CPU (optimized)")
    print("="*50)
    
    train_quick_gan(args.class_name, args.time)

if __name__ == "__main__":
    main()
