"""
Quick 30-Minute GAN Training Script
Optimasi maksimal untuk training cepat dengan hasil yang baik dalam 30 menit
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
import shutil

# Import GAN classes
import sys
sys.path.append('src')
from gan_data_augmentation import Generator, Discriminator

def create_quick_config():
    """Konfigurasi super optimized untuk training 30 menit"""
    return {
        'img_size': 64,
        'batch_size': 8,  # Sangat kecil untuk kecepatan maksimal
        'num_epochs': 80,  # Reduced epochs
        'lr': 0.0004,  # Higher LR untuk konvergensi cepat
        'beta1': 0.5,
        'nz': 100,
        'ngf': 32,  # Sangat reduced features untuk kecepatan
        'ndf': 32,  # Sangat reduced features
        'target_images': 30,  # Target sangat sedikit
        'save_interval': 20,  # Save setiap 20 epoch
        'device': 'cpu',
        'print_interval': 5  # Print setiap 5 epoch
    }

def setup_quick_dataset(class_name, config):
    """Setup dataset untuk satu kelas dengan transformasi minimal"""
    
    dataset_path = r"Dataset Original\train"
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_path):
        raise ValueError(f"Class path tidak ditemukan: {class_path}")
    
    # Transformasi minimal untuk kecepatan
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create temporary dataset structure
    temp_dataset_path = "temp_dataset_30min"
    temp_class_path = os.path.join(temp_dataset_path, "images")
    
    # Clean and create temp directory
    if os.path.exists(temp_dataset_path):
        shutil.rmtree(temp_dataset_path)
    os.makedirs(temp_class_path, exist_ok=True)
    
    # Copy hanya sebagian images untuk kecepatan
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Ambil maksimal 40 gambar untuk training cepat
    max_images = min(40, len(image_files))
    selected_images = image_files[:max_images]
    
    for img_file in selected_images:
        src = os.path.join(class_path, img_file)
        dst = os.path.join(temp_class_path, img_file)
        shutil.copy2(src, dst)
    
    dataset = ImageFolder(temp_dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=0, pin_memory=False)
    
    return dataloader, len(selected_images)

class QuickGenerator(nn.Module):
    """Generator yang lebih ringan untuk training cepat"""
    def __init__(self, nz, ngf, nc):
        super(QuickGenerator, self).__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: ngf x 16 x 16
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # State: (ngf//2) x 32 x 32
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class QuickDiscriminator(nn.Module):
    """Discriminator yang lebih ringan untuk training cepat"""
    def __init__(self, nc, ndf):
        super(QuickDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def train_quick_gan(class_name, max_minutes=30):
    """Training GAN ultra cepat untuk satu kelas dalam 30 menit"""
    
    print(f"🚀 ULTRA QUICK TRAINING - {class_name.upper()}")
    print(f"⏰ Target waktu: {max_minutes} menit")
    print("="*60)
    
    config = create_quick_config()
    device = torch.device(config['device'])
    
    # Setup dataset
    print("📂 Setting up dataset...")
    start_setup = time.time()
    dataloader, original_count = setup_quick_dataset(class_name, config)
    setup_time = time.time() - start_setup
    print(f"   ✅ Dataset ready in {setup_time:.1f}s")
    print(f"   📊 Using {original_count} training images")
    print(f"   🎯 Will generate {config['target_images']} new images")
    
    # Initialize lightweight models
    print("🧠 Initializing lightweight models...")
    netG = QuickGenerator(config['nz'], config['ngf'], 3).to(device)
    netD = QuickDiscriminator(3, config['ndf']).to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function dan optimizers dengan higher LR
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    
    # Training setup
    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.randn(8, config['nz'], 1, 1, device=device)
    
    # Tracking
    start_time = time.time()
    losses_g = []
    losses_d = []
    
    print("🎯 Starting ultra quick training...")
    print(f"   🔧 Epochs: {config['num_epochs']}")
    print(f"   📦 Batch size: {config['batch_size']}")
    print(f"   🎛️  Learning rate: {config['lr']}")
    print(f"   💻 Device: {device}")
    
    # Progress tracking
    epoch_times = []
    
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
            
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            
            # Train with real
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Train with fake
            noise = torch.randn(batch_size, config['nz'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()
        
        # Calculate averages
        avg_d_loss = epoch_d_loss / max(batch_count, 1)
        avg_g_loss = epoch_g_loss / max(batch_count, 1)
        losses_d.append(avg_d_loss)
        losses_g.append(avg_g_loss)
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_elapsed = (time.time() - start_time) / 60
        
        # Estimate remaining time
        if len(epoch_times) > 5:
            avg_epoch_time = np.mean(epoch_times[-5:])
            remaining_epochs = config['num_epochs'] - epoch - 1
            estimated_remaining = (remaining_epochs * avg_epoch_time) / 60
        else:
            estimated_remaining = 0
        
        # Progress update
        if epoch % config['print_interval'] == 0 or epoch == config['num_epochs'] - 1:
            print(f"Epoch [{epoch:3d}/{config['num_epochs']}] "
                  f"D: {avg_d_loss:.3f} G: {avg_g_loss:.3f} "
                  f"⏱️ {epoch_time:.1f}s "
                  f"📈 {total_elapsed:.1f}m/{max_minutes}m "
                  f"🔮 ETA: {estimated_remaining:.1f}m")
        
        # Save sample images
        if epoch % config['save_interval'] == 0 or epoch == config['num_epochs'] - 1:
            save_sample(netG, fixed_noise, epoch, class_name, config)
        
        # Emergency break if we're running out of time
        if total_elapsed > max_minutes * 0.85:  # 85% of time used
            print(f"\n⚠️  Approaching time limit, finishing training early...")
            break
    
    # Final statistics
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed!")
    print(f"   ⏰ Time used: {total_time:.1f}/{max_minutes} minutes")
    print(f"   📊 Epochs completed: {epoch + 1}/{config['num_epochs']}")
    
    # Save final model
    print("💾 Saving models...")
    save_model(netG, netD, class_name, config)
    
    # Generate final images
    print(f"🎨 Generating {config['target_images']} final images...")
    generate_final_images(netG, class_name, config)
    
    # Plot losses
    plot_quick_losses(losses_d, losses_g, class_name, total_time, epoch + 1)
    
    # Cleanup
    cleanup_temp_files()
    
    print(f"\n🎉 ULTRA QUICK TRAINING COMPLETED!")
    print(f"   📁 Results saved to: quick_30min_{class_name}/")
    print(f"   🖼️  Generated images: {config['target_images']}")
    print(f"   📈 Training plots: quick_30min_{class_name}/losses.png")
    print(f"   💾 Models: quick_30min_{class_name}/models/")
    
    return total_time

def save_sample(generator, fixed_noise, epoch, class_name, config):
    """Save sample images during training"""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
        # Denormalize
        fake = (fake + 1) / 2
        fake = torch.clamp(fake, 0, 1)
        
        # Create output directory
        sample_dir = f"quick_30min_{class_name}/samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save grid
        import torchvision.utils as vutils
        vutils.save_image(fake, f"{sample_dir}/epoch_{epoch:03d}.png", 
                         nrow=4, normalize=False)
    generator.train()

def save_model(generator, discriminator, class_name, config):
    """Save trained models"""
    model_dir = f"quick_30min_{class_name}/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(generator.state_dict(), f"{model_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"{model_dir}/discriminator.pth")
    
    # Save config juga
    import json
    with open(f"{model_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

def generate_final_images(generator, class_name, config):
    """Generate final images for augmentation"""
    generator.eval()
    output_dir = f"quick_30min_{class_name}/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    num_generate = config['target_images']
    batch_size = 4  # Generate in very small batches for speed
    
    count = 0
    generation_start = time.time()
    
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
                img_pil.save(f"{output_dir}/generated_{class_name}_{count:03d}.jpg", 
                           quality=85, optimize=True)
                count += 1
    
    generation_time = time.time() - generation_start
    print(f"   ✅ Generated {count} images in {generation_time:.1f}s")

def plot_quick_losses(losses_d, losses_g, class_name, total_time, epochs_completed):
    """Plot training losses"""
    plt.figure(figsize=(12, 8))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    plt.plot(losses_d, label='Discriminator Loss', alpha=0.8, linewidth=2)
    plt.plot(losses_g, label='Generator Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses - {class_name.title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(2, 2, 2)
    window = min(10, len(losses_d) // 4)
    if window > 1:
        ma_d = np.convolve(losses_d, np.ones(window)/window, mode='valid')
        ma_g = np.convolve(losses_g, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses_d)), ma_d, label=f'D Loss (MA-{window})')
        plt.plot(range(window-1, len(losses_g)), ma_g, label=f'G Loss (MA-{window})')
        plt.xlabel('Epoch')
        plt.ylabel('Moving Average Loss')
        plt.title('Smoothed Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Training info
    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.8, f'Class: {class_name.title()}', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f'Training Time: {total_time:.1f} minutes', fontsize=10)
    plt.text(0.1, 0.6, f'Epochs Completed: {epochs_completed}', fontsize=10)
    plt.text(0.1, 0.5, f'Final D Loss: {losses_d[-1]:.3f}', fontsize=10)
    plt.text(0.1, 0.4, f'Final G Loss: {losses_g[-1]:.3f}', fontsize=10)
    plt.text(0.1, 0.3, f'Avg Time/Epoch: {(total_time*60)/epochs_completed:.1f}s', fontsize=10)
    plt.text(0.1, 0.2, f'Device: CPU', fontsize=10)
    plt.text(0.1, 0.1, f'Status: ✅ Completed', fontsize=10, color='green')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Training Summary')
    
    # Loss distribution
    plt.subplot(2, 2, 4)
    plt.hist(losses_d, alpha=0.7, label='Discriminator', bins=15)
    plt.hist(losses_g, alpha=0.7, label='Generator', bins=15)
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = f"quick_30min_{class_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/losses.png", dpi=300, bbox_inches='tight')
    plt.close()

def cleanup_temp_files():
    """Clean up temporary files"""
    if os.path.exists("temp_dataset_30min"):
        shutil.rmtree("temp_dataset_30min")

def print_available_classes():
    """Print available classes"""
    dataset_path = r"Dataset Original\train"
    if os.path.exists(dataset_path):
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        print("📋 Available classes:")
        for i, cls in enumerate(classes, 1):
            class_path = os.path.join(dataset_path, cls)
            img_count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {i}. {cls} ({img_count} images)")
        return classes
    else:
        print("❌ Dataset path tidak ditemukan!")
        return []

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra Quick 30-Minute GAN Training')
    parser.add_argument('--class', dest='class_name', 
                       help='Class to train (akan ditampilkan pilihan jika kosong)')
    parser.add_argument('--time', type=int, default=30,
                       help='Maximum training time in minutes (default: 30)')
    
    args = parser.parse_args()
    
    print("🚀 ULTRA QUICK GAN TRAINING (30 MINUTES)")
    print("="*60)
    
    # Get available classes
    available_classes = print_available_classes()
    
    if not available_classes:
        return
    
    # Select class
    if args.class_name:
        if args.class_name in available_classes:
            class_name = args.class_name
        else:
            print(f"❌ Class '{args.class_name}' tidak ditemukan!")
            print("💡 Gunakan salah satu dari yang tersedia di atas.")
            return
    else:
        print(f"\n📝 Pilih class untuk training (1-{len(available_classes)}):")
        try:
            choice = int(input("Masukkan nomor pilihan: ")) - 1
            if 0 <= choice < len(available_classes):
                class_name = available_classes[choice]
            else:
                print("❌ Pilihan tidak valid!")
                return
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Input tidak valid atau dibatalkan!")
            return
    
    print(f"\n🎯 Training Configuration:")
    print(f"   📁 Class: {class_name}")
    print(f"   ⏰ Max time: {args.time} minutes")
    print(f"   💻 Device: CPU (optimized)")
    print(f"   🎨 Target: 30 generated images")
    print("="*60)
    
    # Confirmation
    try:
        confirm = input(f"\n🚀 Start training '{class_name}' for {args.time} minutes? (y/n): ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ Training dibatalkan.")
            return
    except KeyboardInterrupt:
        print("\n❌ Training dibatalkan.")
        return
    
    # Start training
    start_total = time.time()
    try:
        training_time = train_quick_gan(class_name, args.time)
        total_time = (time.time() - start_total) / 60
        
        print(f"\n🎊 SUCCESS! Ultra quick training completed!")
        print(f"   ⏰ Total time: {total_time:.1f} minutes")
        print(f"   🎯 Training time: {training_time:.1f} minutes")
        print(f"   📁 Results: quick_30min_{class_name}/")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user!")
        cleanup_temp_files()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        cleanup_temp_files()

if __name__ == "__main__":
    main()