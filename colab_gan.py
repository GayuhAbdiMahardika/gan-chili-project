"""
GAN Data Augmentation - Google Colab Version
Script yang dioptimasi untuk running di Google Colab
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
import time
import zipfile

# Check if running in Colab
try:
    from google.colab import files, drive
    IN_COLAB = True
    print("🚀 Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally")

# Set device with Colab optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Hyperparameters optimized for Colab
IMG_SIZE = 64
BATCH_SIZE = 64 if torch.cuda.is_available() else 32  # Larger batch for GPU
NUM_EPOCHS = 150  # Reduced for Colab time limits
LEARNING_RATE = 0.0002
BETA1 = 0.5
NZ = 100
NGF = 64
NDF = 64
NC = 3

class ColabChiliDataset(Dataset):
    """Optimized dataset for Colab"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Load images with error handling
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, file))
        
        print(f"📁 Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"⚠️ Error loading image {idx}: {e}")
            # Return a dummy image if loading fails
            dummy = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return dummy

def weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ColabGenerator(nn.Module):
    """Generator optimized for Colab"""
    def __init__(self, nz, ngf, nc):
        super(ColabGenerator, self).__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class ColabDiscriminator(nn.Module):
    """Discriminator optimized for Colab"""
    def __init__(self, nc, ndf):
        super(ColabDiscriminator, self).__init__()
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
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def setup_colab_dataset():
    """Setup dataset for Colab"""
    print("📊 Setting up dataset...")
    
    # Common dataset paths
    dataset_paths = [
        "Dataset Original/train",
        "data/train",
        "train",
        "/content/Dataset Original/train",
        "/content/data/train"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Dataset not found!")
        print("💡 Please upload your dataset using one of these methods:")
        print("   1. Upload ZIP file and extract")
        print("   2. Mount Google Drive and copy data")
        return None
    
    print(f"✅ Dataset found at: {dataset_path}")
    
    # Check classes
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    available_classes = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if count > 0:
                available_classes.append(class_name)
                print(f"  ✅ {class_name:12}: {count:3d} images")
            else:
                print(f"  ⚠️  {class_name:12}: Empty folder")
        else:
            print(f"  ❌ {class_name:12}: Not found")
    
    return dataset_path, available_classes

def train_gan_colab(data_dir, class_name, epochs=150, target_images=200):
    """
    Train GAN optimized for Google Colab
    """
    print(f"\n🚀 Training GAN for: {class_name}")
    print(f"📱 Device: {device}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = ColabChiliDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"📊 Original images: {len(dataset)}")
    print(f"🎯 Target images: {target_images}")
    print(f"🆕 Need to generate: {target_images - len(dataset)} images")
    
    # Initialize networks
    netG = ColabGenerator(NZ, NGF, NC).to(device)
    netD = ColabDiscriminator(NC, NDF).to(device)
    
    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # Fixed noise for monitoring
    fixed_noise = torch.randn(16, NZ, 1, 1, device=device)
    
    # Training loop
    print("🏋️ Starting training...")
    start_time = time.time()
    
    G_losses = []
    D_losses = []
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f"Training {class_name}")
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        errG_epoch = 0
        errD_epoch = 0
        
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ############################
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
            
            ############################
            # (2) Update G network
            ############################
            netG.zero_grad()
            label.fill_(1.)
            
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            errG_epoch += errG.item()
            errD_epoch += errD.item()
        
        # Average losses for epoch
        errG_epoch /= len(dataloader)
        errD_epoch /= len(dataloader)
        
        G_losses.append(errG_epoch)
        D_losses.append(errD_epoch)
        
        # Update progress bar
        epoch_time = time.time() - epoch_start
        epoch_pbar.set_postfix({
            'D_loss': f"{errD_epoch:.3f}",
            'G_loss': f"{errG_epoch:.3f}",
            'Time': f"{epoch_time:.1f}s"
        })
        
        # Save sample images
        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                fake_sample = netG(fixed_noise).detach().cpu()
                sample_dir = f"colab_samples/{class_name}"
                os.makedirs(sample_dir, exist_ok=True)
                vutils.save_image(fake_sample, f"{sample_dir}/epoch_{epoch}.png", 
                                normalize=True, nrow=4)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    
    # Save model
    model_dir = f"colab_models/{class_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(netG.state_dict(), f"{model_dir}/generator.pth")
    torch.save(netD.state_dict(), f"{model_dir}/discriminator.pth")
    
    print(f"💾 Model saved to {model_dir}")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return netG, netD, G_losses, D_losses

def generate_images_colab(generator, class_name, num_images, output_dir="colab_augmented"):
    """Generate images using trained generator"""
    print(f"\n🎨 Generating {num_images} images for {class_name}...")
    
    generator.eval()
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Generate in batches to avoid memory issues
        batch_size = 32
        generated = 0
        
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            noise = torch.randn(current_batch, NZ, 1, 1, device=device)
            fake_images = generator(noise)
            
            # Save images
            for i in range(current_batch):
                img = fake_images[i].cpu()
                img = (img + 1) / 2.0  # Denormalize
                img = transforms.ToPILImage()(img)
                
                img_name = f"generated_{class_name.replace(' ', '_')}_{generated + i:03d}.jpg"
                img_path = os.path.join(class_output_dir, img_name)
                img.save(img_path)
            
            generated += current_batch
            
            # Show progress
            if generated % 50 == 0 or generated == num_images:
                print(f"  📈 Generated: {generated}/{num_images}")
    
    print(f"✅ Generated {num_images} images saved to {class_output_dir}")

def train_all_classes_colab(dataset_path, available_classes, epochs=150, target_images=200):
    """Train GAN for all available classes"""
    print("🏭 COLAB FULL TRAINING - ALL CLASSES")
    print(f"📊 Classes: {len(available_classes)}")
    print(f"🎯 Target per class: {target_images} images")
    print(f"🔄 Training epochs: {epochs}")
    print("=" * 60)
    
    results = {}
    total_start_time = time.time()
    
    for i, class_name in enumerate(available_classes, 1):
        print(f"\n[{i}/{len(available_classes)}] 🚀 Training: {class_name}")
        print("-" * 40)
        
        class_dir = os.path.join(dataset_path, class_name)
        
        try:
            # Train GAN
            generator, discriminator, g_losses, d_losses = train_gan_colab(
                class_dir, class_name, epochs, target_images
            )
            
            # Calculate how many images to generate
            original_count = len([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            needed_images = max(0, target_images - original_count)
            
            if needed_images > 0:
                generate_images_colab(generator, class_name, needed_images)
            
            results[class_name] = {
                'generator': generator,
                'g_losses': g_losses,
                'd_losses': d_losses,
                'original_count': original_count,
                'generated_count': needed_images
            }
            
            print(f"✅ Completed {class_name}: {original_count} → {target_images} images")
            
        except Exception as e:
            print(f"❌ Error training {class_name}: {str(e)}")
            continue
        
        # Memory cleanup between classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - total_start_time
    print(f"\n🎊 COLAB TRAINING COMPLETED!")
    print(f"⏰ Total time: {total_time/3600:.1f} hours")
    print(f"📁 Results saved in colab_* directories")
    
    return results

def create_colab_summary(results):
    """Create training summary"""
    print("\n📋 TRAINING SUMMARY")
    print("=" * 50)
    
    total_original = 0
    total_generated = 0
    
    for class_name, result in results.items():
        original = result['original_count']
        generated = result['generated_count']
        total_original += original
        total_generated += generated
        
        print(f"  {class_name:15}: {original:3d} → {original + generated:3d} images (+{generated})")
    
    print("-" * 50)
    print(f"  {'TOTAL':15}: {total_original:3d} → {total_original + total_generated:3d} images (+{total_generated})")
    print(f"\n📈 Dataset increase: {(total_generated/total_original)*100:.1f}%")

def download_colab_results():
    """Download all results from Colab"""
    if not IN_COLAB:
        print("❌ This function only works in Google Colab")
        return
    
    print("📦 Creating download package...")
    
    zip_filename = "gan_colab_results.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all result directories
        for root_dir in ["colab_models", "colab_augmented", "colab_samples"]:
            if os.path.exists(root_dir):
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
                print(f"  ✅ Added {root_dir}/")
    
    file_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
    print(f"📦 Package created: {zip_filename} ({file_size:.1f} MB)")
    
    try:
        files.download(zip_filename)
        print("📥 Download started!")
    except Exception as e:
        print(f"❌ Download error: {str(e)}")

# Main execution function
def main_colab():
    """Main function for Colab execution"""
    print("🌶️ GAN DATA AUGMENTATION - GOOGLE COLAB")
    print("=" * 60)
    
    # Setup dataset
    result = setup_colab_dataset()
    if not result:
        return
    
    dataset_path, available_classes = result
    
    if not available_classes:
        print("❌ No valid classes found!")
        return
    
    print(f"\n🎯 Available classes: {len(available_classes)}")
    for cls in available_classes:
        print(f"   - {cls}")
    
    # Ask user for training mode
    print(f"\n🚀 Training Options:")
    print(f"   1. Demo (1 class, 50 epochs, ~15 minutes)")
    print(f"   2. Full training ({len(available_classes)} classes, 150 epochs, ~3-4 hours)")
    
    return dataset_path, available_classes

def demo_training_colab(dataset_path, available_classes):
    """Run demo training for one class"""
    demo_class = available_classes[0]  # Use first available class
    demo_epochs = 50
    
    print(f"\n🎯 DEMO TRAINING")
    print(f"📁 Class: {demo_class}")
    print(f"🔄 Epochs: {demo_epochs}")
    print(f"⏰ Estimated time: 15-20 minutes")
    
    class_dir = os.path.join(dataset_path, demo_class)
    
    # Train
    generator, discriminator, g_losses, d_losses = train_gan_colab(
        class_dir, demo_class, demo_epochs, 150
    )
    
    # Generate some images
    generate_images_colab(generator, demo_class, 30)
    
    print(f"\n🎉 Demo completed for {demo_class}!")
    print(f"📁 Check colab_augmented/{demo_class}/ for results")
    
    return generator, g_losses, d_losses

if __name__ == "__main__":
    main_colab()
