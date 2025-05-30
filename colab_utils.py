"""
Colab Utilities - Helper functions untuk Google Colab
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import shutil

try:
    from google.colab import files, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def check_colab_setup():
    """Check if Colab is properly setup"""
    print("🔍 CHECKING COLAB SETUP")
    print("=" * 40)
    
    # Check if in Colab
    if IN_COLAB:
        print("✅ Running in Google Colab")
    else:
        print("❌ Not running in Google Colab")
        return False
    
    # Check PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
    
    # Check disk space
    disk_usage = shutil.disk_usage("/content")
    free_gb = disk_usage.free / (1024**3)
    print(f"💾 Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 2:
        print("⚠️  Low disk space - consider cleanup")
    
    return True

def mount_drive_and_setup():
    """Mount Google Drive and setup data"""
    if not IN_COLAB:
        print("❌ This function only works in Google Colab")
        return None
    
    try:
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        
        # Common Drive paths
        drive_paths = [
            "/content/drive/MyDrive/Dataset Original",
            "/content/drive/MyDrive/GAN_Project/Dataset Original",
            "/content/drive/MyDrive/dataset",
            "/content/drive/MyDrive/data"
        ]
        
        print("\n🔍 Searching for dataset in Google Drive...")
        for path in drive_paths:
            if os.path.exists(path):
                print(f"✅ Found dataset at: {path}")
                return path
        
        print("❌ Dataset not found in common Drive locations")
        print("💡 Please check these locations in your Drive:")
        for path in drive_paths:
            print(f"   - {path}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error mounting drive: {str(e)}")
        return None

def upload_and_extract_dataset():
    """Upload and extract dataset ZIP file"""
    if not IN_COLAB:
        print("❌ This function only works in Google Colab")
        return None
    
    print("📁 UPLOAD DATASET")
    print("Please upload your dataset ZIP file")
    print("Expected structure inside ZIP:")
    print("  train/")
    print("  ├── healthy/")
    print("  ├── leaf curl/")
    print("  ├── leaf spot/")
    print("  ├── whitefly/")
    print("  └── yellowish/")
    
    try:
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            if filename.endswith('.zip'):
                print(f"📦 Extracting {filename}...")
                
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content')
                
                # Find extracted dataset
                possible_paths = [
                    "/content/train",
                    "/content/Dataset Original/train",
                    "/content/data/train"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"✅ Dataset extracted to: {path}")
                        return path
                
                # If not found, list contents and let user find it
                print("📁 Extracted contents:")
                for item in os.listdir("/content"):
                    if os.path.isdir(f"/content/{item}"):
                        print(f"   📂 {item}/")
                        # Check if it contains train folder
                        train_path = f"/content/{item}/train"
                        if os.path.exists(train_path):
                            print(f"      ✅ Found train folder: {train_path}")
                            return train_path
                
                break
        
        print("❌ No ZIP file uploaded or dataset structure not recognized")
        return None
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return None

def quick_dataset_preview(dataset_path):
    """Show quick preview of dataset"""
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Invalid dataset path")
        return
    
    print(f"👀 DATASET PREVIEW: {dataset_path}")
    print("=" * 50)
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    total_images = 0
    
    fig, axes = plt.subplots(1, len(classes), figsize=(20, 4))
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(image_files)
            total_images += count
            
            # Show sample image
            if image_files:
                sample_img = Image.open(os.path.join(class_path, image_files[0]))
                axes[i].imshow(sample_img)
                axes[i].set_title(f'{class_name}\n{count} images')
            else:
                axes[i].text(0.5, 0.5, 'No images', ha='center', va='center')
                axes[i].set_title(f'{class_name}\n0 images')
        else:
            axes[i].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[i].set_title(f'{class_name}\nNot found')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Total images: {total_images}")
    return total_images

def show_training_samples(class_name, sample_dir="colab_samples"):
    """Show training progress samples"""
    class_sample_dir = os.path.join(sample_dir, class_name)
    
    if not os.path.exists(class_sample_dir):
        print(f"❌ No samples found for {class_name}")
        return
    
    sample_files = [f for f in os.listdir(class_sample_dir) if f.endswith('.png')]
    sample_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not sample_files:
        print(f"❌ No sample files found for {class_name}")
        return
    
    # Show progress: first, middle, last samples
    indices = [0]
    if len(sample_files) > 2:
        indices.append(len(sample_files) // 2)
    if len(sample_files) > 1:
        indices.append(-1)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        sample_file = sample_files[idx]
        epoch = sample_file.split('_')[1].split('.')[0]
        
        img = plt.imread(os.path.join(class_sample_dir, sample_file))
        axes[i].imshow(img)
        axes[i].set_title(f'Epoch {epoch}')
        axes[i].axis('off')
    
    plt.suptitle(f'Training Progress - {class_name}')
    plt.tight_layout()
    plt.show()

def create_comparison_collage(original_dir, generated_dir, class_name, save_path=None):
    """Create comparison collage of original vs generated"""
    if not os.path.exists(original_dir) or not os.path.exists(generated_dir):
        print(f"❌ Missing directories for {class_name}")
        return
    
    # Get sample images
    orig_files = [f for f in os.listdir(original_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:8]
    gen_files = [f for f in os.listdir(generated_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:8]
    
    if not orig_files or not gen_files:
        print(f"❌ Insufficient images for comparison - {class_name}")
        return
    
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    # Original images
    for i in range(8):
        if i < len(orig_files):
            img = Image.open(os.path.join(original_dir, orig_files[i]))
            axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=14, rotation=0, ha='right')
    
    # Generated images
    for i in range(8):
        if i < len(gen_files):
            img = Image.open(os.path.join(generated_dir, gen_files[i]))
            axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Generated', fontsize=14, rotation=0, ha='right')
    
    plt.suptitle(f'Original vs Generated - {class_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison saved to {save_path}")
    
    plt.show()

def monitor_colab_training():
    """Monitor training progress in Colab"""
    print("📊 TRAINING MONITOR")
    print("=" * 40)
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # Check models
    print("\n🤖 Trained Models:")
    for class_name in classes:
        model_path = f"colab_models/{class_name}/generator.pth"
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  ✅ {class_name:15}: {file_size:.1f} MB")
        else:
            print(f"  ❌ {class_name:15}: Not trained")
    
    # Check generated images
    print("\n🎨 Generated Images:")
    total_generated = 0
    for class_name in classes:
        gen_dir = f"colab_augmented/{class_name}"
        if os.path.exists(gen_dir):
            count = len([f for f in os.listdir(gen_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_generated += count
            print(f"  📸 {class_name:15}: {count:3d} images")
        else:
            print(f"  ❌ {class_name:15}: No images")
    
    print(f"\n📊 Total generated images: {total_generated}")
    
    # Check disk usage
    disk_usage = shutil.disk_usage("/content")
    used_gb = (disk_usage.total - disk_usage.free) / (1024**3)
    free_gb = disk_usage.free / (1024**3)
    print(f"💾 Disk usage: {used_gb:.1f} GB used, {free_gb:.1f} GB free")

def cleanup_colab_temp_files():
    """Cleanup temporary files to save space"""
    print("🧹 CLEANING TEMPORARY FILES")
    
    temp_dirs = [
        "/tmp",
        "/content/sample_data",
        "__pycache__"
    ]
    
    freed_space = 0
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                # Calculate size before deletion
                size_before = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(temp_dir)
                    for filename in filenames
                ) / (1024 * 1024)  # MB
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                freed_space += size_before
                print(f"  🗑️  Cleaned {temp_dir}: {size_before:.1f} MB")
                
            except Exception as e:
                print(f"  ⚠️  Error cleaning {temp_dir}: {str(e)}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  🔄 Cleared CUDA cache")
    
    print(f"\n✅ Total space freed: {freed_space:.1f} MB")

def create_colab_summary_report():
    """Create summary report of training results"""
    print("📋 CREATING SUMMARY REPORT")
    
    report = []
    report.append("=" * 60)
    report.append("🌶️ GAN DATA AUGMENTATION - COLAB RESULTS")
    report.append("=" * 60)
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # Training status
    report.append("\n🤖 TRAINING STATUS:")
    trained_classes = 0
    for class_name in classes:
        model_path = f"colab_models/{class_name}/generator.pth"
        if os.path.exists(model_path):
            report.append(f"  ✅ {class_name:15}: Completed")
            trained_classes += 1
        else:
            report.append(f"  ❌ {class_name:15}: Not trained")
    
    # Generated images count
    report.append("\n🎨 GENERATED IMAGES:")
    total_generated = 0
    for class_name in classes:
        gen_dir = f"colab_augmented/{class_name}"
        if os.path.exists(gen_dir):
            count = len([f for f in os.listdir(gen_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_generated += count
            report.append(f"  📸 {class_name:15}: {count:3d} images")
        else:
            report.append(f"  ❌ {class_name:15}: No images")
    
    # Summary statistics
    report.append(f"\n📊 SUMMARY:")
    report.append(f"  Classes trained: {trained_classes}/5")
    report.append(f"  Total generated: {total_generated} images")
    report.append(f"  Success rate: {(trained_classes/5)*100:.1f}%")
    
    # File structure
    report.append(f"\n📁 OUTPUT STRUCTURE:")
    report.append(f"  colab_models/      - Trained GAN models")
    report.append(f"  colab_augmented/   - Generated images")
    report.append(f"  colab_samples/     - Training samples")
    
    # Save report
    report_text = "\n".join(report)
    with open("colab_training_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n💾 Report saved to: colab_training_report.txt")
    
    return report_text

# Quick setup functions
def quick_colab_setup():
    """One-click setup for Colab"""
    print("🚀 QUICK COLAB SETUP")
    print("=" * 40)
    
    # Check setup
    if not check_colab_setup():
        return None
    
    # Try to find dataset
    dataset_path = None
    
    # Option 1: Check if already uploaded
    possible_paths = [
        "/content/Dataset Original/train",
        "/content/train",
        "/content/data/train"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found existing dataset: {path}")
            break
    
    if not dataset_path:
        print("📁 Dataset not found. Choose upload method:")
        print("  1. Upload ZIP file")
        print("  2. Mount Google Drive")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            dataset_path = upload_and_extract_dataset()
        elif choice == "2":
            dataset_path = mount_drive_and_setup()
        else:
            print("❌ Invalid choice")
            return None
    
    if dataset_path:
        print(f"\n✅ Dataset ready: {dataset_path}")
        quick_dataset_preview(dataset_path)
        return dataset_path
    else:
        print("❌ Failed to setup dataset")
        return None

if __name__ == "__main__":
    # Auto setup when imported
    print("🛠️ Colab Utilities Loaded")
    print("Available functions:")
    print("  - quick_colab_setup()")
    print("  - check_colab_setup()")
    print("  - monitor_colab_training()")
    print("  - create_colab_summary_report()")
