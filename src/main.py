"""
Script utama untuk menjalankan seluruh pipeline GAN Data Augmentation
Untuk dataset penyakit tanaman cabai
"""

import os
import sys
import argparse
import torch
from datetime import datetime

def check_requirements():
    """Check apakah semua requirements terpenuhi"""
    required_packages = ['torch', 'torchvision', 'PIL', 'numpy', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install requirements:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("⚠️  CUDA not available, using CPU (training will be slower)")
        return False

def check_dataset():
    """Check dataset availability"""
    base_dir = r"c:\Riset Infromatika\Python V2"
    train_dir = os.path.join(base_dir, "Dataset Original", "train")
    
    if not os.path.exists(train_dir):
        print(f"❌ Dataset not found at {train_dir}")
        return False
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    class_counts = {}
    
    print("📊 Dataset check:")
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            print(f"   {class_name:<12}: {count:3d} images")
        else:
            print(f"   {class_name:<12}: Missing")
            return False
    
    total_images = sum(class_counts.values())
    print(f"   {'Total':<12}: {total_images:3d} images")
    
    return True

def setup_environment():
    """Setup lingkungan kerja"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    dirs = [
        "gan_models",
        "gan_samples", 
        "Dataset Augmented",
        "Dataset Augmented/train",
        "visualization_results"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create subdirectories for each class
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    for class_name in classes:
        for base_dir in ["gan_models", "gan_samples", "Dataset Augmented/train"]:
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    print("✅ Environment setup complete")

def print_banner():
    """Print banner aplikasi"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        GAN DATA AUGMENTATION                                ║
║              Untuk Klasifikasi Penyakit Tanaman Cabai                      ║
║                                                                              ║
║  🌶️  Menggunakan Deep Convolutional GAN (DCGAN)                           ║
║  🎯  Target: 200 gambar per kelas                                           ║
║  📊  Classes: Healthy, Leaf Curl, Leaf Spot, Whitefly, Yellowish          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="GAN Data Augmentation Pipeline")
    parser.add_argument("--mode", choices=['train', 'evaluate', 'visualize', 'all'], 
                       default='all', help="Mode operasi")
    parser.add_argument("--class", dest="class_name", 
                       choices=['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish', 'all'],
                       default='all', help="Kelas yang akan diproses")
    parser.add_argument("--epochs", type=int, default=300, help="Jumlah training epochs")
    parser.add_argument("--target", type=int, default=200, help="Target gambar per kelas")
    parser.add_argument("--skip-checks", action='store_true', help="Skip system checks")
    
    args = parser.parse_args()
    
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Mode: {args.mode}")
    print(f"🎯 Target: {args.target} images per class")
    print(f"🔄 Epochs: {args.epochs}")
    print()
    
    # System checks
    if not args.skip_checks:
        print("🔍 Performing system checks...")
        
        if not check_requirements():
            return 1
            
        check_cuda()
        
        if not check_dataset():
            return 1
        
        setup_environment()
        print()
    
    # Execute based on mode
    try:
        if args.mode in ['train', 'all']:
            print("🚀 Starting GAN training...")
            from run_augmentation import run_all_classes_augmentation, run_single_class_augmentation
            
            if args.class_name == 'all':
                run_all_classes_augmentation(args.epochs, args.target)
            else:
                run_single_class_augmentation(args.class_name, args.epochs, args.target)
        
        if args.mode in ['evaluate', 'all']:
            print("\n📊 Starting evaluation...")
            from evaluate_gan_quality import evaluate_all_classes
            evaluate_all_classes()
        
        if args.mode in ['visualize', 'all']:
            print("\n🖼️  Creating visualizations...")
            from visualize_results import visualize_all_results
            visualize_all_results()
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show final summary
        print("\n📁 Output directories:")
        print("   - Dataset Augmented/train/     : Augmented dataset")
        print("   - gan_models/                  : Trained GAN models")
        print("   - visualization_results/       : Visualization images")
        print("   - gan_samples/                 : Training progress samples")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
