"""
Troubleshooting Guide dan FAQ untuk GAN Data Augmentation
"""

import os
import torch
import psutil
import time

def check_system_resources():
    """Check ketersediaan resource sistem"""
    
    print("=== SYSTEM RESOURCE CHECK ===")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM Total: {memory.total / (1024**3):.1f} GB")
    print(f"RAM Available: {memory.available / (1024**3):.1f} GB")
    print(f"RAM Usage: {memory.percent}%")
    
    if memory.available / (1024**3) < 4:
        print("⚠️  RAM kurang dari 4GB, training mungkin lambat")
    else:
        print("✅ RAM mencukupi")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"\nDisk Space Available: {disk.free / (1024**3):.1f} GB")
    
    if disk.free / (1024**3) < 5:
        print("⚠️  Disk space kurang dari 5GB")
    else:
        print("✅ Disk space mencukupi")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"\nCPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent}%")
    
    # GPU
    print(f"\nGPU CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Device: {gpu_name}")
        print("✅ GPU tersedia, training akan lebih cepat")
    else:
        print("⚠️  GPU tidak tersedia, akan menggunakan CPU")

def diagnose_common_issues():
    """Diagnosa masalah umum"""
    
    print("\n=== COMMON ISSUES DIAGNOSIS ===")
    
    issues_found = []
    
    # Check dataset
    dataset_path = r"c:\Riset Infromatika\Python V3\Dataset Original\train"
    if not os.path.exists(dataset_path):
        issues_found.append("❌ Dataset Original tidak ditemukan")
    else:
        classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                issues_found.append(f"❌ Folder {class_name} tidak ada")
            else:
                img_count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count == 0:
                    issues_found.append(f"❌ Folder {class_name} kosong")
                elif img_count < 50:
                    issues_found.append(f"⚠️  Folder {class_name} hanya {img_count} gambar")
    
    # Check Python packages
    required_packages = ['torch', 'torchvision', 'PIL', 'numpy', 'matplotlib', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues_found.append(f"❌ Package {package} tidak terinstall")
    
    # Check file permissions
    try:
        test_file = "test_write_permission.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        issues_found.append(f"❌ Tidak bisa write di direktori: {str(e)}")
    
    if issues_found:
        print("Issues ditemukan:")
        for issue in issues_found:
            print(f"  {issue}")
        return False
    else:
        print("✅ Tidak ada masalah umum ditemukan")
        return True

def get_training_recommendations():
    """Berikan rekomendasi parameter training"""
    
    print("\n=== TRAINING RECOMMENDATIONS ===")
    
    # Check system specs
    memory_gb = psutil.virtual_memory().total / (1024**3)
    has_gpu = torch.cuda.is_available()
    cpu_cores = psutil.cpu_count()
    
    print("Berdasarkan spesifikasi sistem:")
    
    if has_gpu:
        print("🚀 GPU tersedia - Gunakan mode normal:")
        print("   python main.py --epochs 300 --target 200")
        recommended_batch = 32
        recommended_epochs = 300
    else:
        print("🐌 CPU only - Gunakan mode CPU optimized:")
        print("   python cpu_training.py")
        recommended_batch = 8
        recommended_epochs = 100
    
    if memory_gb < 8:
        print("📉 RAM terbatas - Kurangi batch size")
        recommended_batch = min(recommended_batch, 16)
    
    print(f"\nRecommended settings:")
    print(f"  Batch size: {recommended_batch}")
    print(f"  Epochs: {recommended_epochs}")
    print(f"  Target images: 200 per class")
    
    # Time estimation
    if has_gpu:
        estimated_time = 1.5 * 5  # 1.5 hours per class
    else:
        estimated_time = 3 * 5   # 3 hours per class
    
    print(f"\nEstimated total time: {estimated_time:.1f} hours")

def create_quick_test():
    """Buat test cepat untuk memastikan semua berjalan"""
    
    print("\n=== QUICK FUNCTIONALITY TEST ===")
    
    try:
        # Test 1: Import modules
        print("Testing imports...")
        from gan_data_augmentation import Generator, Discriminator
        print("✅ Imports OK")
        
        # Test 2: Create small models
        print("Testing model creation...")
        device = torch.device("cpu")
        gen = Generator(100, 32, 3).to(device)
        disc = Discriminator(3, 32).to(device)
        print("✅ Model creation OK")
        
        # Test 3: Forward pass
        print("Testing forward pass...")
        noise = torch.randn(1, 100, 1, 1)
        fake_img = gen(noise)
        output = disc(fake_img)
        print(f"✅ Forward pass OK - Generated shape: {fake_img.shape}")
        
        # Test 4: Data loading
        print("Testing data loading...")
        from gan_data_augmentation import ChiliDataset
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset_path = r"c:\Riset Infromatika\Python V3\Dataset Original\train\healthy"
        dataset = ChiliDataset(dataset_path, transform=transform)
        print(f"✅ Data loading OK - Dataset size: {len(dataset)}")
        
        print("\n🎉 All functionality tests passed!")
        print("Sistem siap untuk training GAN!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def show_usage_examples():
    """Tampilkan contoh penggunaan"""
    
    print("\n=== USAGE EXAMPLES ===")
    
    print("1. Training lengkap semua kelas:")
    print("   python main.py")
    print()
    
    print("2. Training satu kelas saja:")
    print("   python main.py --mode train --class healthy --epochs 300")
    print()
    
    print("3. Training cepat dengan CPU:")
    print("   python cpu_training.py demo")
    print()
    
    print("4. Monitoring progress:")
    print("   python monitor.py monitor")
    print()
    
    print("5. Evaluasi hasil:")
    print("   python main.py --mode evaluate")
    print()
    
    print("6. Visualisasi hasil:")
    print("   python main.py --mode visualize")
    print()
    
    print("7. Analisis dataset:")
    print("   python analyze_dataset.py")
    print()

def create_troubleshooting_guide():
    """Buat panduan troubleshooting"""
    
    print("\n=== TROUBLESHOOTING GUIDE ===")
    
    common_errors = {
        "CUDA out of memory": [
            "Kurangi BATCH_SIZE di gan_data_augmentation.py",
            "Gunakan mode CPU: python cpu_training.py",
            "Restart Python dan clear cache: torch.cuda.empty_cache()"
        ],
        
        "Training sangat lambat": [
            "Pastikan menggunakan GPU jika tersedia",
            "Kurangi jumlah epoch untuk test: --epochs 50",
            "Gunakan cpu_training.py untuk optimasi CPU"
        ],
        
        "FileNotFoundError": [
            "Cek path dataset di main.py dan gan_data_augmentation.py",
            "Pastikan struktur folder sesuai dengan README",
            "Gunakan absolute path jika perlu"
        ],
        
        "ImportError": [
            "Install requirements: pip install -r requirements.txt",
            "Update pip: python -m pip install --upgrade pip",
            "Cek virtual environment aktif"
        ],
        
        "Model tidak converge": [
            "Tambah jumlah epoch training",
            "Adjust learning rate (0.0001 - 0.0005)",
            "Cek balance antara generator dan discriminator loss"
        ],
        
        "Generated images buruk": [
            "Training lebih lama (500+ epochs)",
            "Cek kualitas dataset asli",
            "Adjust network architecture (NGF, NDF)",
            "Tambah regularization techniques"
        ]
    }
    
    for error, solutions in common_errors.items():
        print(f"\n🔧 {error}:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")

def main():
    """Main troubleshooting function"""
    
    print("="*60)
    print("GAN DATA AUGMENTATION - TROUBLESHOOTING & FAQ")
    print("="*60)
    
    # System check
    check_system_resources()
    
    # Diagnose issues
    system_ok = diagnose_common_issues()
    
    # Quick test
    if system_ok:
        test_ok = create_quick_test()
        
        if test_ok:
            # Show recommendations
            get_training_recommendations()
            show_usage_examples()
        else:
            print("\n❌ Quick test failed. Check troubleshooting guide below.")
            create_troubleshooting_guide()
    else:
        print("\n❌ System issues found. Please fix them first.")
        create_troubleshooting_guide()
    
    print("\n" + "="*60)
    print("Jika masih ada masalah, cek README.md untuk dokumentasi lengkap")
    print("="*60)

if __name__ == "__main__":
    main()
