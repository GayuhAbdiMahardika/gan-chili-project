"""
Script test untuk memastikan semua komponen GAN berfungsi
"""

import torch
import torch.nn as nn
import os
import sys

def test_torch_installation():
    """Test instalasi PyTorch"""
    print("Testing PyTorch installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch OK")
    return True

def test_dataset_access():
    """Test akses ke dataset"""
    print("\nTesting dataset access...")
    
    base_dir = r"c:\Riset Infromatika\Python V2"
    train_dir = os.path.join(base_dir, "Dataset Original", "train")
    
    if not os.path.exists(train_dir):
        print(f"❌ Dataset tidak ditemukan: {train_dir}")
        return False
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    total_images = 0
    
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {class_name:<12}: {count:3d} images")
            total_images += count
        else:
            print(f"❌ Kelas {class_name} tidak ditemukan")
            return False
    
    print(f"  {'Total':<12}: {total_images:3d} images")
    print("✅ Dataset OK")
    return True

def test_gan_components():
    """Test komponen GAN"""
    print("\nTesting GAN components...")
    
    try:
        # Import modules
        sys.path.append(r"c:\Riset Infromatika\Python V2")
        from gan_data_augmentation import Generator, Discriminator, weights_init
        
        # Test Generator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nz, ngf, nc = 100, 64, 3
        
        generator = Generator(nz, ngf, nc).to(device)
        generator.apply(weights_init)
        
        # Test forward pass
        noise = torch.randn(4, nz, 1, 1, device=device)
        fake_images = generator(noise)
        
        print(f"  Generator output shape: {fake_images.shape}")
        assert fake_images.shape == (4, 3, 64, 64), "Generator output shape incorrect"
        
        # Test Discriminator
        ndf = 64
        discriminator = Discriminator(nc, ndf).to(device)
        discriminator.apply(weights_init)
        
        # Test forward pass
        disc_output = discriminator(fake_images)
        print(f"  Discriminator output shape: {disc_output.shape}")
        assert disc_output.shape == (4,), "Discriminator output shape incorrect"
        
        print("✅ GAN components OK")
        return True
        
    except Exception as e:
        print(f"❌ Error testing GAN components: {str(e)}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    
    try:
        sys.path.append(r"c:\Riset Infromatika\Python V2")
        from gan_data_augmentation import ChiliDataset
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Test dataset loading
        train_dir = r"c:\Riset Infromatika\Python V3\Dataset Original\train\healthy"
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = ChiliDataset(train_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test loading one batch
        for batch in dataloader:
            print(f"  Batch shape: {batch.shape}")
            assert batch.shape[1:] == (3, 64, 64), "Batch shape incorrect"
            break
        
        print(f"  Dataset size: {len(dataset)}")
        print("✅ Data loading OK")
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loading: {str(e)}")
        return False

def test_directories():
    """Test pembuatan direktori"""
    print("\nTesting directory creation...")
    
    base_dir = r"c:\Riset Infromatika\Python V2"
    test_dirs = [
        "test_gan_models",
        "test_gan_samples",
        "test_output"
    ]
    
    try:
        for dir_name in test_dirs:
            full_path = os.path.join(base_dir, dir_name)
            os.makedirs(full_path, exist_ok=True)
            assert os.path.exists(full_path), f"Failed to create {full_path}"
        
        # Cleanup test directories
        import shutil
        for dir_name in test_dirs:
            full_path = os.path.join(base_dir, dir_name)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
        
        print("✅ Directory operations OK")
        return True
        
    except Exception as e:
        print(f"❌ Error testing directories: {str(e)}")
        return False

def main():
    """Jalankan semua test"""
    print("="*50)
    print("GAN DATA AUGMENTATION - SYSTEM TEST")
    print("="*50)
    
    tests = [
        test_torch_installation,
        test_dataset_access,
        test_directories,
        test_gan_components,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {str(e)}")
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Semua test berhasil! Sistem siap untuk training GAN.")
        print("\nUntuk memulai augmentasi data:")
        print("python main.py")
        print("\nAtau untuk training satu kelas:")
        print("python main.py --mode train --class healthy --epochs 300")
    else:
        print("⚠️  Beberapa test gagal. Periksa error di atas.")
        print("Pastikan semua requirements terinstall dan dataset tersedia.")
    
    print("="*50)

if __name__ == "__main__":
    main()
