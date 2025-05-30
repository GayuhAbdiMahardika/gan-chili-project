"""
Script konfigurasi untuk menjalankan GAN Data Augmentation
Untuk dataset penyakit tanaman cabai
"""

import os
import argparse
from gan_data_augmentation import train_gan, generate_images, Generator
import torch

def setup_directories():
    """Buat direktori yang diperlukan"""
    dirs = [
        "gan_models",
        "gan_samples", 
        "Dataset Augmented/train",
        "Dataset Augmented/train/healthy",
        "Dataset Augmented/train/leaf curl",
        "Dataset Augmented/train/leaf spot", 
        "Dataset Augmented/train/whitefly",
        "Dataset Augmented/train/yellowish"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def run_single_class_augmentation(class_name, train_epochs=300, target_images=200):
    """
    Jalankan augmentasi untuk satu kelas saja
    
    Args:
        class_name: Nama kelas ('healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish')
        train_epochs: Jumlah epoch untuk training
        target_images: Target jumlah gambar setelah augmentasi
    """
    
    # Path setup
    base_dir = r"c:\Riset Infromatika\Python V2"
    train_dir = os.path.join(base_dir, "Dataset Original", "train")
    class_dir = os.path.join(train_dir, class_name)
    
    if not os.path.exists(class_dir):
        print(f"Error: Directory {class_dir} tidak ditemukan!")
        return
    
    # Hitung jumlah gambar yang sudah ada
    existing_images = len([f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Kelas: {class_name}")
    print(f"Gambar existing: {existing_images}")
    print(f"Target gambar: {target_images}")
    print(f"Perlu generate: {max(0, target_images - existing_images)} gambar")
    
    if existing_images >= target_images:
        print(f"Kelas {class_name} sudah mencukupi!")
        return
    
    # Setup direktori
    setup_directories()
    
    # Train GAN
    print(f"\nMemulai training GAN untuk kelas {class_name}...")
    generator, discriminator, g_losses, d_losses = train_gan(
        class_dir, class_name, target_images
    )
    
    # Generate gambar tambahan
    num_to_generate = target_images - existing_images
    output_dir = os.path.join(base_dir, "Dataset Augmented", "train", class_name)
    
    # Copy gambar asli ke folder augmented
    import shutil
    for file in os.listdir(class_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(class_dir, file)
            dst = os.path.join(output_dir, file)
            shutil.copy2(src, dst)
    
    # Generate gambar baru
    generate_images(generator, class_name, num_to_generate, output_dir)
    
    # Verifikasi hasil
    final_count = len([f for f in os.listdir(output_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nHasil augmentasi kelas {class_name}:")
    print(f"- Gambar asli: {existing_images}")
    print(f"- Gambar generated: {num_to_generate}")
    print(f"- Total gambar: {final_count}")
    print(f"- Target tercapai: {'✓' if final_count >= target_images else '✗'}")

def run_all_classes_augmentation(train_epochs=300, target_images=200):
    """Jalankan augmentasi untuk semua kelas"""
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    print("=== AUGMENTASI DATA PENYAKIT TANAMAN CABAI ===")
    print(f"Target per kelas: {target_images} gambar")
    print(f"Training epochs: {train_epochs}")
    print("=" * 50)
    
    for i, class_name in enumerate(classes, 1):
        print(f"\n[{i}/{len(classes)}] Processing kelas: {class_name}")
        print("-" * 30)
        
        try:
            run_single_class_augmentation(class_name, train_epochs, target_images)
        except Exception as e:
            print(f"Error saat memproses kelas {class_name}: {str(e)}")
            continue
        
        print(f"Selesai kelas {class_name}")
        print("=" * 50)
    
    print("\n🎉 AUGMENTASI DATA SELESAI! 🎉")
    print("\nRingkasan:")
    
    # Tampilkan ringkasan hasil
    base_dir = r"c:\Riset Infromatika\Python V2"
    augmented_dir = os.path.join(base_dir, "Dataset Augmented", "train")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in classes:
        # Original count
        original_dir = os.path.join(base_dir, "Dataset Original", "train", class_name)
        original_count = len([f for f in os.listdir(original_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(original_dir) else 0
        
        # Augmented count
        aug_dir = os.path.join(augmented_dir, class_name)
        aug_count = len([f for f in os.listdir(aug_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(aug_dir) else 0
        
        total_original += original_count
        total_augmented += aug_count
        
        print(f"- {class_name:12}: {original_count:3d} → {aug_count:3d} gambar")
    
    print(f"\nTotal dataset:")
    print(f"- Original : {total_original} gambar")
    print(f"- Augmented: {total_augmented} gambar")
    print(f"- Peningkatan: {((total_augmented/total_original - 1) * 100):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Data Augmentation untuk Penyakit Tanaman Cabai")
    parser.add_argument("--class", dest="class_name", type=str, 
                       choices=['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish', 'all'],
                       default='all', help="Kelas yang akan di-augmentasi")
    parser.add_argument("--epochs", type=int, default=300, help="Jumlah training epochs")
    parser.add_argument("--target", type=int, default=200, help="Target jumlah gambar per kelas")
    
    args = parser.parse_args()
    
    if args.class_name == 'all':
        run_all_classes_augmentation(args.epochs, args.target)
    else:
        run_single_class_augmentation(args.class_name, args.epochs, args.target)
