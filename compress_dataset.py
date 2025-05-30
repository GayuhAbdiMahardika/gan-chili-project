#!/usr/bin/env python3
"""
Script untuk compress dataset menjadi ZIP file untuk upload ke Colab
"""

import zipfile
import os
from pathlib import Path

def compress_dataset():
    """Compress Dataset Original folder to ZIP"""
    
    dataset_path = Path("Dataset Original")
    zip_path = Path("Dataset_Original.zip")
    
    if not dataset_path.exists():
        print("❌ Folder 'Dataset Original' tidak ditemukan!")
        print("   Pastikan script ini dijalankan di folder yang sama dengan 'Dataset Original'")
        return False
    
    print("📦 Compressing dataset...")
    print(f"   Source: {dataset_path.absolute()}")
    print(f"   Output: {zip_path.absolute()}")
    
    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        file_count = 0
        
        # Walk through all files in dataset
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = Path(root) / file
                    # Add file to ZIP with relative path
                    arcname = file_path.relative_to(".")
                    zipf.write(file_path, arcname)
                    file_count += 1
                    
                    if file_count % 10 == 0:
                        print(f"   📁 Processed {file_count} files...")
    
    # Check ZIP file size
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Dataset berhasil di-compress!")
    print(f"   📁 Total files: {file_count}")
    print(f"   📦 ZIP size: {zip_size:.1f} MB")
    print(f"   💾 File location: {zip_path.absolute()}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Upload file '{zip_path.name}' ke Google Colab")
    print(f"   2. Atau upload ke Google Drive")
    print(f"   3. Jalankan cell ZIP upload di notebook Colab")
    
    return True

def verify_dataset_structure():
    """Verify dataset has correct structure"""
    
    dataset_path = Path("Dataset Original")
    required_structure = {
        "train": ["healthy", "leaf curl", "leaf spot", "whitefly", "yellowish"],
        "test": ["healthy", "leaf curl", "leaf spot", "whitefly", "yellowish"],
        "val": ["healthy", "leaf curl", "leaf spot", "whitefly", "yellowish"]
    }
    
    print("🔍 Verifying dataset structure...")
    
    for split, classes in required_structure.items():
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ Missing folder: {split_path}")
            return False
        
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"❌ Missing class folder: {class_path}")
                return False
            
            # Count files
            image_files = list(class_path.glob("*.jpg")) + \
                         list(class_path.glob("*.jpeg")) + \
                         list(class_path.glob("*.png"))
            
            print(f"   📊 {split}/{class_name}: {len(image_files)} files")
    
    print("✅ Dataset structure verified!")
    return True

if __name__ == "__main__":
    print("🌶️ GAN Chili Dataset Compressor")
    print("=" * 40)
    
    # Verify structure first
    if verify_dataset_structure():
        print()
        # Compress dataset
        if compress_dataset():
            print("\n🎉 Ready for Colab upload!")
        else:
            print("\n❌ Compression failed!")
    else:
        print("\n❌ Dataset structure verification failed!")
        print("   Please check your dataset folder structure.")
