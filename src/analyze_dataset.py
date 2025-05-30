"""
Script untuk analisis dataset penyakit tanaman cabai
Memberikan insight tentang karakteristik dataset sebelum augmentasi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from collections import Counter
import cv2

def analyze_image_properties(dataset_path):
    """Analisis properti gambar dalam dataset"""
    
    print("=== ANALISIS PROPERTI GAMBAR ===")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    all_sizes = []
    all_aspects = []
    all_file_sizes = []
    class_stats = {}
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            continue
            
        sizes = []
        aspects = []
        file_sizes = []
        mean_colors = []
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nAnalyzing {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                # File size
                file_size = os.path.getsize(img_path) / 1024  # KB
                file_sizes.append(file_size)
                all_file_sizes.append(file_size)
                
                # Image properties
                with Image.open(img_path) as img:
                    width, height = img.size
                    sizes.append((width, height))
                    all_sizes.append((width, height))
                    
                    aspect_ratio = width / height
                    aspects.append(aspect_ratio)
                    all_aspects.append(aspect_ratio)
                    
                    # Color analysis
                    img_rgb = img.convert('RGB')
                    img_array = np.array(img_rgb)
                    mean_color = np.mean(img_array, axis=(0, 1))
                    mean_colors.append(mean_color)
                    
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
        
        # Statistics untuk kelas ini
        if sizes:
            widths, heights = zip(*sizes)
            
            class_stats[class_name] = {
                'count': len(sizes),
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'avg_aspect': np.mean(aspects),
                'avg_file_size': np.mean(file_sizes),
                'avg_color': np.mean(mean_colors, axis=0) if mean_colors else [0, 0, 0]
            }
    
    # Print summary
    print(f"\n{'Class':<12} {'Count':<6} {'Avg Size':<12} {'Aspect':<8} {'File Size':<10} {'Avg Color (RGB)'}")
    print("-" * 80)
    
    for class_name, stats in class_stats.items():
        print(f"{class_name:<12} {stats['count']:<6} "
              f"{stats['avg_width']:.0f}x{stats['avg_height']:.0f}{'':>4} "
              f"{stats['avg_aspect']:.2f}{'':>4} "
              f"{stats['avg_file_size']:.1f} KB{'':>3} "
              f"({stats['avg_color'][0]:.0f}, {stats['avg_color'][1]:.0f}, {stats['avg_color'][2]:.0f})")
    
    return class_stats, all_sizes, all_aspects, all_file_sizes

def plot_dataset_analysis(class_stats, all_sizes, all_aspects, all_file_sizes):
    """Plot analisis dataset"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Analysis - Penyakit Tanaman Cabai', fontsize=16, fontweight='bold')
    
    # 1. Count per class
    classes = list(class_stats.keys())
    counts = [class_stats[c]['count'] for c in classes]
    
    bars = ax1.bar(classes, counts, color=['green', 'orange', 'red', 'blue', 'yellow'], alpha=0.7)
    ax1.set_title('Jumlah Gambar per Kelas')
    ax1.set_ylabel('Jumlah Gambar')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # 2. Image size distribution
    widths, heights = zip(*all_sizes)
    ax2.scatter(widths, heights, alpha=0.6, s=30)
    ax2.set_title('Distribusi Ukuran Gambar')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Aspect ratio distribution
    ax3.hist(all_aspects, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Distribusi Aspect Ratio')
    ax3.set_xlabel('Aspect Ratio (Width/Height)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(all_aspects), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_aspects):.2f}')
    ax3.legend()
    
    # 4. File size distribution
    ax4.hist(all_file_sizes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title('Distribusi Ukuran File')
    ax4.set_xlabel('File Size (KB)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.mean(all_file_sizes), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_file_sizes):.1f} KB')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Dataset analysis plot saved as 'dataset_analysis.png'")

def analyze_color_distribution(dataset_path):
    """Analisis distribusi warna per kelas"""
    
    print("\n=== ANALISIS DISTRIBUSI WARNA ===")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            continue
        
        all_colors = {'R': [], 'G': [], 'B': []}
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample subset untuk efisiensi
        sample_files = image_files[:min(20, len(image_files))]
        
        for img_file in sample_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert('RGB')
                    img_array = np.array(img_rgb)
                    
                    # Flatten and sample pixels
                    pixels = img_array.reshape(-1, 3)
                    sample_pixels = pixels[::100]  # Sample setiap 100 pixel
                    
                    all_colors['R'].extend(sample_pixels[:, 0])
                    all_colors['G'].extend(sample_pixels[:, 1])
                    all_colors['B'].extend(sample_pixels[:, 2])
                    
            except Exception as e:
                continue
        
        # Plot histogram warna
        if all_colors['R']:
            axes[i].hist(all_colors['R'], bins=50, alpha=0.7, color='red', label='Red')
            axes[i].hist(all_colors['G'], bins=50, alpha=0.7, color='green', label='Green')
            axes[i].hist(all_colors['B'], bins=50, alpha=0.7, color='blue', label='Blue')
            
            axes[i].set_title(f'{class_name.title()}\nColor Distribution')
            axes[i].set_xlabel('Pixel Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Print mean colors
            mean_r = np.mean(all_colors['R'])
            mean_g = np.mean(all_colors['G'])
            mean_b = np.mean(all_colors['B'])
            
            print(f"{class_name:<12}: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('color_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Color distribution analysis saved as 'color_distribution_analysis.png'")

def check_data_quality(dataset_path):
    """Check kualitas data dan potensi masalah"""
    
    print("\n=== CHECK KUALITAS DATA ===")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    issues = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            issues.append(f"❌ Folder {class_name} tidak ditemukan")
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nChecking {class_name}:")
        
        # Check jumlah gambar
        if len(image_files) < 50:
            issues.append(f"⚠️  {class_name}: Hanya {len(image_files)} gambar (kurang dari 50)")
        
        # Check ukuran gambar
        small_images = 0
        corrupted_images = 0
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    if width < 64 or height < 64:
                        small_images += 1
                        
            except Exception as e:
                corrupted_images += 1
                issues.append(f"❌ {class_name}: {img_file} corrupted - {str(e)}")
        
        if small_images > 0:
            issues.append(f"⚠️  {class_name}: {small_images} gambar berukuran < 64px")
        
        if corrupted_images > 0:
            issues.append(f"❌ {class_name}: {corrupted_images} gambar corrupted")
        else:
            print(f"  ✅ Semua {len(image_files)} gambar valid")
    
    print(f"\n=== RINGKASAN KUALITAS DATA ===")
    if issues:
        print("Issues ditemukan:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Tidak ada masalah ditemukan pada dataset!")
    
    return len(issues) == 0

def main():
    """Main function untuk analisis dataset"""
    
    dataset_path = r"c:\UPN\SMT 6\Riset Infromatika\Python V3\GAN_Project\Dataset Original\train"
    
    print("="*60)
    print("ANALISIS DATASET PENYAKIT TANAMAN CABAI")
    print("="*60)
    
    # 1. Analisis properti gambar
    class_stats, all_sizes, all_aspects, all_file_sizes = analyze_image_properties(dataset_path)
    
    # 2. Plot analisis
    plot_dataset_analysis(class_stats, all_sizes, all_aspects, all_file_sizes)
    
    # 3. Analisis warna
    analyze_color_distribution(dataset_path)
    
    # 4. Check kualitas
    data_quality_ok = check_data_quality(dataset_path)
    
    # 5. Rekomendasi
    print(f"\n=== REKOMENDASI UNTUK GAN TRAINING ===")
    
    total_images = sum([stats['count'] for stats in class_stats.values()])
    avg_size = np.mean([np.sqrt(w*h) for w, h in all_sizes])
    
    print(f"Total gambar: {total_images}")
    print(f"Rata-rata ukuran: {avg_size:.0f}px")
    
    if data_quality_ok:
        print("✅ Dataset dalam kondisi baik untuk training GAN")
    else:
        print("⚠️  Ada issues pada dataset yang perlu diperbaiki")
    
    if avg_size > 256:
        print("💡 Ukuran gambar cukup besar, pertimbangkan resize ke 128x128 untuk efisiensi")
    elif avg_size < 64:
        print("⚠️  Ukuran gambar kecil, hasil GAN mungkin kurang detail")
    
    if total_images < 500:
        print("💡 Dataset relatif kecil, GAN augmentation sangat direkomendasikan")
    
    print(f"\nUntuk memulai augmentasi:")
    print(f"python main.py --target 200 --epochs 300")

if __name__ == "__main__":
    main()
