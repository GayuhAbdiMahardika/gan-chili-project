"""
Script untuk visualisasi hasil augmentasi data GAN
Membuat grid gambar untuk membandingkan data asli vs generated
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import random

def create_comparison_grid(original_folder, generated_folder, class_name, num_samples=8):
    """
    Buat grid perbandingan gambar asli vs generated
    
    Args:
        original_folder: Path ke folder gambar asli
        generated_folder: Path ke folder gambar generated
        class_name: Nama kelas
        num_samples: Jumlah sample yang ditampilkan
    """
    
    # Get image files
    original_files = [f for f in os.listdir(original_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    generated_files = [f for f in os.listdir(generated_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'generated' in f.lower()]
    
    if len(original_files) == 0 or len(generated_files) == 0:
        print(f"Warning: Tidak ada gambar ditemukan untuk kelas {class_name}")
        return
    
    # Random sample
    original_sample = random.sample(original_files, min(num_samples, len(original_files)))
    generated_sample = random.sample(generated_files, min(num_samples, len(generated_files)))
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, num_samples, hspace=0.3, wspace=0.1)
    
    fig.suptitle(f'Data Augmentation Results - {class_name.title()}', fontsize=16, fontweight='bold')
    
    # Original images (top row)
    for i, img_file in enumerate(original_sample):
        ax = fig.add_subplot(gs[0, i])
        img_path = os.path.join(original_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.set_title('Original' if i == 0 else '', fontsize=12)
        ax.axis('off')
    
    # Generated images (bottom row)
    for i, img_file in enumerate(generated_sample):
        ax = fig.add_subplot(gs[1, i])
        img_path = os.path.join(generated_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.set_title('Generated' if i == 0 else '', fontsize=12)
        ax.axis('off')
    
    # Save comparison
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/comparison_{class_name.replace(' ', '_')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved for {class_name}")

def create_dataset_overview():
    """Buat overview lengkap dari semua dataset"""
    
    base_dir = r"c:\Riset Infromatika\Python V2"
    original_base = os.path.join(base_dir, "Dataset Original", "train")
    generated_base = os.path.join(base_dir, "Dataset Augmented", "train")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # Count images
    original_counts = []
    augmented_counts = []
    
    for class_name in classes:
        # Original count
        original_folder = os.path.join(original_base, class_name)
        original_count = len([f for f in os.listdir(original_folder) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(original_folder) else 0
        
        # Augmented count
        augmented_folder = os.path.join(generated_base, class_name)
        augmented_count = len([f for f in os.listdir(augmented_folder) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(augmented_folder) else 0
        
        original_counts.append(original_count)
        augmented_counts.append(augmented_count)
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Chart 1: Original vs Augmented counts
    bars1 = ax1.bar(x - width/2, original_counts, width, label='Original', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, augmented_counts, width, label='Augmented', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Disease Classes')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Size: Original vs Augmented')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.title() for c in classes], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Chart 2: Improvement percentage
    improvements = [(aug - orig) / orig * 100 if orig > 0 else 0 
                   for orig, aug in zip(original_counts, augmented_counts)]
    
    bars3 = ax2.bar(x, improvements, color='green', alpha=0.7)
    ax2.set_xlabel('Disease Classes')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Data Augmentation Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.title() for c in classes], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save overview
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n=== DATASET OVERVIEW ===")
    print(f"{'Class':<12} {'Original':<10} {'Augmented':<12} {'Improvement':<12}")
    print("-" * 50)
    
    total_original = sum(original_counts)
    total_augmented = sum(augmented_counts)
    
    for i, class_name in enumerate(classes):
        print(f"{class_name.title():<12} {original_counts[i]:<10} {augmented_counts[i]:<12} {improvements[i]:<12.1f}%")
    
    print("-" * 50)
    print(f"{'Total':<12} {total_original:<10} {total_augmented:<12} {((total_augmented/total_original - 1) * 100):<12.1f}%")
    
    print(f"\nDataset overview saved to visualization_results/dataset_overview.png")

def create_training_progress_visualization():
    """Visualisasi progress training dari semua kelas"""
    
    # Check if gan_losses files exist
    loss_files = [f for f in os.listdir('.') if f.startswith('gan_losses_') and f.endswith('.png')]
    
    if not loss_files:
        print("No training loss files found. Run training first.")
        return
    
    # Create combined loss visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    for i, class_name in enumerate(classes):
        loss_file = f"gan_losses_{class_name}.png"
        if os.path.exists(loss_file):
            # For now, just show placeholder
            axes[i].text(0.5, 0.5, f'{class_name.title()}\nTraining Completed', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            axes[i].set_title(f'{class_name.title()} - Training Status')
        else:
            axes[i].text(0.5, 0.5, f'{class_name.title()}\nNot Trained Yet', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            axes[i].set_title(f'{class_name.title()} - Training Status')
        
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].axis('off')
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.suptitle('GAN Training Progress Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_progress.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training progress visualization saved")

def visualize_all_results():
    """Generate semua visualisasi"""
    
    print("=== GENERATING VISUALIZATIONS ===")
    
    base_dir = r"c:\Riset Infromatika\Python V2"
    original_base = os.path.join(base_dir, "Dataset Original", "train")
    generated_base = os.path.join(base_dir, "Dataset Augmented", "train")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # 1. Create dataset overview
    print("1. Creating dataset overview...")
    create_dataset_overview()
    
    # 2. Create comparison grids for each class
    print("2. Creating comparison grids...")
    for class_name in classes:
        original_folder = os.path.join(original_base, class_name)
        generated_folder = os.path.join(generated_base, class_name)
        
        if os.path.exists(original_folder) and os.path.exists(generated_folder):
            create_comparison_grid(original_folder, generated_folder, class_name)
        else:
            print(f"Skipping {class_name} - folders not found")
    
    # 3. Create training progress visualization
    print("3. Creating training progress visualization...")
    create_training_progress_visualization()
    
    print("\n✅ All visualizations completed!")
    print("Check the 'visualization_results' folder for output images.")

if __name__ == "__main__":
    visualize_all_results()
