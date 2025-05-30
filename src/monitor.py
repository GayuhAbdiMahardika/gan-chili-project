"""
Script untuk monitoring progress training GAN real-time
"""

import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def monitor_training_progress():
    """Monitor progress training secara real-time"""
    
    print("=== GAN Training Progress Monitor ===")
    print("Monitoring folder 'gan_samples' untuk progress training...")
    print("Press Ctrl+C to stop monitoring")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    last_check = {}
    
    try:
        while True:
            print(f"\n[{time.strftime('%H:%M:%S')}] Checking training progress...")
            
            for class_name in classes:
                sample_dir = f"gan_samples/{class_name}"
                if os.path.exists(sample_dir):
                    # Count sample files
                    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
                    
                    if sample_files:
                        latest_file = max(sample_files, key=lambda x: os.path.getctime(os.path.join(sample_dir, x)))
                        latest_epoch = latest_file.replace('epoch_', '').replace('.png', '')
                        
                        # Check if there's new progress
                        if class_name not in last_check or last_check[class_name] != latest_epoch:
                            print(f"  {class_name:<12}: Epoch {latest_epoch} completed")
                            last_check[class_name] = latest_epoch
                        else:
                            print(f"  {class_name:<12}: Still at epoch {latest_epoch}")
                    else:
                        print(f"  {class_name:<12}: No samples yet")
                else:
                    print(f"  {class_name:<12}: Training not started")
            
            # Check for loss files
            loss_files = [f for f in os.listdir('.') if f.startswith('gan_losses_') and f.endswith('.png')]
            if loss_files:
                print(f"\nCompleted trainings: {len(loss_files)}/5 classes")
                for loss_file in loss_files:
                    class_name = loss_file.replace('gan_losses_', '').replace('.png', '').replace('_', ' ')
                    print(f"  ✅ {class_name}")
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def create_progress_summary():
    """Buat ringkasan progress training"""
    
    print("=== Training Progress Summary ===")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    # Check samples
    print("\nSample Generation Progress:")
    for class_name in classes:
        sample_dir = f"gan_samples/{class_name}"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
            if sample_files:
                latest_epoch = max([int(f.replace('epoch_', '').replace('.png', '')) for f in sample_files])
                print(f"  {class_name:<12}: {len(sample_files)} samples, latest epoch {latest_epoch}")
            else:
                print(f"  {class_name:<12}: No samples")
        else:
            print(f"  {class_name:<12}: Not started")
    
    # Check models
    print("\nTrained Models:")
    for class_name in classes:
        model_dir = f"gan_models/{class_name}"
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/generator.pth"):
            print(f"  {class_name:<12}: ✅ Model saved")
        else:
            print(f"  {class_name:<12}: ❌ No model")
    
    # Check generated images
    print("\nGenerated Images:")
    augmented_base = "Dataset Augmented/train"
    if os.path.exists(augmented_base):
        for class_name in classes:
            class_dir = os.path.join(augmented_base, class_name)
            if os.path.exists(class_dir):
                generated_count = len([f for f in os.listdir(class_dir) 
                                     if f.lower().startswith('generated')])
                total_count = len([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name:<12}: {generated_count} generated, {total_count} total")
            else:
                print(f"  {class_name:<12}: No augmented data")
    else:
        print("  No augmented dataset found")

def view_latest_samples():
    """Tampilkan sample terbaru dari training"""
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    
    fig, axes = plt.subplots(1, len(classes), figsize=(20, 4))
    fig.suptitle('Latest Training Samples', fontsize=16)
    
    for i, class_name in enumerate(classes):
        sample_dir = f"gan_samples/{class_name}"
        
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
            
            if sample_files:
                # Get latest sample
                latest_file = max(sample_files, key=lambda x: os.path.getctime(os.path.join(sample_dir, x)))
                latest_path = os.path.join(sample_dir, latest_file)
                
                # Display image
                img = Image.open(latest_path)
                axes[i].imshow(img)
                
                # Extract epoch number
                epoch = latest_file.replace('epoch_', '').replace('.png', '')
                axes[i].set_title(f'{class_name}\nEpoch {epoch}')
            else:
                axes[i].text(0.5, 0.5, 'No samples\nyet', ha='center', va='center')
                axes[i].set_title(f'{class_name}\nNot started')
        else:
            axes[i].text(0.5, 0.5, 'Training\nnot started', ha='center', va='center')
            axes[i].set_title(f'{class_name}\nNot started')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('latest_samples_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Latest samples overview saved as 'latest_samples_overview.png'")

def estimate_completion_time():
    """Estimasi waktu penyelesaian training"""
    
    print("=== Training Time Estimation ===")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    completed = 0
    in_progress = 0
    
    for class_name in classes:
        model_path = f"gan_models/{class_name}/generator.pth"
        sample_dir = f"gan_samples/{class_name}"
        
        if os.path.exists(model_path):
            completed += 1
        elif os.path.exists(sample_dir) and os.listdir(sample_dir):
            in_progress += 1
    
    print(f"Status:")
    print(f"  Completed: {completed}/5 classes")
    print(f"  In progress: {in_progress}/5 classes")
    print(f"  Not started: {5 - completed - in_progress}/5 classes")
    
    if completed > 0:
        # Estimate based on completed classes
        # Assuming each class takes ~2-3 hours on CPU with 300 epochs
        estimated_per_class = 2.5  # hours
        remaining_classes = 5 - completed
        estimated_remaining = remaining_classes * estimated_per_class
        
        print(f"\nEstimated remaining time: {estimated_remaining:.1f} hours")
        print(f"Expected completion: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + estimated_remaining * 3600))}")
    else:
        print(f"\nEstimated total time: 12-15 hours for all classes")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            monitor_training_progress()
        elif command == "summary":
            create_progress_summary()
        elif command == "samples":
            view_latest_samples()
        elif command == "time":
            estimate_completion_time()
        else:
            print("Unknown command")
    else:
        print("GAN Training Monitor")
        print("Commands:")
        print("  python monitor.py monitor    # Real-time monitoring")
        print("  python monitor.py summary    # Progress summary")
        print("  python monitor.py samples    # View latest samples")
        print("  python monitor.py time       # Time estimation")
