# 🌶️ Panduan Google Colab - GAN Data Augmentation

Panduan lengkap untuk menjalankan GAN Data Augmentation di Google Colab dengan integrasi GitHub.

## 🚀 Quick Start (5 Menit)

### 1. Buka Google Colab

- Kunjungi [Google Colab](https://colab.research.google.com/)
- Login dengan akun Google
- **Penting**: Aktifkan GPU (Runtime → Change runtime type → Hardware accelerator → GPU)

### 2. Setup Repository

**Opsi A: Clone dari GitHub (Rekomendasi)**

```python
# Di Colab, jalankan:
!git clone https://github.com/username/gan-chili-project.git
%cd gan-chili-project
```

**Opsi B: Upload Notebook Manual**

- Upload file `colab_setup.ipynb` ke Colab
- Atau copy-paste code dari file tersebut

### 3. Setup Dataset

**Opsi A: GitHub Repository (Terbaik)**

```python
# Dataset sudah ada di repository
# Otomatis ter-download saat clone repository
# Pastikan dataset ada di folder: data/train/
```

**Opsi B: Upload ZIP**

```python
# Compress dataset Anda menjadi ZIP terlebih dahulu
# Structure: dataset.zip
#   └── train/
#       ├── healthy/
#       ├── leaf curl/
#       ├── leaf spot/
#       ├── whitefly/
#       └── yellowish/
```

**Opsi C: Google Drive**

```python
# Upload dataset ke Google Drive
# Mount drive di Colab dan copy data
```

## 🔐 GitHub Integration

### 1. Setup Personal Access Token

1. Pergi ke GitHub Settings → Developer settings → Personal access tokens
2. Generate token baru dengan permissions:
   - `repo` (full control)
   - `workflow` (jika menggunakan GitHub Actions)
3. Copy token dan simpan dengan aman

### 2. Authentication di Colab

```python
# Metode 1: Personal Access Token (Rekomendasi)
import os
os.environ['GITHUB_TOKEN'] = 'your_personal_access_token_here'

# Setup git credentials
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

# Metode 2: GitHub CLI (Alternative)
!gh auth login
```

### 3. Repository Operations

```python
# Pull latest changes sebelum training
!git pull origin main

# Training your model...

# Push results setelah training
!git add .
!git commit -m "Training results - epoch 100"
!git push origin main
```

### 4. Automated Backup

```python
# Auto-backup setiap 30 menit selama training
def auto_backup():
    import time
    import subprocess

    while training_active:
        time.sleep(1800)  # 30 minutes
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', f'Auto-backup: {time.strftime("%Y-%m-%d %H:%M")}'])
        subprocess.run(['git', 'push', 'origin', 'main'])
```

## ⚙️ Penyesuaian Code

### 1. Hyperparameters yang Dioptimasi

```python
# Original (Local)
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 0.0002

# Colab Optimized
BATCH_SIZE = 64 if torch.cuda.is_available() else 32  # Larger for GPU
NUM_EPOCHS = 150  # Reduced for time limit
LEARNING_RATE = 0.0002  # Same
```

### 2. Memory Management

```python
# Clear CUDA cache between training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Generate images in batches
batch_size = 32  # Instead of generating all at once
```

### 3. Progress Monitoring

```python
# Use tqdm for better progress visualization
from tqdm import tqdm
epoch_pbar = tqdm(range(epochs), desc=f"Training {class_name}")
```

## 🎯 Mode Training

### Mode 1: Demo Training (15-20 menit)

```python
# Training 1 kelas dengan 50 epochs
demo_class = "healthy"
demo_epochs = 50
target_images = 150

# Expected output: 30 generated images
```

### Mode 2: Full Training (3-4 jam)

```python
# Training semua kelas dengan 150 epochs
classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
epochs = 150
target_images = 200

# Expected output: 600+ generated images
```

## 📊 Estimasi Waktu & Resource

### GPU Requirements

| GPU Type | Demo (1 class) | Full (5 classes) | Memory Usage |
| -------- | -------------- | ---------------- | ------------ |
| T4       | 15-20 min      | 3-4 hours        | ~4-6 GB      |
| V100     | 8-12 min       | 2-3 hours        | ~6-8 GB      |
| A100     | 5-8 min        | 1.5-2 hours      | ~8-10 GB     |

### Storage Requirements

- Dataset Original: ~50 MB
- Generated Images: ~200-300 MB
- Models: ~50 MB
- Total: ~400 MB

## 🔧 Troubleshooting

### 1. "Runtime disconnected"

```python
# Solution: Enable Colab Pro untuk session yang lebih stabil
# Atau save checkpoint secara berkala
torch.save(netG.state_dict(), f"checkpoint_G_{epoch}.pth")
```

### 2. "Out of Memory"

```python
# Reduce batch size
BATCH_SIZE = 32  # or 16

# Clear cache more frequently
torch.cuda.empty_cache()

# Generate images in smaller batches
```

### 3. "Dataset not found"

```python
# Check dataset structure
!ls -la
!ls -la "Dataset Original/train"

# Common paths yang dicoba:
paths = [
    "Dataset Original/train",
    "data/train",
    "train",
    "/content/Dataset Original/train"
]
```

### 4. "Training too slow"

```python
# Pastikan GPU aktif
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Gunakan batch size yang lebih besar untuk GPU
BATCH_SIZE = 64
```

### 5. GitHub Authentication Error

```python
# Error: "Permission denied (publickey)"
# Solution: Use Personal Access Token instead of SSH

# Setup token authentication
!git remote set-url origin https://YOUR_TOKEN@github.com/username/repo.git

# Error: "Support for password authentication was removed"
# Solution: Use token as password
# Username: your_github_username
# Password: your_personal_access_token
```

### 6. Large Files in Git

```python
# Error: "file is larger than 100 MB"
# Solution: Use Git LFS or add to .gitignore

# Add to .gitignore:
# *.pth
# *.pkl
# colab_models/
# large_datasets/

# Or use Git LFS:
!git lfs track "*.pth"
!git add .gitattributes
```

### 7. Merge Conflicts

```python
# Pull latest changes before pushing
!git pull origin main

# If conflicts occur:
!git status
!git add .
!git commit -m "Resolve merge conflicts"
!git push origin main
```

## 💡 Pro Tips untuk GitHub Workflow

### 1. Efficient Repository Structure

```
gan-chili-project/
├── .gitignore              # Ignore large files
├── colab_setup.ipynb       # Main Colab notebook
├── colab_gan.py           # GAN implementation
├── colab_utils.py         # Utility functions
├── requirements_colab.txt  # Dependencies
├── data/                  # Small sample dataset
│   └── train/
├── models/               # Trained models (ignored)
├── results/              # Training outputs (ignored)
└── README.md            # Project documentation
```

### 2. Smart .gitignore

```gitignore
# Large files
*.pth
*.pkl
*.h5
*.weights

# Datasets (too large for git)
data/full_dataset/
dataset.zip

# Outputs
colab_models/
colab_augmented/
__pycache__/
.ipynb_checkpoints/

# Temporary files
*.log
*.tmp
```

### 3. Automated Workflow

```python
# Setup function untuk workflow otomatis
def setup_github_workflow():
    """Setup complete GitHub workflow"""

    # 1. Clone repository
    !git clone https://github.com/username/gan-project.git
    %cd gan-project

    # 2. Install dependencies
    !pip install -r requirements_colab.txt

    # 3. Setup git credentials
    !git config --global user.name "Your Name"
    !git config --global user.email "your.email@example.com"

    # 4. Pull latest changes
    !git pull origin main

    print("✅ GitHub workflow setup complete!")

# Auto-backup function
def save_and_push_results():
    """Save and push training results to GitHub"""

    # Add only important files
    !git add colab_setup.ipynb
    !git add "results/*.png"
    !git add "logs/*.txt"

    # Commit with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    !git commit -m f"Training results - {timestamp}"

    # Push to GitHub
    !git push origin main

    print(f"✅ Results saved to GitHub - {timestamp}")
```

### 4. Data Management Strategy

```python
# Option 1: Small sample dataset in repo
# Keep only 10-20 images per class for testing
data/
└── sample_train/
    ├── healthy/ (10 images)
    ├── leaf_curl/ (10 images)
    └── ...

# Option 2: External dataset storage
# Store full dataset in Google Drive or Kaggle
# Use download script in notebook:
def download_full_dataset():
    # Download from Google Drive
    !gdown --id "your_drive_file_id" --output dataset.zip
    !unzip dataset.zip
```

## 📁 Structure Output

Setelah training, akan terbuat folder:

```
colab_models/          # Model GAN terlatih
├── healthy/
│   ├── generator.pth
│   └── discriminator.pth
├── leaf curl/
├── leaf spot/
├── whitefly/
└── yellowish/

colab_augmented/       # Gambar hasil augmentasi
├── healthy/           (150-200 images)
├── leaf curl/         (150-200 images)
├── leaf spot/         (150-200 images)
├── whitefly/          (150-200 images)
└── yellowish/         (150-200 images)

colab_samples/         # Sample progress training
├── healthy/
│   ├── epoch_0.png
│   ├── epoch_25.png
│   └── ...
└── ...
```

## 💾 Download Results

### Auto Download

```python
# Otomatis create ZIP dan download
download_colab_results()
```

### Manual Download

```python
# Download file tertentu
from google.colab import files
files.download("colab_models/healthy/generator.pth")
```

## 🎯 Tips Optimisasi

### 1. Untuk Hasil Terbaik

- Gunakan GPU T4 atau lebih tinggi
- Training minimal 100 epochs per kelas
- Monitor loss convergence
- Check sample quality secara berkala

### 2. Untuk Training Cepat

- Gunakan demo mode terlebih dahulu
- Reduce epochs to 50-100
- Generate lebih sedikit gambar per kelas

### 3. Untuk Memory Efficiency

- Reduce batch size jika OOM
- Clear cache between classes
- Generate images in batches

## 🚨 Perbedaan dari Versi Local

### 1. Path Handling

```python
# Local
data_dir = r"c:\Riset Infromatika\Python V3\Dataset Original\train\healthy"

# Colab
data_dir = "/content/Dataset Original/train/healthy"
```

### 2. Device Selection

```python
# Local - bisa CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colab - prioritas GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tapi lebih sering dapat GPU
```

### 3. File Operations

```python
# Local - langsung save
torch.save(model, "model.pth")

# Colab - save + download
torch.save(model, "model.pth")
files.download("model.pth")
```

## 📋 Checklist Sebelum Training

- [ ] GPU aktif di Colab
- [ ] Dataset ter-upload dengan struktur benar
- [ ] Dependencies ter-install
- [ ] Cukup storage space (~1GB)
- [ ] Session time limit (12 jam untuk free)

## 🎓 Untuk Thesis/Research

### Demo First

1. Jalankan demo training (20 menit)
2. Verifikasi hasil sample bagus
3. Lanjut ke full training

### Documentation

1. Screenshot training progress
2. Save loss plots
3. Export comparison grids
4. Download semua hasil

### Quality Check

1. Visual inspection generated images
2. Compare dengan original
3. Check diversity of generated samples
4. Verify no mode collapse

## ⚡ Command Cheat Sheet

```python
# Setup
dataset_path, classes = main_colab()

# Demo training
generator, losses_g, losses_d = demo_training_colab(dataset_path, classes)

# Full training
results = train_all_classes_colab(dataset_path, classes)

# Download
download_colab_results()
```

## 📞 Support

Jika ada masalah:

1. Check GPU availability: `torch.cuda.is_available()`
2. Check memory usage: `torch.cuda.memory_summary()`
3. Restart runtime jika perlu
4. Refer ke troubleshooting section
