# 🚀 Google Colab Quickstart Guide

## GAN Chili Project - Panduan Cepat

### 📋 Langkah-langkah menjalankan di Google Colab:

## 1. **Buka Notebook di Colab**

Klik link berikut untuk membuka langsung di Google Colab:

```
https://colab.research.google.com/github/GayuhAbdiMahardika/gan-chili-project/blob/main/colab_setup.ipynb
```

## 2. **Aktifkan GPU**

- Klik **Runtime** → **Change runtime type**
- Pilih **GPU** sebagai Hardware accelerator
- Klik **Save**
- Restart runtime jika diminta

## 3. **Jalankan Setup Otomatis**

Jalankan cell pertama di notebook yang akan:

- ✅ Clone repository dari GitHub
- ✅ Install dependencies (PyTorch, etc.)
- ✅ Setup environment
- ✅ Verifikasi GPU

## 4. **Upload Dataset**

### Opsi A: Upload Zip File

```python
# Cell untuk upload dataset
from google.colab import files
import zipfile
import os

print("📁 Upload file dataset (zip format)")
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

print("✅ Dataset berhasil di-upload!")
```

### Opsi B: Google Drive (Recommended)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset dari Drive
!cp -r "/content/drive/MyDrive/Dataset Original" ./
print("✅ Dataset berhasil di-copy dari Google Drive!")
```

## 5. **Verifikasi Setup**

```python
# Jalankan cell ini untuk verifikasi
import torch
from colab_gan import DCGAN
from colab_utils import setup_colab_environment

print(f"🔥 GPU Status: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

print("✅ Semua setup berhasil! Siap untuk training.")
```

## 6. **Mulai Training**

```python
# Training GAN untuk augmentasi data
from colab_gan import DCGAN
from colab_utils import load_chili_dataset

# Load dataset
train_loader = load_chili_dataset("Dataset Original/train", batch_size=32)

# Initialize GAN
gan = DCGAN(
    image_size=64,
    nz=100,
    ngf=64,
    ndf=64,
    num_epochs=100
)

# Start training
gan.train(train_loader)
```

## 7. **Menyimpan Hasil ke GitHub**

```python
# Setelah training selesai, simpan ke GitHub
!git add results/
!git commit -m "Training results - $(date)"
!git push

print("✅ Hasil training berhasil disimpan ke GitHub!")
```

---

## 🔧 Troubleshooting

### ❌ Jika terjadi error "No module named..."

```python
!pip install -r requirements_colab.txt
```

### ❌ Jika GPU tidak terdeteksi

- Restart runtime: **Runtime** → **Restart runtime**
- Pastikan GPU sudah dipilih di runtime settings

### ❌ Jika dataset tidak ditemukan

```python
# Check struktur folder
!ls -la
!ls -la "Dataset Original/"
```

### ❌ Jika git push gagal

```python
# Setup git credentials
!git config --global user.name "Your GitHub Username"
!git config --global user.email "your.email@gmail.com"

# Atau clone ulang dengan token
!git clone https://YOUR_TOKEN@github.com/GayuhAbdiMahardika/gan-chili-project.git
```

---

## 💡 Tips & Best Practices

1. **Simpan progress secara berkala**

   ```python
   # Auto-save setiap 50 epoch
   if epoch % 50 == 0:
       !git add .
       !git commit -m f"Checkpoint epoch {epoch}"
       !git push
   ```

2. **Monitor training**

   ```python
   # Visualisasi progress
   import matplotlib.pyplot as plt
   plt.plot(losses)
   plt.title('Training Progress')
   plt.show()
   ```

3. **Download hasil training**
   ```python
   # Download model yang sudah ditraining
   from google.colab import files
   files.download('models/generator.pth')
   files.download('results/generated_samples.png')
   ```

## 🌶️ Happy Training!

Selamat mencoba GAN untuk augmentasi data chili! 🚀
