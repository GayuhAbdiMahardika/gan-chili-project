# 🚀 Google Colab Quickstart Guide

## GAN Chili Project - Panduan Cepat

### 📋 Langkah-langkah menjalankan di Google Colab:

## 🎯 **SOLUSI UNTUK ERROR YANG ANDA ALAMI:**

### ✅ **Step-by-step fix:**

1. **Buka Colab dengan link baru:** https://colab.research.google.com/github/GayuhAbdiMahardika/gan-chili-project/blob/main/colab_setup.ipynb

2. **Jalankan CELL PERTAMA** - Setup GitHub:
   ```python
   # 🚀 STEP 1: Clone GitHub Repository
   !git clone https://github.com/GayuhAbdiMahardika/gan-chili-project.git
   %cd gan-chili-project
   ```

3. **Jalankan CELL KEDUA** - Check Dataset:
   ```python
   # 📁 STEP 2: Setup Dataset
   # Cell ini akan mengecek apakah dataset ada dan memberi instruksi
   ```

4. **Upload Dataset** - Pilih salah satu:
   - **Option A:** Google Drive (recommended)
   - **Option B:** ZIP file upload

---

## 📦 **Cara Compress Dataset untuk Upload:**

### **Metode 1: Menggunakan Script Python**
```bash
# Jalankan di terminal/command prompt
cd "c:\Riset Infromatika\Python V3\GAN_Project"
python compress_dataset.py
```

### **Metode 2: Manual Compress**
1. Right-click folder `Dataset Original`
2. Send to → Compressed (zipped) folder
3. Rename menjadi `Dataset_Original.zip`
4. Upload ke Colab atau Google Drive

---

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

**📁 STRUKTUR FOLDER DI GOOGLE DRIVE:**

```
My Drive/
└── Dataset Original/
    ├── train/
    │   ├── healthy/
    │   ├── leaf curl/
    │   ├── leaf spot/
    │   ├── whitefly/
    │   └── yellowish/
    ├── test/ (sama seperti train)
    └── val/ (sama seperti train)
```

**🔗 Cara Upload ke Google Drive:**

1. Buka https://drive.google.com
2. Upload folder `Dataset Original` ke My Drive
3. Atau zip folder dan upload, lalu extract di Colab

**💻 Code untuk Copy dari Google Drive:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset dari Drive ke Colab
!cp -r "/content/drive/MyDrive/Dataset Original" ./

# Verifikasi struktur folder
!ls -la "Dataset Original/"
!ls -la "Dataset Original/train/"

print("✅ Dataset berhasil di-copy dari Google Drive!")
```

**🗜️ Jika upload dalam bentuk ZIP:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract ZIP file dari Drive
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/Dataset_Original.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

print("✅ Dataset berhasil di-extract dari ZIP!")
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

### ❌ Error: "Not a git repository"
**Penyebab:** Notebook tidak dijalankan dari repository yang di-clone
**Solusi:**
```python
# Pastikan Anda menjalankan cell setup pertama dengan benar:
!git clone https://github.com/GayuhAbdiMahardika/gan-chili-project.git
%cd gan-chili-project

# Verifikasi lokasi
!pwd
!ls -la
```

### ❌ Error: "Dataset not found!"
**Penyebab:** Dataset belum di-upload atau path salah
**Solusi - Pilih salah satu:**

**Option A: Google Drive**
```python
# 1. Upload folder "Dataset Original" ke Google Drive
# 2. Jalankan cell Google Drive upload di notebook
# 3. Verifikasi struktur:
!ls -la "Dataset Original/"
!ls -la "Dataset Original/train/"
```

**Option B: ZIP Upload**
```python
# 1. Compress folder "Dataset Original" menjadi ZIP
# 2. Jalankan cell ZIP upload di notebook
# 3. Upload file ZIP saat diminta
```

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
