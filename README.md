# 🌶️ GAN Data Augmentation for Chili Plant Disease Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/gan-chili-project/blob/main/colab_setup.ipynb)

Implementasi Generative Adversarial Networks (GAN) untuk augmentasi data penyakit tanaman cabai menggunakan PyTorch. Project ini dirancang khusus untuk Google Colab dengan integrasi GitHub workflow yang seamless.

## 🎯 Features

- **🚀 Google Colab Ready**: Setup 1-click untuk training di Colab
- **🔄 GitHub Integration**: Automatic backup dan version control
- **📊 Multi-Class GAN**: Support 5 kelas penyakit cabai
- **💾 Auto-Save**: Periodic backup selama training
- **📈 Progress Monitoring**: Real-time training visualization
- **🎨 Smart Augmentation**: High-quality synthetic image generation

## 🌿 Dataset Classes

Dataset terdiri dari 5 kelas penyakit tanaman cabai:

1. **Healthy** - Daun sehat (80 → 200 gambar)
2. **Leaf Curl** - Keriting daun (80 → 200 gambar)
3. **Leaf Spot** - Bercak daun (80 → 200 gambar)
4. **Whitefly** - Serangan kutu kebul (80 → 200 gambar)
5. **Yellowish** - Menguning (80 → 200 gambar)

## ⚡ Quick Start

### Option 1: Google Colab (Recommended)

1. **Klik badge Colab di atas** atau [buka langsung](https://colab.research.google.com/github/YOUR_USERNAME/gan-chili-project/blob/main/colab_setup.ipynb)
2. **Aktifkan GPU**: Runtime → Change runtime type → Hardware accelerator → GPU
3. **Jalankan semua cell** - setup otomatis!

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gan-chili-project.git
cd gan-chili-project

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

## 🛠️ Project Structure

```
gan-chili-project/
├── 📓 colab_setup.ipynb      # Main Colab notebook
├── 🧠 colab_gan.py          # GAN implementation
├── 🔧 colab_utils.py        # Utility functions
├── 📋 requirements_colab.txt # Colab dependencies
├── 📖 COLAB_GUIDE.md        # Detailed guide
├── 🚀 setup_github.py       # GitHub setup script
├── 📊 main.py               # Local training script
├── 📁 data/                 # Sample dataset
│   └── train/
│       ├── healthy/
│       ├── leaf_curl/
│       ├── leaf_spot/
│       ├── whitefly/
│       └── yellowish/
└── 📄 README.md
```

│ ├── leaf spot/ (80 gambar)
│ ├── whitefly/ (80 gambar)
│ └── yellowish/ (80 gambar)
└── ...

```

### Struktur Dataset Setelah Augmentasi

```

Dataset Augmented/
└── train/
├── healthy/ (200 gambar)
├── leaf curl/ (200 gambar)
├── leaf spot/ (200 gambar)
├── whitefly/ (200 gambar)
└── yellowish/ (200 gambar)

````

## 🛠️ Requirements

### Hardware

- **GPU**: NVIDIA GPU dengan CUDA support (disarankan)
- **RAM**: Minimal 8GB, disarankan 16GB+
- **Storage**: ~5GB untuk model dan hasil augmentasi

### Software

- Python 3.7+
- CUDA 11.0+ (jika menggunakan GPU)

### Dependencies

```bash
pip install -r requirements.txt
````

Dependencies yang dibutuhkan:

- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `Pillow>=8.0.0`
- `numpy>=1.21.0`
- `matplotlib>=3.3.0`
- `tqdm>=4.60.0`
- `seaborn>=0.11.0`
- `opencv-python>=4.5.0`

### 📊 Karakteristik Dataset Asli

Berdasarkan analisis dataset:

| Kelas     | Jumlah | Avg Size | Aspect Ratio | File Size | Avg Color (RGB) |
| --------- | ------ | -------- | ------------ | --------- | --------------- |
| Healthy   | 80     | 265x195  | 1.42         | 13.3 KB   | (116, 122, 79)  |
| Leaf Curl | 80     | 230x183  | 1.29         | 12.0 KB   | (114, 131, 78)  |
| Leaf Spot | 80     | 244x195  | 1.29         | 8.6 KB    | (113, 135, 87)  |
| Whitefly  | 80     | 237x184  | 1.32         | 8.0 KB    | (112, 131, 90)  |
| Yellowish | 80     | 218x179  | 1.25         | 9.9 KB    | (120, 132, 70)  |

**Total: 400 gambar, rata-rata ukuran: 209px**

## 🚀 Quick Start Guide

### 📋 Persiapan (5 menit)

1. **Clone/Download Project**

   ```bash
   # Pastikan semua file ada di direktori kerja
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Test Sistem**
   ```bash
   python test_system.py
   ```
   **Expected Output:** `5/5 tests passed`

### 🎯 Pilihan Eksekusi

#### A. 🚀 **DEMO MODE** (Rekomendasi Pertama)

```bash
python cpu_training.py demo
```

**⏱️ Waktu:** 5 menit  
**📊 Output:** 50 sample gambar  
**🎯 Tujuan:** Testing dan preview sistem

**Expected Output:**

```
=== QUICK DEMO - CPU GAN Training ===
Training 1 kelas dengan 20 epoch untuk demo cepat...
=== CPU-Optimized GAN Training untuk healthy ===
Epoch [ 19/20] Loss_D: 0.626 Loss_G: 7.697 Time: 1.8s ETA: 0.0m
Training completed in 0.4 minutes
Generated 10 images saved to quick_demo_output
Demo selesai! Cek folder 'quick_demo_output' untuk hasil.
```

✅ **Demo berhasil jika:**

- Folder `quick_demo_output/` terbuat
- Berisi 10 file `.jpg` (generated_healthy_xxx.jpg)
- Training selesai dalam < 5 menit
- No error messages

#### B. 💼 **PRODUCTION MODE** (Untuk Thesis)

```bash
python main.py
```

**⏱️ Waktu:** 12-15 jam  
**📊 Output:** 600 gambar (120 per kelas)  
**🎯 Tujuan:** Dataset lengkap untuk research

#### C. 🎨 **CUSTOM MODE** (Kontrol Manual)

```bash
# Training satu kelas
python main.py --mode train --class "healthy" --epochs 300

# Target khusus
python main.py --target 250 --epochs 500
```

### 📈 Monitoring & Evaluasi

```bash
# Monitor progress (terminal terpisah)
python monitor.py

# Analisis dataset
python analyze_dataset.py

# Evaluasi kualitas
python evaluate_gan_quality.py

# Visualisasi hasil
python visualize_results.py
```

## 🚀 Cara Penggunaan

### 🔍 Step 1: Test Sistem

Sebelum memulai training, pastikan sistem berfungsi dengan baik:

```bash
python test_system.py
```

### 📊 Step 2: Analisis Dataset

Analisis karakteristik dataset sebelum augmentasi:

```bash
python analyze_dataset.py
```

### 🚀 Step 3: Pilihan Training

#### A. Quick Demo (Rekomendasi untuk Testing)

```bash
python cpu_training.py demo
```

- Training cepat (10 epochs, ~5 menit)
- Untuk testing dan melihat cara kerja sistem
- Menghasilkan 50 sample gambar

#### B. Production Training (Untuk Thesis)

```bash
python main.py
```

- Training lengkap semua kelas (300 epochs each)
- Estimasi waktu: 12-15 jam
- Menghasilkan 120 gambar per kelas (600 total)

#### C. Training CPU-Optimized

```bash
python cpu_training.py
```

- Khusus untuk komputer tanpa GPU
- Parameter sudah dioptimasi untuk CPU
- Batch size kecil untuk efisiensi memori

### 🎯 Step 4: Mode Training Khusus

#### Training untuk kelas tertentu:

```bash
python main.py --mode train --class "leaf curl" --epochs 300
```

#### Training dengan target khusus:

```bash
python main.py --mode train --target 250 --epochs 500
```

#### Monitoring progress training:

```bash
python monitor.py
```

### 📈 Step 5: Evaluasi & Visualisasi

#### Evaluasi kualitas hasil:

```bash
python evaluate_gan_quality.py
```

#### Visualisasi hasil:

```bash
python visualize_results.py
```

### 3. Script Individual

#### Menjalankan augmentasi:

```bash
# Semua kelas
python run_augmentation.py

# Kelas tertentu
python run_augmentation.py --class healthy --epochs 300 --target 200
```

#### Evaluasi kualitas:

```bash
python evaluate_gan_quality.py
```

#### Visualisasi:

```bash
python visualize_results.py
```

## 📊 Evaluasi Kualitas

Program menyediakan beberapa metrik evaluasi:

### 1. FID (Fréchet Inception Distance)

- **Range**: 0-∞ (lower is better)
- **Good**: < 50
- **Acceptable**: 50-150

### 2. Inception Score (IS)

- **Range**: 1-∞ (higher is better)
- **Good**: > 2.0
- **Acceptable**: 1.5-2.0

### 3. Visual Quality Assessment

- Perbandingan distribusi warna
- Grid perbandingan gambar asli vs generated

## 🏗️ Arsitektur GAN

### Generator

```
Input: Noise Vector (100D)
├── ConvTranspose2d(100, 512, 4x4) → 4x4
├── ConvTranspose2d(512, 256, 4x4) → 8x8
├── ConvTranspose2d(256, 128, 4x4) → 16x16
├── ConvTranspose2d(128, 64, 4x4)  → 32x32
└── ConvTranspose2d(64, 3, 4x4)    → 64x64
Output: RGB Image (64x64x3)
```

### Discriminator

```
Input: RGB Image (64x64x3)
├── Conv2d(3, 64, 4x4)    → 32x32
├── Conv2d(64, 128, 4x4)  → 16x16
├── Conv2d(128, 256, 4x4) → 8x8
├── Conv2d(256, 512, 4x4) → 4x4
└── Conv2d(512, 1, 4x4)   → 1x1
Output: Real/Fake probability
```

## 📁 Struktur Output

Setelah menjalankan program, akan terbuat folder-folder berikut:

```
📁 Dataset Augmented/          # Dataset hasil augmentasi
📁 gan_models/                 # Model GAN yang sudah ditraining
📁 gan_samples/               # Sample gambar selama training
📁 visualization_results/      # Hasil visualisasi dan analisis
📄 gan_losses_*.png           # Grafik loss training per kelas
```

### Detail Output:

**Dataset Augmented/**

- Berisi dataset lengkap (asli + generated)
- Siap digunakan untuk training klasifikasi

**gan_models/**

- `generator.pth`: Model generator terlatih
- `discriminator.pth`: Model discriminator terlatih

**visualization_results/**

- `dataset_overview.png`: Overview statistik dataset
- `comparison_*.png`: Grid perbandingan per kelas
- `training_progress.png`: Status training

## ⚙️ Konfigurasi

### Hyperparameters (dapat diubah di `gan_data_augmentation.py`):

```python
IMG_SIZE = 64          # Ukuran gambar output
BATCH_SIZE = 32        # Batch size training
NUM_EPOCHS = 300       # Jumlah epoch training
LEARNING_RATE = 0.0002 # Learning rate
BETA1 = 0.5           # Adam optimizer beta1
NZ = 100              # Dimensi noise vector
```

### Modifikasi Target:

Untuk mengubah target jumlah gambar per kelas:

```bash
python main.py --target 300  # Target 300 gambar per kelas
```

## 🔧 Troubleshooting

### 1. Test Sistem Gagal

```bash
python troubleshoot.py
```

Script ini akan membantu mendiagnosis dan memperbaiki masalah umum.

### 2. CUDA Out of Memory

```python
# Gunakan CPU training
python cpu_training.py

# Atau kurangi batch size di gan_data_augmentation.py
BATCH_SIZE = 16  # atau 8
```

### 3. Training Terlalu Lama

```python
# Gunakan demo mode terlebih dahulu
python cpu_training.py demo

# Atau kurangi jumlah epoch
NUM_EPOCHS = 100
```

### 4. Error Path atau Dataset

```bash
# Periksa struktur dataset
python analyze_dataset.py

# Pastikan path dataset benar di main.py
```

### 5. Kualitas Gambar Buruk

- Tambah epoch training (300+ epochs)
- Periksa FID score dengan `python evaluate_gan_quality.py`
- Gunakan parameter CPU-optimized jika perlu

### 6. Mode Collapse

- Restart training dengan seed berbeda
- Sesuaikan learning rate
- Gunakan parameter yang sudah dioptimasi

## 📈 Tips Optimisasi

### 1. Meningkatkan Kualitas:

- Train lebih lama (500+ epochs)
- Gunakan progressive growing
- Implementasikan self-attention
- Tambahkan regularization techniques

### 2. Mempercepat Training:

- Gunakan mixed precision training
- Increase batch size (jika memori cukup)
- Gunakan multiple GPUs

### 3. Dataset Specific:

- Augmentasi data asli sebelum training GAN
- Fine-tune pre-trained generator
- Gunakan transfer learning

## 🎯 Hasil yang Diharapkan

Setelah augmentasi berhasil:

- **Total dataset**: 1000 gambar (200 per kelas)
- **Peningkatan**: 150% dari dataset asli
- **FID Score**: < 100 (target kualitas)
- **Visual Quality**: Mirip dengan data asli

### Timeline Estimasi:

| Mode              | Waktu     | Output     | Rekomendasi    | Command                          |
| ----------------- | --------- | ---------- | -------------- | -------------------------------- |
| **Demo**          | 5 menit   | 50 samples | Testing sistem | `python cpu_training.py demo`    |
| **Single Class**  | 2-3 jam   | 120 gambar | Per kelas      | `python main.py --class healthy` |
| **Full Training** | 12-15 jam | 600 gambar | Production     | `python main.py`                 |
| **CPU Training**  | 8-10 jam  | 600 gambar | Tanpa GPU      | `python cpu_training.py`         |

### 📊 Metrik Kualitas Target:

| Metrik              | Target | Good  | Acceptable |
| ------------------- | ------ | ----- | ---------- |
| **FID Score**       | < 50   | < 100 | < 150      |
| **Inception Score** | > 2.0  | > 1.8 | > 1.5      |
| **Visual Quality**  | 90%+   | 80%+  | 70%+       |

### 🔬 Interpretasi Hasil untuk Thesis

#### 1. **FID Score (Fréchet Inception Distance)**

- Mengukur kemiripan distribusi gambar generated vs real
- Semakin rendah = semakin baik
- **< 50**: Excellent quality untuk thesis
- **50-100**: Good quality, acceptable untuk research
- **> 150**: Perlu improvement

#### 2. **Inception Score (IS)**

- Mengukur diversity dan quality gambar generated
- Semakin tinggi = semakin baik
- **> 2.0**: Excellent untuk plant disease classification
- **1.5-2.0**: Good, masih layak untuk thesis
- **< 1.5**: Perlu retrain dengan parameter berbeda

#### 3. **Visual Quality Assessment**

- Grid comparison antara real vs generated
- Color distribution analysis
- **90%+**: Hampir tidak bisa dibedakan
- **80%+**: Minor differences, still good
- **< 70%**: Obvious artificial artifacts

### 📝 Dokumentasi untuk Thesis

Setelah training selesai, gunakan hasil berikut untuk dokumentasi thesis:

1. **Grafik Training Loss** (`gan_losses_*.png`)
2. **Dataset Analysis** (`dataset_analysis.png`)
3. **Color Distribution** (`color_distribution_analysis.png`)
4. **Comparison Grids** (`comparison_*.png`)
5. **Evaluation Report** (`evaluation_report.txt`)

### File Output yang Dihasilkan:

```
📁 Dataset Augmented/
├── train/
│   ├── healthy/        (200 gambar)
│   ├── leaf curl/      (200 gambar)
│   ├── leaf spot/      (200 gambar)
│   ├── whitefly/       (200 gambar)
│   └── yellowish/      (200 gambar)
📁 gan_models/
├── generator_healthy.pth
├── generator_leaf_curl.pth
├── ... (model per kelas)
📁 gan_samples/
├── training_samples_healthy/
├── training_samples_leaf_curl/
├── ... (progress samples)
📁 visualization_results/
├── dataset_analysis.png
├── color_distribution_analysis.png
├── comparison_grid_*.png
└── evaluation_report.txt
```

## 📚 Referensi

1. Goodfellow, I., et al. "Generative Adversarial Nets." NIPS 2014.
2. Radford, A., et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR 2016.
3. Heusel, M., et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NIPS 2017.

## 👥 Kontribusi

Program ini dibuat untuk tugas penelitian "Pengembangan Aplikasi Berbasis Visual Transformer dan Generative Adversarial Networks untuk Klasifikasi Penyakit Tanaman Cabai".

## ❓ FAQ (Frequently Asked Questions)

### Q: Apa yang harus dilakukan pertama kali?

**A:** Jalankan `python test_system.py` untuk memastikan semua komponen berfungsi. Lalu coba `python cpu_training.py demo` untuk melihat cara kerja sistem.

### Q: Komputer saya tidak punya GPU, apakah bisa?

**A:** Ya! Gunakan `python cpu_training.py` yang sudah dioptimasi untuk CPU. Demo berhasil berjalan dengan CPU dalam 5 menit.

### Q: Berapa lama training untuk thesis?

**A:** Untuk hasil production, estimasi 12-15 jam dengan `python main.py`. Untuk CPU: 8-10 jam dengan `python cpu_training.py`.

### Q: Bagaimana cara melihat progress training?

**A:** Gunakan `python monitor.py` di terminal terpisah saat training berlangsung.

### Q: Hasil GAN buruk, apa yang harus dilakukan?

**A:**

1. Cek FID score dengan `python evaluate_gan_quality.py`
2. Tambah epoch training jika < 300
3. Gunakan demo mode dulu untuk testing
4. Lihat tutorial troubleshooting di bawah

### Q: Apakah bisa training hanya satu kelas?

**A:** Ya! Gunakan `python main.py --mode train --class "healthy" --epochs 300`

### Q: File apa yang penting untuk thesis?

**A:**

- `Dataset Augmented/` → dataset final (1000 gambar)
- `visualization_results/` → analisis dan grafik
- `evaluation_report.txt` → metrik kualitas
- `*_losses_*.png` → grafik training progress

### Q: Error "CUDA not available", apakah masalah?

**A:** Tidak masalah! Sistem otomatis switch ke CPU training yang sudah dioptimasi.

### Q: Demo berhasil, langkah selanjutnya?

**A:** Jika demo sukses, lanjut ke production dengan `python main.py` untuk dataset lengkap.

## 📞 Support

Jika mengalami masalah:

1. Cek error logs di terminal
2. Pastikan semua requirements terinstall
3. Verifikasi dataset structure
4. Cek ketersediaan memory/storage
