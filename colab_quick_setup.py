
# 🚀 QUICK SETUP UNTUK GOOGLE COLAB
# Copy-paste script ini ke sel pertama Colab Anda

# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/gan-chili-project.git
%cd gan-chili-project

# 2. Install dependencies
!pip install -r requirements_colab.txt

# 3. Setup GPU (jika belum)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU tidak aktif. Aktifkan di Runtime → Change runtime type → GPU")

# 4. Setup Git credentials (ganti dengan info Anda)
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

# 5. Test import
try:
    from colab_gan import DCGAN
    from colab_utils import setup_colab_environment
    print("✅ Import berhasil! Siap untuk training.")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("🎉 Setup selesai! Jalankan sel berikutnya untuk memulai training.")
