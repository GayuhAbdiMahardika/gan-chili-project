#!/usr/bin/env python3
"""
GitHub Repository Setup Script for GAN Chili Project
Membantu setup repository GitHub dengan mudah
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} berhasil")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        return True
    except:
        print("❌ Git tidak terinstall. Install Git terlebih dahulu.")
        return False

def setup_local_repository():
    """Initialize local git repository"""
    
    # Check if already a git repo
    if os.path.exists('.git'):
        print("📁 Repository sudah ada")
        return True
        
    # Initialize git
    if not run_command('git init', 'Inisialisasi Git repository'):
        return False
        
    # Add all files
    if not run_command('git add .', 'Menambahkan semua file'):
        return False
        
    # Initial commit
    if not run_command('git commit -m "Initial commit - GAN Chili Project"', 
                      'Commit pertama'):
        return False
        
    print("✅ Local repository berhasil dibuat")
    return True

def setup_github_credentials():
    """Setup GitHub credentials"""
    print("\n📝 Setup GitHub Credentials")
    
    # Get user input
    name = input("Masukkan nama GitHub Anda: ")
    email = input("Masukkan email GitHub Anda: ")
    
    # Set git config
    run_command(f'git config --global user.name "{name}"', 
               'Setup nama pengguna')
    run_command(f'git config --global user.email "{email}"', 
               'Setup email pengguna')
    
    print("✅ GitHub credentials berhasil diset")

def connect_to_github():
    """Connect local repo to GitHub"""
    print("\n🔗 Menghubungkan ke GitHub Repository")
    
    print("Silakan buat repository baru di GitHub dengan nama:")
    print("📌 gan-chili-project")
    print("🌐 Kunjungi: https://github.com/new")
    
    repo_url = input("\nMasukkan URL repository GitHub Anda (format: https://github.com/username/repo.git): ")
    
    if not repo_url:
        print("❌ URL repository tidak boleh kosong")
        return False
        
    # Add remote origin
    if not run_command(f'git remote add origin {repo_url}', 
                      'Menambahkan remote origin'):
        return False
        
    # Push to GitHub
    print("\n🚀 Pushing ke GitHub...")
    print("Anda mungkin diminta memasukkan username dan Personal Access Token")
    
    if not run_command('git push -u origin main', 'Push ke GitHub'):
        print("\n💡 Tips:")
        print("1. Pastikan repository GitHub sudah dibuat")
        print("2. Gunakan Personal Access Token sebagai password")
        print("3. Buat token di: https://github.com/settings/tokens")
        return False
        
    return True

def create_colab_quick_setup():
    """Create quick setup script for Colab"""
    
    colab_script = '''
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
'''
    
    with open('colab_quick_setup.py', 'w', encoding='utf-8') as f:
        f.write(colab_script)
        
    print("📄 File colab_quick_setup.py berhasil dibuat")

def main():
    """Main setup function"""
    print("🌶️ GitHub Repository Setup - GAN Chili Project")
    print("="*50)
    
    # Check prerequisites
    if not check_git_installed():
        return
        
    # Setup local repository
    print("\n1️⃣ Setup Local Repository")
    if not setup_local_repository():
        return
        
    # Setup credentials
    print("\n2️⃣ Setup GitHub Credentials")
    setup_github_credentials()
    
    # Connect to GitHub
    print("\n3️⃣ Connect to GitHub")
    if not connect_to_github():
        print("\n⚠️ Koneksi GitHub gagal. Anda bisa coba lagi nanti dengan:")
        print("git remote add origin <URL_REPOSITORY>")
        print("git push -u origin main")
        
    # Create Colab setup
    print("\n4️⃣ Membuat Quick Setup untuk Colab")
    create_colab_quick_setup()
    
    print("\n🎉 Setup selesai!")
    print("\n📋 Langkah selanjutnya:")
    print("1. Edit file colab_quick_setup.py (ganti YOUR_USERNAME dengan username GitHub Anda)")
    print("2. Buka Google Colab")
    print("3. Copy-paste isi colab_quick_setup.py ke sel pertama")
    print("4. Jalankan dan mulai training!")
    
    print("\n💡 Tips:")
    print("- Gunakan Personal Access Token untuk authentication")
    print("- Buat token di: https://github.com/settings/tokens")
    print("- Simpan hasil training dengan: git add . && git commit -m 'training results' && git push")

if __name__ == "__main__":
    main()
