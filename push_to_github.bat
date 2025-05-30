@echo off
REM GitHub Push Script untuk GAN Chili Project
REM Jalankan script ini setelah membuat repository GitHub

echo ==============================================
echo 🌶️ GAN Chili Project - GitHub Push Script
echo ==============================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git tidak terinstall! Please install Git first.
    echo 📥 Download: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo ✅ Git detected!
echo.

REM Get user input
set /p username="📝 Masukkan GitHub username Anda: "
if "%username%"=="" (
    echo ❌ Username tidak boleh kosong!
    pause
    exit /b 1
)

set /p reponame="📝 Masukkan nama repository (default: gan-chili-project): "
if "%reponame%"=="" set reponame=gan-chili-project

echo.
echo 📋 Repository Info:
echo    Username: %username%
echo    Repository: %reponame%
echo    URL: https://github.com/%username%/%reponame%.git
echo.

set /p confirm="❓ Apakah informasi sudah benar? (y/n): "
if /i not "%confirm%"=="y" (
    echo ❌ Setup dibatalkan.
    pause
    exit /b 1
)

echo.
echo 🚀 Starting GitHub setup...
echo.

REM Setup remote origin
echo 🔗 Adding remote origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/%username%/%reponame%.git
if errorlevel 1 (
    echo ❌ Failed to add remote origin!
    pause
    exit /b 1
)
echo ✅ Remote origin added successfully!

REM Set main branch
echo 🌳 Setting main branch...
git branch -M main
if errorlevel 1 (
    echo ❌ Failed to set main branch!
    pause
    exit /b 1
)
echo ✅ Main branch set successfully!

REM Push to GitHub
echo.
echo 📤 Pushing to GitHub...
echo 💡 Anda akan diminta username dan password:
echo    - Username: %username%
echo    - Password: Your Personal Access Token (NOT your GitHub password)
echo.
echo 🔐 Cara mendapatkan Personal Access Token:
echo    1. Buka: https://github.com/settings/tokens
echo    2. Generate new token (classic)
echo    3. Select 'repo' permissions
echo    4. Copy token sebagai password
echo.

git push -u origin main
if errorlevel 1 (
    echo.
    echo ❌ Push failed! Common solutions:
    echo 💡 1. Pastikan repository sudah dibuat di GitHub
    echo 💡 2. Gunakan Personal Access Token sebagai password
    echo 💡 3. Check username spelling
    echo.
    pause
    exit /b 1
)

echo.
echo 🎉 SUCCESS! Repository berhasil di-push ke GitHub!
echo.
echo 📋 Next steps:
echo 1. 🔗 Buka: https://github.com/%username%/%reponame%
echo 2. 📓 Test Colab: https://colab.research.google.com/github/%username%/%reponame%/blob/main/colab_setup.ipynb
echo 3. 📖 Baca: GITHUB_SETUP_COMPLETE.md
echo.
echo 🎯 Quick Colab setup:
echo    !git clone https://github.com/%username%/%reponame%.git
echo    %%cd %reponame%
echo.

REM Create quick reference file
echo # Quick GitHub Commands > github_commands.txt
echo. >> github_commands.txt
echo # Colab Setup >> github_commands.txt
echo !git clone https://github.com/%username%/%reponame%.git >> github_commands.txt
echo %%cd %reponame% >> github_commands.txt
echo. >> github_commands.txt
echo # Local Development >> github_commands.txt
echo git add . >> github_commands.txt
echo git commit -m "Update training results" >> github_commands.txt
echo git push origin main >> github_commands.txt
echo. >> github_commands.txt
echo # Repository URL: https://github.com/%username%/%reponame% >> github_commands.txt
echo # Colab Direct: https://colab.research.google.com/github/%username%/%reponame%/blob/main/colab_setup.ipynb >> github_commands.txt

echo 📄 Reference commands saved to: github_commands.txt
echo.
echo Happy coding! 🚀
pause
