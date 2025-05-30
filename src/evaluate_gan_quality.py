"""
Script untuk evaluasi kualitas gambar yang dihasilkan GAN
Menggunakan berbagai metrik seperti FID, IS, dan LPIPS
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageQualityEvaluator:
    """Class untuk evaluasi kualitas gambar hasil GAN"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.inception_model = self._load_inception_model()
        
    def _load_inception_model(self):
        """Load pre-trained Inception model untuk FID calculation"""
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()  # Remove final classification layer
        model.eval()
        return model.to(self.device)
    
    def _preprocess_images(self, image_folder, img_size=299):
        """Preprocess gambar untuk Inception model"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        images = []
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc="Loading images"):
            img_path = os.path.join(image_folder, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found in {image_folder}")
            
        return torch.cat(images, dim=0)
    
    def _get_inception_features(self, images):
        """Extract features menggunakan Inception model"""
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch_features = self.inception_model(batch)
                features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0).numpy()
    
    def calculate_fid(self, real_folder, fake_folder):
        """
        Calculate Frechet Inception Distance (FID)
        Lower is better (0 = identical distributions)
        """
        print("Calculating FID...")
        
        # Load and preprocess images
        real_images = self._preprocess_images(real_folder)
        fake_images = self._preprocess_images(fake_folder)
        
        # Get Inception features
        real_features = self._get_inception_features(real_images)
        fake_features = self._get_inception_features(fake_images)
        
        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return fid
    
    def calculate_inception_score(self, fake_folder, splits=10):
        """
        Calculate Inception Score (IS)
        Higher is better (natural images ~= 11.24)
        """
        print("Calculating Inception Score...")
        
        # Load images
        images = self._preprocess_images(fake_folder)
        
        # Get predictions
        with torch.no_grad():
            predictions = []
            batch_size = 32
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # Use full inception model for classification
                inception_full = inception_v3(pretrained=True)
                inception_full.eval().to(self.device)
                
                with torch.no_grad():
                    pred = inception_full(batch)
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()
        
        # Calculate IS
        scores = []
        for i in range(splits):
            part = predictions[i * (len(predictions) // splits): (i + 1) * (len(predictions) // splits)]
            
            # Calculate KL divergence
            py = np.mean(part, axis=0)
            scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
        
        return np.mean(scores), np.std(scores)
    
    def analyze_dataset_distribution(self, folder_path):
        """Analisis distribusi warna dan tekstur dalam dataset"""
        print(f"Analyzing dataset distribution for {os.path.basename(folder_path)}...")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Statistik warna
        mean_colors = []
        std_colors = []
        
        for img_file in tqdm(image_files[:100], desc="Analyzing images"):  # Sample 100 images
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img) / 255.0
                
                mean_colors.append(np.mean(img_array, axis=(0, 1)))
                std_colors.append(np.std(img_array, axis=(0, 1)))
                
            except Exception as e:
                continue
        
        mean_colors = np.array(mean_colors)
        std_colors = np.array(std_colors)
        
        return {
            'mean_rgb': np.mean(mean_colors, axis=0),
            'std_rgb': np.mean(std_colors, axis=0),
            'num_images': len(image_files)
        }
    
    def generate_evaluation_report(self, original_folder, generated_folder, class_name):
        """Generate laporan evaluasi lengkap"""
        print(f"\n=== EVALUATION REPORT: {class_name} ===")
        
        # Cek apakah folder ada
        if not os.path.exists(original_folder):
            print(f"Error: Original folder {original_folder} tidak ditemukan!")
            return
        
        if not os.path.exists(generated_folder):
            print(f"Error: Generated folder {generated_folder} tidak ditemukan!")
            return
        
        try:
            # FID Score
            fid_score = self.calculate_fid(original_folder, generated_folder)
            print(f"FID Score: {fid_score:.2f} (lower is better)")
            
            # Inception Score
            is_mean, is_std = self.calculate_inception_score(generated_folder)
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f} (higher is better)")
            
            # Dataset analysis
            original_stats = self.analyze_dataset_distribution(original_folder)
            generated_stats = self.analyze_dataset_distribution(generated_folder)
            
            print(f"\nDataset Statistics:")
            print(f"Original images: {original_stats['num_images']}")
            print(f"Generated images: {generated_stats['num_images']}")
            
            print(f"\nColor Distribution (RGB):")
            print(f"Original  - Mean: {original_stats['mean_rgb']}")
            print(f"Generated - Mean: {generated_stats['mean_rgb']}")
            print(f"Difference: {np.abs(original_stats['mean_rgb'] - generated_stats['mean_rgb'])}")
            
            # Quality assessment
            quality_score = self._assess_quality(fid_score, is_mean)
            print(f"\nOverall Quality Assessment: {quality_score}")
            
            return {
                'fid_score': fid_score,
                'inception_score': (is_mean, is_std),
                'original_stats': original_stats,
                'generated_stats': generated_stats,
                'quality_assessment': quality_score
            }
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    def _assess_quality(self, fid_score, is_score):
        """Berikan penilaian kualitas berdasarkan skor"""
        if fid_score < 50 and is_score > 2.0:
            return "Excellent ⭐⭐⭐⭐⭐"
        elif fid_score < 100 and is_score > 1.5:
            return "Good ⭐⭐⭐⭐"
        elif fid_score < 150 and is_score > 1.0:
            return "Fair ⭐⭐⭐"
        elif fid_score < 200:
            return "Poor ⭐⭐"
        else:
            return "Very Poor ⭐"

def evaluate_all_classes():
    """Evaluasi semua kelas yang telah di-augmentasi"""
    
    base_dir = r"c:\Riset Infromatika\Python V2"
    original_base = os.path.join(base_dir, "Dataset Original", "train")
    generated_base = os.path.join(base_dir, "Dataset Augmented", "train")
    
    classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']
    evaluator = ImageQualityEvaluator()
    
    results = {}
    
    for class_name in classes:
        original_folder = os.path.join(original_base, class_name)
        generated_folder = os.path.join(generated_base, class_name)
        
        if os.path.exists(original_folder) and os.path.exists(generated_folder):
            result = evaluator.generate_evaluation_report(
                original_folder, generated_folder, class_name
            )
            results[class_name] = result
        else:
            print(f"Skipping {class_name} - folders not found")
    
    # Generate summary report
    print("\n" + "="*50)
    print("SUMMARY EVALUATION REPORT")
    print("="*50)
    
    avg_fid = []
    avg_is = []
    
    for class_name, result in results.items():
        if result:
            print(f"{class_name:12} | FID: {result['fid_score']:6.1f} | IS: {result['inception_score'][0]:4.2f} | {result['quality_assessment']}")
            avg_fid.append(result['fid_score'])
            avg_is.append(result['inception_score'][0])
    
    if avg_fid and avg_is:
        print("-" * 50)
        print(f"{'Average':12} | FID: {np.mean(avg_fid):6.1f} | IS: {np.mean(avg_is):4.2f}")
        
        print(f"\nRecommendations:")
        if np.mean(avg_fid) > 150:
            print("- Consider training for more epochs")
            print("- Adjust learning rate or network architecture")
        if np.mean(avg_is) < 1.5:
            print("- Improve discriminator training")
            print("- Use progressive growing or spectral normalization")
    
    return results

if __name__ == "__main__":
    evaluate_all_classes()
