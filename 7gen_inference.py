# 7Gen Inference - Rakam Üretme Arayüzü
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Model yapısı (eğitimde kullandığımız ile aynı olmalı)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 7Gen sınıfı
class SevenGenInference:
    def __init__(self, model_path='models/7gen_generator.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = 100
        
        # Modeli yükle
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator.eval()
        
        print(f"🚀 7Gen yüklendi! Cihaz: {self.device}")
    
    def generate_digit(self, digit, count=5):
        """Belirli bir rakamdan istenen sayıda üret"""
        with torch.no_grad():
            z = torch.randn(count, self.latent_dim).to(self.device)
            labels = torch.full((count,), digit).to(self.device)
            
            images = self.generator(z, labels)
            images = (images + 1) / 2  # [-1,1] -> [0,1]
            
            return images.cpu()
    
    def visualize_digits(self, digit, count=5, save_path=None):
        """Üretilen rakamları görselleştir"""
        images = self.generate_digit(digit, count)
        
        fig, axes = plt.subplots(1, count, figsize=(2*count, 2))
        if count == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            ax.imshow(images[i][0], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Digit: {digit}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Görsel kaydedildi: {save_path}")
        
        plt.show()
    
    def generate_grid(self, samples_per_digit=10, save_path=None):
        """Her rakamdan örneklerle 10x10 grid oluştur"""
        all_images = []
        
        for digit in range(10):
            images = self.generate_digit(digit, samples_per_digit)
            all_images.append(images)
        
        all_images = torch.cat(all_images, dim=0)
        
        fig, axes = plt.subplots(10, samples_per_digit, figsize=(15, 15))
        
        for i in range(10):
            for j in range(samples_per_digit):
                idx = i * samples_per_digit + j
                axes[i, j].imshow(all_images[idx][0], cmap='gray')
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(f'{i}', rotation=0, size=20, labelpad=20)
        
        plt.suptitle('7Gen - Üretilen Rakamlar', size=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Grid kaydedildi: {save_path}")
        
        plt.show()
    
    def save_as_png(self, digit, count=1, output_dir='output'):
        """Tekil PNG dosyaları olarak kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.generate_digit(digit, count)
        
        for i in range(count):
            img = images[i][0].numpy()
            img = (img * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img)
            filename = f"{output_dir}/digit_{digit}_{i+1}.png"
            pil_img.save(filename)
            
            print(f"💾 Kaydedildi: {filename}")
    
    def interactive_generate(self):
        """İnteraktif kullanım"""
        print("\n🎮 7Gen İnteraktif Mod")
        print("Çıkmak için 'q' yazın")
        
        while True:
            try:
                digit_input = input("\nHangi rakamı üretmek istersin? (0-9): ")
                
                if digit_input.lower() == 'q':
                    print("👋 Görüşürüz!")
                    break
                
                digit = int(digit_input)
                if 0 <= digit <= 9:
                    count = int(input("Kaç tane üreteyim? (1-20): "))
                    if 1 <= count <= 20:
                        self.visualize_digits(digit, count)
                    else:
                        print("❌ 1-20 arası bir sayı gir!")
                else:
                    print("❌ 0-9 arası bir rakam gir!")
                    
            except ValueError:
                print("❌ Geçerli bir sayı gir!")
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break

# Ana kullanım
if __name__ == "__main__":
    # 7Gen'i başlat
    seven_gen = SevenGenInference()
    
    # Örnekler
    print("\n📝 Örnek kullanımlar:")
    print("1. Tekil rakam üret")
    seven_gen.visualize_digits(digit=7, count=5)
    
    print("\n2. Grid oluştur")
    seven_gen.generate_grid(samples_per_digit=10, save_path='7gen_showcase.png')
    
    print("\n3. PNG olarak kaydet")
    seven_gen.save_as_png(digit=5, count=3, output_dir='output')
    
    print("\n4. İnteraktif mod")
    seven_gen.interactive_generate()