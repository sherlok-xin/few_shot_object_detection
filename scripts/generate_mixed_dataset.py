import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import yaml
from tqdm import tqdm
import shutil
import numpy as np

def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    return path

def verify_data_structure(config_path):
    """验证数据目录结构"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = os.path.dirname(config_path)
    required_dirs = {
        'train_images': os.path.join(base_dir, '..', 'data/train/images'),
        'train_labels': os.path.join(base_dir, '..', 'data/train/labels'),
        'val_images': os.path.join(base_dir, '..', 'data/val/images'),
        'val_labels': os.path.join(base_dir, '..', 'data/val/labels'),
    }
    
    # 确保所有必需的目录都存在
    for dir_name, dir_path in required_dirs.items():
        ensure_directory_exists(dir_path)
    
    return True

class FGSM:
    def __init__(self, model, epsilon=0.07):
        self.model = model
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def generate(self, image):
        """生成FGSM对抗样本"""
        image = image.convert('RGB')
        x = self.transform(image).unsqueeze(0).to(self.device)
        x.requires_grad = True
        
        output = self.model(x)
        target = output.argmax(dim=1)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        perturbed_x = x + self.epsilon * x.grad.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        perturbed_image = transforms.ToPILImage()(perturbed_x.squeeze(0).cpu())
        
        return perturbed_image

def generate_fgsm_samples(config_path, num_samples_per_image=1):
    """生成FGSM对抗样本"""
    print("开始验证数据目录结构...")
    verify_data_structure(config_path)
    
    # 确保配置文件路径是绝对路径
    config_path = os.path.abspath(config_path)
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取训练数据目录的绝对路径
    base_dir = os.path.dirname(config_path)
    train_images_dir = os.path.join(base_dir, '..', 'data/train/images')
    train_labels_dir = os.path.join(base_dir, '..', 'data/train/labels')
    
    print(f"Looking for images in: {train_images_dir}")
    
    # 设置输出路径
    output_base = os.path.join(base_dir, '..', 'data/augmented_fgsm')
    output_images_dir = ensure_directory_exists(os.path.join(output_base, 'images'))
    output_labels_dir = ensure_directory_exists(os.path.join(output_base, 'labels'))
    
    print(f"Output directory: {output_base}")
    
    # 加载预训练模型
    print("Loading pretrained ResNet model...")
    model = models.resnet18(pretrained=True)
    fgsm = FGSM(model)
    
    # 检查训练图像目录
    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(f"训练图像目录不存在: {train_images_dir}")
    
    image_files = [f for f in os.listdir(train_images_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"警告：在 {train_images_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print("开始生成FGSM对抗样本...")
    
    for img_name in tqdm(image_files):
        try:
            img_path = os.path.join(train_images_dir, img_name)
            original_image = Image.open(img_path).convert('RGB')
            
            for i in range(num_samples_per_image):
                perturbed_image = fgsm.generate(original_image)
                
                base_name = os.path.splitext(img_name)[0]
                ext = os.path.splitext(img_name)[1]
                output_img_path = os.path.join(
                    output_images_dir,
                    f'fgsm_{base_name}_{i}{ext}'
                )
                perturbed_image.save(output_img_path)
                
                label_name = f"{base_name}.txt"
                original_label_path = os.path.join(train_labels_dir, label_name)
                if os.path.exists(original_label_path):
                    output_label_path = os.path.join(
                        output_labels_dir,
                        f'fgsm_{base_name}_{i}.txt'
                    )
                    shutil.copy2(original_label_path, output_label_path)
                
        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {e}")
            continue

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'data.yaml')
        
        print("开始生成FGSM对抗样本...")
        print(f"配置文件路径: {config_path}")
        
        generate_fgsm_samples(config_path, num_samples_per_image=1)
        print("FGSM对抗样本生成完成！")
    except Exception as e:
        print(f"发生错误: {e}")
