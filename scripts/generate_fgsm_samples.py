import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import yaml
from tqdm import tqdm

def fgsm_attack(image, epsilon, data_grad):
    """执行FGSM攻击"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_fgsm_samples(config_path, epsilon=0.07):
    """生成FGSM对抗样本"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置输出路径
    output_base = os.path.join(os.path.dirname(config['train']), 'augmented_fgsm')
    os.makedirs(os.path.join(output_base, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, 'labels'), exist_ok=True)
    
    # 加载YOLOv8模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    # 处理训练集中的每张图片
    train_path = config['train']
    images_path = os.path.join(train_path, 'images')
    labels_path = os.path.join(train_path, 'labels')
    
    for img_name in tqdm(os.listdir(images_path)):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        # 加载原始图像
        img_path = os.path.join(images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 转换为tensor
        img_tensor = transform(image).unsqueeze(0)
        img_tensor.requires_grad = True
        
        # 获取模型预测
        output = model(img_tensor)
        loss = F.cross_entropy(output, torch.tensor([0]))  # 示例loss
        
        # 计算梯度
        loss.backward()
        
        # 生成对抗样本
        perturbed_image = fgsm_attack(img_tensor, epsilon, img_tensor.grad.data)
        
        # 保存对抗样本
        output_img_path = os.path.join(output_base, 'images', f'fgsm_{img_name}')
        save_image = transforms.ToPILImage()(perturbed_image.squeeze())
        save_image.save(output_img_path)
        
        # 复制对应的标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label_path = os.path.join(labels_path, label_name)
        dst_label_path = os.path.join(output_base, 'labels', f'fgsm_{label_name}')
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

if __name__ == "__main__":
    generate_fgsm_samples('scripts/data.yaml')
