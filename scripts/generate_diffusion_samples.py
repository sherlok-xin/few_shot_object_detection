import os
import numpy as np
from PIL import Image, ImageEnhance
import yaml
from tqdm import tqdm
import shutil
import random

class SimpleAugmentor:
    @staticmethod
    def rotate(image, angle):
        """旋转图像"""
        return image.rotate(angle)
    
    @staticmethod
    def adjust_brightness(image, factor):
        """调整亮度"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_contrast(image, factor):
        """调整对比度"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def add_noise(image, level=0.1):
        """添加高斯噪声"""
        img_array = np.array(image)
        noise = np.random.normal(0, level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    @staticmethod
    def horizontal_flip(image):
        """水平翻转"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)

def setup_directories(base_path):
    """设置必要的目录结构"""
    dirs = ['images', 'labels']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)
    return base_path

def get_absolute_path(path, config_file_path):
    """获取绝对路径"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(config_file_path), path))

def generate_augmented_samples(config_path, num_samples_per_image=2):
    """使用简单的图像增强方法生成数据样本"""
    # 确保配置文件路径是绝对路径
    config_path = os.path.abspath(config_path)
    
    # 加载配置
    print(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取训练数据目录的绝对路径
    train_path = get_absolute_path(config['train'], config_path)
    original_images_path = os.path.join(train_path, 'images')
    original_labels_path = os.path.join(train_path, 'labels')
    
    print(f"Looking for images in: {original_images_path}")
    if not os.path.exists(original_images_path):
        raise FileNotFoundError(f"训练图像目录不存在: {original_images_path}")
        
    # 设置输出路径
    output_base = os.path.join(os.path.dirname(train_path), 'augmented_diffusion')
    setup_directories(output_base)
    
    print(f"Output directory: {output_base}")
    
    augmentor = SimpleAugmentor()
    augmentation_methods = [
        (augmentor.rotate, {'angle': 15}),
        (augmentor.rotate, {'angle': -15}),
        (augmentor.adjust_brightness, {'factor': 1.2}),
        (augmentor.adjust_contrast, {'factor': 1.2}),
        (augmentor.add_noise, {'level': 0.1}),
        (augmentor.horizontal_flip, {})
    ]
    
    # 检查训练图像目录是否为空
    image_files = [f for f in os.listdir(original_images_path) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"警告：在 {original_images_path} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print("开始生成增强样本...")
    
    for img_name in tqdm(image_files):
        # 加载原始图像
        img_path = os.path.join(original_images_path, img_name)
        try:
            original_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图像 {img_path}: {e}")
            continue
        
        # 获取对应的标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        original_label_path = os.path.join(original_labels_path, label_name)
        
        # 随机选择增强方法
        selected_methods = random.sample(augmentation_methods, num_samples_per_image)
        
        for i, (method, params) in enumerate(selected_methods):
            try:
                # 应用增强方法
                augmented_image = method(original_image, **params)
                
                # 保存增强后的图像
                aug_img_name = f"aug_{i}_{img_name}"
                aug_img_path = os.path.join(output_base, 'images', aug_img_name)
                augmented_image.save(aug_img_path)
                
                # 复制并调整标签文件
                if os.path.exists(original_label_path):
                    aug_label_name = f"aug_{i}_{label_name}"
                    aug_label_path = os.path.join(output_base, 'labels', aug_label_name)
                    
                    # 如果是水平翻转，需要调整标签坐标
                    if method == augmentor.horizontal_flip:
                        with open(original_label_path, 'r') as f:
                            lines = f.readlines()
                        
                        with open(aug_label_path, 'w') as f:
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # 水平翻转时，x坐标需要变换：x_new = 1 - x
                                    x = 1 - float(parts[1])
                                    parts[1] = str(x)
                                f.write(' '.join(parts) + '\n')
                    else:
                        # 对于其他增强方法，直接复制标签文件
                        shutil.copy2(original_label_path, aug_label_path)
            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {e}")
                continue

if __name__ == "__main__":
    try:
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'data.yaml')
        
        print("开始数据增强过程...")
        print(f"配置文件路径: {config_path}")
        
        generate_augmented_samples(config_path)
        print("数据增强完成！")
    except Exception as e:
        print(f"发生错误: {e}")
