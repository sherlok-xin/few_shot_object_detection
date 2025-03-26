import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import yaml
import shutil
from tqdm import tqdm

def setup_directories(base_path):
    """设置必要的目录结构"""
    dirs = ['images', 'labels']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)
    return base_path

def generate_diffusion_samples(config_path, num_samples=100):
    """使用扩散模型生成数据样本"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置输出路径
    output_base = os.path.join(os.path.dirname(config['train']), 'augmented_diffusion')
    setup_directories(output_base)
    
    # 初始化扩散模型
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    # 获取类别信息
    classes = config.get('names', [])
    
    for class_name in tqdm(classes, desc="Generating samples for classes"):
        prompt = f"A clear photo of a {class_name}, high quality, detailed"
        
        for i in range(num_samples):
            # 生成图像
            image = pipe(prompt).images[0]
            
            # 保存图像
            image_path = os.path.join(output_base, 'images', f"{class_name}_{i}.png")
            image.save(image_path)
            
            # 创建对应的标签文件
            label_path = os.path.join(output_base, 'labels', f"{class_name}_{i}.txt")
            # 这里需要根据您的具体需求修改标签格式
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.8 0.8")  # 示例标签格式

if __name__ == "__main__":
    generate_diffusion_samples('scripts/data.yaml')
