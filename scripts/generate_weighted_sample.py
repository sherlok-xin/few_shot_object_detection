import os
from diffusers import DDPMPipeline
from gan_model import GANPipeline  # 假设您有一个 GAN 模型的管道
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluate import evaluate_detection  

def evaluate_detection(samples):
    # 这里填写您的检测评估逻辑
    pass

# 检查并设置 CUDA 设备
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check that you have installed the NVIDIA driver and CUDA.")

# 加载扩散模型和管道
diffusion_model_id = "google/ddpm-cifar10-32"
diffusion_pipeline = DDPMPipeline.from_pretrained(diffusion_model_id)
diffusion_pipeline.to("cuda")

# 加载对抗生成模型和管道
gan_model_id = "path_to_your_gan_model"
gan_pipeline = GANPipeline.from_pretrained(gan_model_id)
gan_pipeline.to("cuda")

# 设置权重
diffusion_weight = 0.5
gan_weight = 0.5

# 生成样本
num_samples = 10
diffusion_samples = diffusion_pipeline(num_samples).images
gan_samples = gan_pipeline(num_samples).images

# 加权样本
weighted_samples = []
for i in range(num_samples):
    diffusion_img = diffusion_samples[i]
    gan_img = gan_samples[i]
    
    # 将图像转换为张量
    diffusion_tensor = torch.tensor(diffusion_img) * diffusion_weight
    gan_tensor = torch.tensor(gan_img) * gan_weight
    
    # 加权融合
    weighted_tensor = diffusion_tensor + gan_tensor
    weighted_img = Image.fromarray(weighted_tensor.numpy().astype('uint8'))
    
    weighted_samples.append(weighted_img)

# 保存加权样本
output_dir = "data/augmented/images"
os.makedirs(output_dir, exist_ok=True)
for i, sample in enumerate(weighted_samples):
    sample.save(os.path.join(output_dir, f"weighted_sample_{i}.png"))

# 准备数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='data/augmented/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载检测模型
detection_model = torch.load('path_to_trained_model')  # 请替换为实际模型路径

# 进行检测评估
results = evaluate_detection(detection_model, dataloader)
print(results)
