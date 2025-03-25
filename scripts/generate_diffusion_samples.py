import os
from diffusers import DDPMPipeline
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluate import evaluate_detection  

# 检查并设置 CUDA 设备
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check that you have installed the NVIDIA driver and CUDA.")

# 加载模型和管道
model_id = "google/ddpm-cifar10-32"
pipeline = DDPMPipeline.from_pretrained(model_id)

# 将模型移动到 CUDA 设备
pipeline.to("cuda")

# 生成样本
num_samples = 10
samples = pipeline(num_samples).images

# 保存样本
output_dir = "data/augmented/images"
os.makedirs(output_dir, exist_ok=True)
for i, sample in enumerate(samples):
    sample.save(os.path.join(output_dir, f"diff_sample_{i}.png"))

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
