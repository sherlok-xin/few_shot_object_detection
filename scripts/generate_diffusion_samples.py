import os
from diffusers import DDPMPipeline
import torch

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
output_dir = "/content/drive/MyDrive/Colab Notebooks/few_shot_object_detection/samples_diffusion"
os.makedirs(output_dir, exist_ok=True)
for i, sample in enumerate(samples):
    sample.save(os.path.join(output_dir, f"sample_{i}.png"))

# 进行检测评估
# 假设检测评估函数为 evaluate_detection(samples)
def evaluate_detection(samples):
    # 这里填写您的检测评估逻辑
    pass

evaluate_detection(samples)
