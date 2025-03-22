import torch
from diffusers import DDPMPipeline
import os
from PIL import Image

# 扩散模型加载
model_id = "google/ddpm-cifar10-32"
pipeline = DDPMPipeline.from_pretrained(model_id)
pipeline.to("cuda")

# 输出目录
output_dir = "/content/small_sample_object_detection/data/augmented/images"
os.makedirs(output_dir, exist_ok=True)

# 生成样本
num_samples = 100  # 生成样本数量
images = pipeline(num_inference_steps=50, batch_size=num_samples).images

# 保存生成的样本
for i, img in enumerate(images):
    img = Image.fromarray(img)
    img.save(os.path.join(output_dir, f"generated_{i}.png"))

print("扩散模型生成样本完成")