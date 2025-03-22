import torch
import torchvision.transforms as T
from PIL import Image
import os
from ultralytics import YOLO

# 加载已经训练好的YOLOv8模型
model = YOLO('yolov8s.pt')
model.load('runs/detect/yolov8_small_sample/weights/best.pt')

# 对抗样本生成函数
def generate_adversarial_sample(image_path, epsilon):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # 确保模型在评估模式
    model.model.eval()

    # 需要梯度计算
    image_tensor.requires_grad = True

    # 前向传播
    outputs = model(image_tensor)

    # 计算损失
    loss = -outputs[0, 0, 4]  # 假设目标是第一个检测框的置信度

    # 反向传播
    model.model.zero_grad()
    loss.backward()

    # 生成对抗样本
    adversarial_image = image_tensor + epsilon * image_tensor.grad.sign()
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return T.ToPILImage()(adversarial_image.squeeze())

# 输入图像路径和对抗强度
input_image_path = 'path/to/your/image.jpg'
epsilon = 0.01

# 输出目录
output_dir = "/content/small_sample_object_detection/data/augmented/images"
os.makedirs(output_dir, exist_ok=True)

# 生成并保存对抗样本
adversarial_image = generate_adversarial_sample(input_image_path, epsilon)
adversarial_image.save(os.path.join(output_dir, 'adversarial_image.jpg'))