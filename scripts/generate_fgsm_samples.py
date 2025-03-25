import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from PIL import Image
from evaluate import evaluate_detection  

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Load the pretrained model
model = resnet18(pretrained=True)
model.eval()

# Define the data loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='data/train/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Set the epsilon value for the FGSM attack
epsilon = 0.1

# Create directory to save adversarial samples
output_dir = "data/augmented/images"
os.makedirs(output_dir, exist_ok=True)

# Generate FGSM samples
samples = []
for i, (image, label) in enumerate(dataloader):
    image.requires_grad = True
    
    # Forward pass the data through the model
    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    # If the initial prediction is wrong, don't bother attacking, just move on
    if init_pred.item() != label.item():
        continue

    # Calculate the loss
    loss = F.nll_loss(output, label)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = image.grad.data

    # Call FGSM Attack
    perturbed_image = fgsm_attack(image, epsilon, data_grad)

    # Save the adversarial image
    img = transforms.ToPILImage()(perturbed_image.squeeze(0))
    img.save(os.path.join(output_dir, f"adv_sample_{i}.png"))
    samples.append(img)

    if i >= 9:  # Save only 10 samples
        break

# 准备数据加载器
dataset = datasets.ImageFolder(root='data/augmented/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载检测模型
detection_model = torch.load('path_to_trained_model')  # 请替换为实际模型路径

# 进行检测评估
results = evaluate_detection(detection_model, dataloader)
print(results)
