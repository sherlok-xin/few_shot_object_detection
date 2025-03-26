import os
import shutil
import random
import yaml
from tqdm import tqdm

def create_mixed_dataset(config_path, diffusion_ratio, fgsm_ratio):
    """创建混合增强数据集"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = os.path.dirname(config['train'])
    ratio_str = f"{diffusion_ratio}_{fgsm_ratio}"
    
    # 设置输出路径
    output_base = os.path.join(base_path, f'augmented_mixed_{ratio_str}')
    os.makedirs(os.path.join(output_base, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, 'labels'), exist_ok=True)
    
    # 源数据路径
    original_path = config['train']
    diffusion_path = os.path.join(base_path, 'augmented_diffusion')
    fgsm_path = os.path.join(base_path, 'augmented_fgsm')
    
    # 复制原始数据
    print("Copying original dataset...")
    for subdir in ['images', 'labels']:
        src_dir = os.path.join(original_path, subdir)
        dst_dir = os.path.join(output_base, subdir)
        for file_name in os.listdir(src_dir):
            shutil.copy2(
                os.path.join(src_dir, file_name),
                os.path.join(dst_dir, file_name)
            )
    
    # 按比例选择扩散模型样本
    print("Adding diffusion samples...")
    diffusion_samples = os.listdir(os.path.join(diffusion_path, 'images'))
    num_diffusion = int(len(diffusion_samples) * (diffusion_ratio / (diffusion_ratio + fgsm_ratio)))
    selected_diffusion = random.sample(diffusion_samples, num_diffusion)
    
    for img_name in tqdm(selected_diffusion):
        # 复制图像
        shutil.copy2(
            os.path.join(diffusion_path, 'images', img_name),
            os.path.join(output_base, 'images', f'diff_{img_name}')
        )
        # 复制标签
        label_name = os.path.splitext(img_name)[0] + '.txt'
        shutil.copy2(
            os.path.join(diffusion_path, 'labels', label_name),
            os.path.join(output_base, 'labels', f'diff_{label_name}')
        )
    
    # 按比例选择FGSM样本
    print("Adding FGSM samples...")
    fgsm_samples = os.listdir(os.path.join(fgsm_path, 'images'))
    num_fgsm = int(len(fgsm_samples) * (fgsm_ratio / (diffusion_ratio + fgsm_ratio)))
    selected_fgsm = random.sample(fgsm_samples, num_fgsm)
    
    for img_name in tqdm(selected_fgsm):
        # 复制图像
        shutil.copy2(
            os.path.join(fgsm_path, 'images', img_name),
            os.path.join(output_base, 'images', f'fgsm_{img_name}')
        )
        # 复制标签
        label_name = os.path.splitext(img_name)[0] + '.txt'
        shutil.copy2(
            os.path.join(fgsm_path, 'labels', label_name),
            os.path.join(output_base, 'labels', f'fgsm_{label_name}')
        )

def main():
    # 创建不同比例的混合数据集
    ratios = [
        (1, 1),  # 1:1
        (1, 2),  # 1:2
        (2, 1)   # 2:1
    ]
    
    for diff_ratio, fgsm_ratio in ratios:
        print(f"\nCreating mixed dataset with ratio {diff_ratio}:{fgsm_ratio}")
        create_mixed_dataset('scripts/data.yaml', diff_ratio, fgsm_ratio)

if __name__ == "__main__":
    main()
