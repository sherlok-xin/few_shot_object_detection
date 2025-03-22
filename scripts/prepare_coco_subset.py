import os
import shutil
import random
from pycocotools.coco import COCO

# 设置随机种子以确保结果可重复
random.seed(42)

# COCO 数据集路径
coco_data_dir = '/content/datasets/coco_subset'  # 更新为实际路径
coco_val_images_dir = os.path.join(coco_data_dir, 'val2017')
coco_val_annotations_file = os.path.join(coco_data_dir, 'annotations', 'instances_val2017.json')

# 输出路径
output_dir = '/content/small_sample_object_detection/data'
train_output_dir = os.path.join(output_dir, 'train')
val_output_dir = os.path.join(output_dir, 'val')
test_output_dir = os.path.join(output_dir, 'test')

# 创建输出目录
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(os.path.join(train_output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_output_dir, 'labels'), exist.ok=True)
os.makedirs(val_output_dir, exist.ok=True)
os.makedirs(os.path.join(val_output_dir, 'images'), exist.ok=True)
os.makedirs(os.path.join(val_output_dir, 'labels'), exist.ok=True)
os.makedirs(test_output_dir, exist.ok=True)
os.makedirs(os.path.join(test_output_dir, 'images'), exist.ok=True)
os.makedirs(os.path.join(test_output_dir, 'labels'), exist.ok=True)

# 加载 COCO 数据集
coco_val = COCO(coco_val_annotations_file)

# 获取所有图像 ID
val_img_ids = coco_val.getImgIds()

# 随机选择训练、验证和测试集
random.shuffle(val_img_ids)

num_train = int(0.6 * len(val_img_ids))  # 60% 训练集
num_val = int(0.2 * len(val_img_ids))  # 20% 验证集
train_ids = val_img_ids[:num_train]
val_ids = val_img_ids[num_train:num_train + num_val]
test_ids = val_img_ids[num_train + num_val:]  # 剩余的作为测试集

def save_coco_subset(coco, image_ids, images_dir, output_dir):
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 复制图像文件
        src_image_path = os.path.join(images_dir, img_info['file_name'])
        dst_image_path = os.path.join(output_dir, 'images', img_info['file_name'])
        
        # 检查源图像文件是否存在
        if not os.path.exists(src_image_path):
            print(f"文件不存在: {src_image_path}")
            continue
        
        shutil.copy(src_image_path, dst_image_path)
        
        # 创建标签文件
        label_file_path = os.path.join(output_dir, 'labels', img_info['file_name'].replace('.jpg', '.txt'))
        with open(label_file_path, 'w') as f:
            for ann in anns:
                category_id = ann['category_id'] - 1  # 类别 ID 从 0 开始
                if category_id < 0 or category_id >= 80:
                    print(f"忽略无效标签: {category_id} 在图像 {img_info['file_name']}")
                    continue
                bbox = ann['bbox']
                center_x = (bbox[0] + bbox[2] / 2) / img_info['width']
                center_y = (bbox[1] + bbox[3] / 2) / img_info['height']
                width = bbox[2] / img_info['width']
                height = bbox[3] / img_info['height']
                f.write(f'{category_id} {center_x} {center_y} {width} {height}\n')

# 保存训练集
print(f"保存训练集到 {train_output_dir}")
save_coco_subset(coco_val, train_ids, coco_val_images_dir, train_output_dir)

# 保存验证集
print(f"保存验证集到 {val_output_dir}")
save_coco_subset(coco_val, val_ids, coco_val_images_dir, val_output_dir)

# 保存测试集
print(f"保存测试集到 {test_output_dir}")
save_coco_subset(coco_val, test_ids, coco_val_images_dir, test_output_dir)

print('数据集准备完成')