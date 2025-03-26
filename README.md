# Few-Shot Object Detection Data Augmentation Experiment

## 实验目标

探究扩散模型和对抗样本生成技术（FGSM）在 few-shot object detection 中的数据增强效果，并找到最佳的增强策略以提升模型性能和鲁棒性。

## 实验流程

### 数据准备

1. 使用 `prepare_coco_subset.py` 文件准备 COCO 数据集，划分训练集、验证集和测试集。
2. 使用 `generate_diffusion_samples.py` 文件生成基于扩散模型的增强数据集。
3. 使用 `generate_fgsm_samples.py` 文件生成基于 FGSM 的对抗样本数据集。

### 实验分组

1. **对照组**：仅使用原始 COCO 数据集进行训练和评估，作为基线。
2. **扩散模型增强组**：使用原始数据集 + 扩散模型生成的数据集进行训练和评估。
3. **FGSM 增强组**：使用原始数据集 + FGSM 生成的对抗样本数据集进行训练和评估。
4. **混合增强组**：使用原始数据集 + 扩散模型生成的数据集 + FGSM 生成的对抗样本数据集进行训练和评估，并尝试不同的混合比例（例如 1:1、1:2、2:1 等，要求能够自行调节两种生成数据集的权重）。

### 模型训练和评估

使用 `train_and_evaluate_yolov8.py` 文件对每组实验进行训练和评估，记录 mAP、Precision、Recall 等指标。

### 结果分析

1. 对比各组实验结果，分析不同数据增强方法对模型性能的影响。
2. 比较不同混合比例下混合增强组的性能，寻找最佳的增强策略。
3. 分析数据增强方法对模型鲁棒性和泛化能力的影响。
## Project Structure

```
small_sample_object_detection/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── augmented/
│       ├── images/
│       
├── scripts/
│   ├── prepare_coco_subset.py
│   ├── train_and_evaluate_yolov8.py
│   ├── generate_diffusion_samples.py
│   ├── generate_fgsm_samples.py
│   ├── compare_results.py
│   ├── train_and_evaluate_with_mixed_data.py
│   ├── evaluate.py
│   ├── data.yaml
│   └── data_augmented.yaml
├── requirements.txt
└── README.md
```

## Setup

### Clone the repository

```bash
git clone https://github.com/sherlok-xin/few_shot_object_detection.git
cd few_shot_object_detection
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Prepare the dataset (replace with your own if needed)

```bash
mkdir -p ~/datasets/coco_subset
cd ~/datasets/coco_subset
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

## Experiment Steps

### Step 1: Data Preprocessing

```bash
python scripts/prepare_coco_subset.py
```

### Step 2: Train and Evaluate YOLOv8(对照组)

```bash
python scripts/train_and_evaluate_yolov8.py
```

### Step 3: Generate Diffusion Samples

```bash
python scripts/generate_diffusion_samples.py
```

### Step 4: Generate Adversarial Samples with FGSM

```bash
python scripts/generate_fgsm_samples.py
```

### Step 5: Generate Weighted Samples

```bash
python scripts/train_and_evaluate_with_mixed_data.py
```

### Step 7: Train and Evaluate with Data

```bash
python scripts/train_and_evaluate_yolov8.py
```
