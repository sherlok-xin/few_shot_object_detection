# few_shot_object_detection

## Small Sample Object Detection

利用YOLOv8进行小样本目标检测（few_shot_object_detection），并利用扩散模型（diffusion）以及对抗样本生成技术扩充数据集以提高算法性能

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
│       └── labels/
├── scripts/
│   ├── prepare_coco_subset.py
│   ├── train_and_evaluate_yolov8.py
│   ├── generate_diffusion_samples.py
│   ├── generate_fgsm_samples.py
│   ├── generate_weighted_samples.py
│   ├── train_and_evaluate_augmented_yolov8.py
│   ├── data.yaml
│   └── data_augmented.yaml
├── requirements.txt
└── README.md
```

## Setup

### Clone the repository

```bash
git clone https://github.com/sherlok-xin/small_sample_object_detection.git
cd small_sample_object_detection
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

### Step 2: Train and Evaluate YOLOv8

```bash
python scripts/train_and_evaluate_yolov8.py
```

### Step 3: Visualize Training Results

1. **Use TensorBoard to view the training process**

```bash
tensorboard --logdir runs/detect/yolov8_small_sample/
```

2. **View saved model weights**

Model weights saved during training are located in the `runs/detect/yolov8_small_sample/weights/` directory. `last.pt` is the final model weight, and `best.pt` is the best evaluated model weight.

3. **View evaluation results**

```python
from ultralytics.yolov8.utils.metrics import DetMetrics
import pickle

# Load evaluation results
results_path = 'runs/detect/yolov8_small_sample/results.pkl'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Print AP class index
print("AP Class Index:", results.ap_class_index)

# Print confusion matrix
print("Confusion Matrix:")
print(results.confusion_matrix)

# Print evaluation curves
print("Evaluation Curves:", results.curves)

# View box metrics
print("Box Metrics:", results.box)
```

4. **Plot Precision-Recall curve**

```python
import matplotlib.pyplot as plt

# Plot Precision-Recall curve
for curve in results.curves:
    if 'Precision-Recall' in curve:
        plt.plot(results.curves[curve]['x'], results.curves[curve]['y'], label=curve)
        
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### Step 4: Generate Diffusion Samples

```bash
python scripts/generate_diffusion_samples.py
```

### Step 5: Generate Adversarial Samples with FGSM

```bash
python scripts/generate_fgsm_samples.py
```

### Step 6: Generate Weighted Samples

```bash
python scripts/generate_weighted_samples.py
```

### Step 7: Train and Evaluate with Augmented Data

```bash
python scripts/train_and_evaluate_augmented_yolov8.py
```
