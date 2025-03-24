# few_shot_object_detection
# Small Sample Object Detection

This project demonstrates an approach to improve YOLOv8 performance on small sample object detection tasks by augmenting the dataset using diffusion models and adversarial samples.

## Project Structure
small_sample_object_detection/ 
├── data/ 
│ ├── train/ 
│ │ ├── images/ 
│ │ └── labels/ 
│ ├── val/ 
│ │ ├── images/ 
│ │ └── labels/ 
│ ├── test/ 
│ │ ├── images/ 
│ │ └── labels/ 
│ └── augmented/ 
│ ├── images/ 
│ └── labels/ 
├── scripts/ 
│ ├── prepare_coco_subset.py 
│ ├── train_and_evaluate_yolov8.py 
│ ├── generate_diffusion_samples.py 
│ ├── generate_adversarial_samples.py 
│ ├── train_and_evaluate_augmented_yolov8.py 
│ ├── data.yaml 
│ └── data_augmented.yaml 
├── requirements.txt 
└── README.md

导入项目
git clone https://github.com/your-username/small_sample_object_detection.git
cd small_sample_object_detection

安装要求
pip install -r requirements.txt

准备数据集（可换成其他数据集）
mkdir -p ~/datasets/coco_subset
cd ~/datasets/coco_subset
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

step1:数据集预处理：
python scripts/prepare_coco_subset.py

step2:开始训练评估
python scripts/train_and_evaluate_yolov8.py

插入：可视化训练结果：
1.使用Tensorboard查看训练过程
tensorboard --logdir runs/detect/yolov8_small_sample2/
2.查看保存的模型权重：
训练过程中保存的模型权重文件位于 runs/detect/yolov8_small_sample2/weights/ 目录下
last.pt为最后一次训练的模型权重，best.pt为最佳评估的模型权重
3.查看评估结果：评估结果对象包含了各种评估指标和混淆矩阵，您通过 Python 脚本来加载和查看这些结果。
from ultralytics.utils.metrics import DetMetrics
import pickle

%加载评估结果对象
results_path = 'runs/detect/yolov8_small_sample2/results.pkl'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

%打印AP类别索引
print("AP Class Index:", results.ap_class_index)

%打印混淆矩阵
print("Confusion Matrix:")
print(results.confusion_matrix)

%打印评估曲线
print("Evaluation Curves:", results.curves)

%查看边界框相关的评估指标
print("Box Metrics:", results.box)

4.绘制precision_recall曲线：
import matplotlib.pyplot as plt

%绘制 Precision-Recall 曲线
for curve in results.curves:
    if 'Precision-Recall' in curve:
        plt.plot(results.curves[curve]['x'], results.curves[curve]['y'], label=curve)
        
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()



step3:扩散生成样本
python scripts/generate_diffusion_samples.py

step4:对抗生成样本
python scripts/generate_adversarial_samples.py

step5:使用增强数据进行训练和评估
python scripts/train_and_evaluate_augmented_yolov8.py
