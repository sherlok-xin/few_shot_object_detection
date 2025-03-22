from ultralytics import YOLO
import os

# 数据配置文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_config = os.path.join(current_dir, 'data.yaml')

# 确保数据配置文件存在
if not os.path.exists(data_config):
    raise RuntimeError(f"数据配置文件不存在: {data_config}")

# 加载YOLOv8模型
model = YOLO('yolov8s.pt')

# 训练模型
model.train(data=data_config, epochs=100, imgsz=640, batch=16, name='yolov8_small_sample', device=0)  # 使用第一块GPU

# 初次评估模型
initial_results = model.val(data=data_config, device=0)  # 使用第一块GPU
print("初次评估结果:", initial_results)