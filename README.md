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