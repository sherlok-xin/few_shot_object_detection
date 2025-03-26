from ultralytics import YOLO
import os
import yaml
import json
from datetime import datetime
import numpy as np

class ExperimentManager:
    def __init__(self, base_config_path):
        self.base_config_path = base_config_path
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.current_dir, '..', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def prepare_experiment_config(self, experiment_type, mix_ratio=None):
        """准备不同实验组的配置"""
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if experiment_type == "control":
            return config
        
        # 创建实验特定的数据路径
        if experiment_type == "diffusion":
            config['train'] = os.path.join(config['train'], 'augmented_diffusion')
        elif experiment_type == "fgsm":
            config['train'] = os.path.join(config['train'], 'augmented_fgsm')
        elif experiment_type == "mixed":
            if not mix_ratio:
                raise ValueError("Mixed experiment requires mix_ratio")
            config['train'] = os.path.join(config['train'], f'augmented_mixed_{mix_ratio}')
            
        # 保存实验配置
        experiment_config_path = os.path.join(
            self.current_dir, 
            f'data_{experiment_type}.yaml'
        )
        with open(experiment_config_path, 'w') as f:
            yaml.dump(config, f)
            
        return experiment_config_path

    def run_experiment(self, experiment_type, mix_ratio=None):
        """运行单个实验"""
        config_path = self.prepare_experiment_config(experiment_type, mix_ratio)
        
        # 实验名称
        exp_name = f'yolov8_{experiment_type}'
        if mix_ratio:
            exp_name += f'_ratio_{mix_ratio}'
        
        # 加载模型
        model = YOLO('yolov8s.pt')
        
        # 训练模型
        results = model.train(
            data=config_path,
            epochs=100,
            imgsz=640,
            batch=16,
            name=exp_name,
            device=0
        )
        
        # 评估模型
        val_results = model.val(data=config_path, device=0)
        
        # 保存结果
        self.save_results(experiment_type, val_results, mix_ratio)
        
        return val_results
    
    def save_results(self, experiment_type, results, mix_ratio=None):
        """保存实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            self.results_dir,
            f'results_{experiment_type}{"_"+str(mix_ratio) if mix_ratio else ""}_{timestamp}.json'
        )
        
        metrics = {
            'experiment_type': experiment_type,
            'mix_ratio': mix_ratio,
            'timestamp': timestamp,
            'metrics': {
                'mAP': float(results.box.map),    # mAP_0.5:0.95
                'mAP50': float(results.box.map50), # mAP_0.5
                'mAP75': float(results.box.map75), # mAP_0.75
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)

def main():
    # 初始化实验管理器
    experiment_manager = ExperimentManager('scripts/data.yaml')
    
    # 运行对照组实验
    print("Running control group experiment...")
    control_results = experiment_manager.run_experiment("control")
    
    # 运行扩散模型增强组实验
    print("Running diffusion model experiment...")
    diffusion_results = experiment_manager.run_experiment("diffusion")
    
    # 运行FGSM增强组实验
    print("Running FGSM experiment...")
    fgsm_results = experiment_manager.run_experiment("fgsm")
    
    # 运行混合增强组实验（不同比例）
    mix_ratios = ["1_1", "1_2", "2_1"]  # 表示1:1, 1:2, 2:1的比例
    for ratio in mix_ratios:
        print(f"Running mixed experiment with ratio {ratio}...")
        mixed_results = experiment_manager.run_experiment("mixed", mix_ratio=ratio)

if __name__ == "__main__":
    main()
