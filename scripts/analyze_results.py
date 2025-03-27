import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results = self._load_results()
        
    def _load_results(self):
        """加载所有实验结果"""
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.results_dir, filename), 'r') as f:
                    data = json.load(f)
                    # 添加实验组标识
                    if 'experiment_type' not in data:
                        data['experiment_type'] = self._infer_experiment_type(filename)
                    results.append(data)
        return results
    
    def _infer_experiment_type(self, filename):
        """从文件名推断实验类型"""
        if 'control' in filename:
            return 'Control'
        elif 'diffusion' in filename:
            return 'Diffusion'
        elif 'fgsm' in filename:
            return 'FGSM'
        elif 'mixed' in filename:
            # 提取混合比例
            if '1_1' in filename:
                return 'Mixed (1:1)'
            elif '1_2' in filename:
                return 'Mixed (1:2)'
            elif '2_1' in filename:
                return 'Mixed (2:1)'
            return 'Mixed'
        return 'Unknown'
    
    def create_comparison_table(self):
        """创建实验结果对比表"""
        data = []
        for result in self.results:
            metrics = result['metrics']
            data.append({
                'Experiment': result['experiment_type'],
                'mAP': metrics['mAP'] * 100,
                'mAP50': metrics['mAP50'] * 100,
                'mAP75': metrics['mAP75'] * 100,
                'Precision': metrics['precision'] * 100,
                'Recall': metrics['recall'] * 100
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('mAP', ascending=False)
    
    def plot_metrics_comparison(self, save_path=None):
        """绘制不同指标的对比图"""
        df = self.create_comparison_table()
        
        # 设置图表样式
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        # 绘制mAP指标对比
        sns.barplot(data=df.melt(id_vars=['Experiment'], 
                                value_vars=['mAP', 'mAP50', 'mAP75'],
                                var_name='Metric', value_name='Value'),
                   x='Value', y='Experiment', hue='Metric', ax=axes[0])
        axes[0].set_title('Mean Average Precision Comparison')
        axes[0].set_xlabel('Value (%)')
        
        # 绘制Precision-Recall对比
        sns.barplot(data=df.melt(id_vars=['Experiment'], 
                                value_vars=['Precision', 'Recall'],
                                var_name='Metric', value_name='Value'),
                   x='Value', y='Experiment', hue='Metric', ax=axes[1])
        axes[1].set_title('Precision-Recall Comparison')
        axes[1].set_xlabel('Value (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig
    
    def generate_analysis_report(self, output_dir):
        """生成完整的分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建比较表
        df = self.create_comparison_table()
        
        # 生成图表
        fig = self.plot_metrics_comparison(
            save_path=os.path.join(output_dir, 'metrics_comparison.png')
        )
        
        # 生成报告文本
        report_path = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 目标检测实验结果分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验结果对比\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## 性能分析\n\n")
            
            # 找出最佳性能
            best_map = df.loc[df['mAP'].idxmax()]
            f.write(f"### 最佳性能\n")
            f.write(f"- 最佳实验组: {best_map['Experiment']}\n")
            f.write(f"- mAP: {best_map['mAP']:.2f}%\n")
            f.write(f"- Precision: {best_map['Precision']:.2f}%\n")
            f.write(f"- Recall: {best_map['Recall']:.2f}%\n\n")
            
            f.write("### 各实验组分析\n\n")
            
            # 对每个实验组进行分析
            for exp_type in df['Experiment'].unique():
                exp_data = df[df['Experiment'] == exp_type].iloc[0]
                f.write(f"#### {exp_type}\n")
                f.write(f"- mAP: {exp_data['mAP']:.2f}%\n")
                f.write(f"- mAP50: {exp_data['mAP50']:.2f}%\n")
                f.write(f"- mAP75: {exp_data['mAP75']:.2f}%\n")
                f.write(f"- Precision: {exp_data['Precision']:.2f}%\n")
                f.write(f"- Recall: {exp_data['Recall']:.2f}%\n")
                
                # 添加性能分析评论
                if exp_type == best_map['Experiment']:
                    f.write("\n该实验组获得了最佳性能，表明此方法最适合当前任务。\n")
                elif exp_data['mAP'] >= df['mAP'].mean():
                    f.write("\n该实验组表现优于平均水平，是一个可靠的方案。\n")
                else:
                    f.write("\n该实验组性能有待提升，可能需要进一步调整参数或方法。\n")
                f.write("\n")
            
            f.write("## 结论和建议\n\n")
            
            # 根据结果提供具体建议
            f.write("### 主要发现\n")
            f.write("1. " + ("扩散模型增强" if "Diffusion" in best_map['Experiment'] else 
                           "FGSM对抗样本" if "FGSM" in best_map['Experiment'] else 
                           "混合增强") + f"方法在mAP指标上表现最佳，达到{best_map['mAP']:.2f}%\n")
            
            # 计算平均性能提升
            control_map = df[df['Experiment'] == 'Control']['mAP'].iloc[0]
            avg_improvement = ((df[df['Experiment'] != 'Control']['mAP'].mean() - control_map) / control_map) * 100
            
            f.write(f"2. 数据增强平均提升了{avg_improvement:.2f}%的mAP\n")
            
            f.write("\n### 建议\n")
            f.write(f"1. 建议主要采用{best_map['Experiment']}方法进行数据增强\n")
            f.write("2. 可以考虑进一步优化以下方面：\n")
            f.write("   - 调整数据增强参数\n")
            f.write("   - 探索不同的混合比例\n")
            f.write("   - 结合其他数据增强技术\n")
            
        print(f"分析报告已生成: {report_path}")
        return report_path

def main():
    # 设置结果目录
    results_dir = 'results'  # 确保这是您的实验结果目录
    output_dir = 'analysis_results'
    
    # 创建分析器
    analyzer = ExperimentAnalyzer(results_dir)
    
    try:
        # 生成分析报告
        report_path = analyzer.generate_analysis_report(output_dir)
        
        print("\n分析完成！您可以在以下位置找到分析结果：")
        print(f"- 分析报告: {report_path}")
        print(f"- 可视化图表: {os.path.join(output_dir, 'metrics_comparison.png')}")
        
    except Exception as e:
        print(f"生成分析报告时出错: {e}")

if __name__ == "__main__":
    main()
