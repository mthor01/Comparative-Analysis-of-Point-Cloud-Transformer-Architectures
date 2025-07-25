#!/usr/bin/env python3
"""
Multi-Model Evaluation Results Comparison (Paper Style)
Paper-ready plots with double column format, thick borders, and tueplots styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from tueplots import bundles, figsizes, fontsizes, cycler
import warnings
warnings.filterwarnings('ignore')

TUEBINGEN_COLORS = ['#A51E37', '#00749E', '#C4071B', '#009A93', '#E2001A', '#4B4B4D']

plt.rcParams.update(fontsizes.icml2022())
plt.rcParams.update(cycler.cycler(color=TUEBINGEN_COLORS))

plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular'
})

PAPER_CONFIG = {
    'axes.linewidth': 2.0,       # Thick borders
    'lines.linewidth': 1.0,      # Thick lines
    'patch.linewidth': 1.0,      # Thick patch borders
    'grid.linewidth': 1.0,       # Thick grid
    'xtick.major.width': 1.5,    # Thick tick marks
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'axes.edgecolor': 'black',   # Black borders
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
}

plt.rcParams.update(PAPER_CONFIG)


class PaperStyleEvaluationComparison:
    """Class for creating paper-style evaluation comparison plots."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize the paper-style evaluation comparison.
        
        Args:
            data_dir: Directory containing the data folders for each model
            output_dir: Directory to save plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_colors = {
            'PtV-1': TUEBINGEN_COLORS[0],      # First Tübingen color
            'PtV-3': TUEBINGEN_COLORS[1],      # Second Tübingen color  
            'OctFormer': TUEBINGEN_COLORS[2],  # Third Tübingen color
            'Superpoint': TUEBINGEN_COLORS[3]  # Fourth Tübingen color
        }
        
        # S3DIS dataset class names
        self.class_names = [
            'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
            'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
        ]
        
        self.evaluation_data = {}
        
    def parse_pointcept_style_log(self, log_file):
        """Parse Pointcept-style evaluation log (PtV-1, PtV-3, OctFormer)."""
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract overall metrics
        overall_pattern = r'Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)'
        overall_match = re.search(overall_pattern, content)
        
        overall_metrics = None
        if overall_match:
            overall_metrics = {
                'mIoU': float(overall_match.group(1)),
                'mAcc': float(overall_match.group(2)),
                'allAcc': float(overall_match.group(3))
            }
        
        # Extract per-class metrics
        class_pattern = r'Class_(\d+) - (\w+) Result: iou/accuracy ([\d.]+)/([\d.]+)'
        class_matches = re.findall(class_pattern, content)
        
        class_results = []
        for match in class_matches:
            class_id, class_name, iou, accuracy = match
            class_results.append({
                'class_id': int(class_id),
                'class_name': class_name,
                'iou': float(iou),
                'accuracy': float(accuracy)
            })
        
        return overall_metrics, class_results
    
    def parse_superpoint_style_log(self, log_file):
        """Parse Superpoint-style evaluation log."""
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Extract overall metrics
        overall_metrics = {}
        class_results = []
        
        for line in lines:
            line = line.strip()
            
            # Parse overall metrics (Superpoint values are in percentage, need to convert to decimal)
            if line.startswith('allAcc'):
                overall_metrics['allAcc'] = float(line.split()[1]) / 100.0
            elif line.startswith('mIoU'):
                # mIoU 64	53.83203887939453
                parts = line.split()
                overall_metrics['mIoU'] = float(parts[-1]) / 100.0
            elif line.startswith('mAcc'):
                # mAcc 64	65.34590911865234
                parts = line.split()
                overall_metrics['mAcc'] = float(parts[-1]) / 100.0
            
            # Parse class results (also convert from percentage to decimal)
            elif line.startswith('Class_'):
                # Class_0 - ceiling IoU 84.99200439453125
                parts = line.split()
                class_id = int(parts[0].split('_')[1])
                class_name = parts[2]
                iou = float(parts[-1]) / 100.0  # Convert percentage to decimal
                
                class_results.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'iou': iou,
                    'accuracy': iou  # Superpoint only provides IoU
                })
        
        return overall_metrics, class_results
    
    def parse_ptv1_style_log(self, log_file):
        """Parse PtV-1 style evaluation log (decimal format)."""
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Extract overall metrics
        overall_metrics = {}
        class_results = []
        
        for line in lines:
            line = line.strip()
            
            # Parse overall metrics (PtV-1 values are already in decimal format)
            if line.startswith('allAcc'):
                overall_metrics['allAcc'] = float(line.split()[1])
            elif line.startswith('mIoU'):
                overall_metrics['mIoU'] = float(line.split()[1])
            elif line.startswith('mAcc'):
                overall_metrics['mAcc'] = float(line.split()[1])
            
            # Parse class results
            elif line.startswith('Class_'):
                # Class_0 - ceiling IoU 0.9298
                parts = line.split()
                class_id = int(parts[0].split('_')[1])
                class_name = parts[2]
                iou = float(parts[-1])  # Already in decimal format
                
                class_results.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'iou': iou,
                    'accuracy': iou  # PtV-1 only provides IoU
                })
        
        return overall_metrics, class_results
    
    def load_all_evaluation_data(self):
        """Load evaluation data for all available models."""
        models = {
            'PtV-1': ('PtV-1/eval.txt', 'ptv1'),
            'PtV-3': ('PtV-3/eval_log.txt', 'pointcept'),
            'OctFormer': ('OctFormer/eval-log.txt', 'pointcept'),
            'Superpoint': ('Superpoint/eval.txt', 'superpoint')
        }
        
        for model_name, (log_path, log_format) in models.items():
            full_path = self.data_dir / log_path
            if not full_path.exists():
                continue
            
            try:
                if log_format == 'pointcept':
                    overall_metrics, class_results = self.parse_pointcept_style_log(full_path)
                elif log_format == 'ptv1':
                    overall_metrics, class_results = self.parse_ptv1_style_log(full_path)
                else:  # superpoint
                    overall_metrics, class_results = self.parse_superpoint_style_log(full_path)
                
                self.evaluation_data[model_name] = {
                    'overall': overall_metrics,
                    'classes': class_results
                }
                
            except Exception as e:
                pass
        
    def plot_overall_metrics_comparison(self):
        """Plot overall metrics comparison - paper style."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(figsizes.icml2022_half()['figure.figsize'][0] * 1.5, figsizes.icml2022_half()['figure.figsize'][1]))
        
        models = []
        mious = []
        maccs = []
        allaccs = []
        colors = []
        
        for model_name, data in self.evaluation_data.items():
            if data['overall']:
                models.append(model_name)
                mious.append(data['overall'].get('mIoU', 0))
                maccs.append(data['overall'].get('mAcc', 0))
                allaccs.append(data['overall'].get('allAcc', 0))
                colors.append(self.model_colors[model_name])
        
        # mIoU comparison
        bars1 = ax1.bar(models, mious, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=2, width=0.6)
        for bar, value in zip(bars1, mious):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=8)
        
        ax1.set_title("Mean IoU")
        ax1.set_ylabel("mIoU Score")
        ax1.set_ylim(0, max(mious) * 1.15 if mious else 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # mAcc comparison
        bars2 = ax2.bar(models, maccs, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=2, width=0.6)
        for bar, value in zip(bars2, maccs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=8)
        
        ax2.set_title("Mean Accuracy")
        ax2.set_ylabel("mAcc Score")
        ax2.set_ylim(0, max(maccs) * 1.15 if maccs else 1)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Overall Accuracy comparison
        bars3 = ax3.bar(models, allaccs, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=2, width=0.6)
        for bar, value in zip(bars3, allaccs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=8)
        
        ax3.set_title("Overall Accuracy")
        ax3.set_ylabel("Overall Accuracy")
        ax3.set_ylim(0, max(allaccs) * 1.15 if allaccs else 1)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "paper_multi_model_overall_comparison.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    
    def plot_architecture_performance_heatmap(self):
        """Create a heatmap showing performance across classes and models - paper style."""
        fig, ax = plt.subplots(figsize=(figsizes.icml2022_half()['figure.figsize'][0] * 1.5, figsizes.icml2022_half()['figure.figsize'][1]))
        
        # Prepare data matrix
        models = list(self.evaluation_data.keys())
        n_classes = len(self.class_names)
        
        iou_matrix = np.zeros((len(models), n_classes))
        
        for i, model_name in enumerate(models):
            class_data = self.evaluation_data[model_name]['classes']
            for class_result in class_data:
                class_id = class_result['class_id']
                if class_id < n_classes:
                    iou_matrix[i, class_id] = class_result['iou']
        
        # Create heatmap
        im = ax.imshow(iou_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(n_classes):
                value = iou_matrix[i, j]
                color = 'white' if value < 0.5 else 'black'
                text = ax.text(j, i, f'{value:.2f}',
                              ha="center", va="center", color=color, 
                              fontsize=7)
        
        ax.set_title("Performance Heatmap - IoU Scores")
        ax.set_xlabel("Object Classes")
        ax.set_ylabel("Architectures")
        
        # Add colorbar with thick border
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('IoU Score')
        cbar.outline.set_linewidth(2)
        
        # Use subplots_adjust instead of tight_layout for colorbar compatibility
        plt.subplots_adjust(bottom=0.15, right=0.85, top=0.9)
        plt.savefig(self.output_dir / "paper_multi_model_performance_heatmap.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

    

def main():
    """Main function to run the paper-style evaluation comparison script."""
    # Configuration
    data_dir = "Data"
    output_dir = "output"
    
    # Create comparison instance
    comparator = PaperStyleEvaluationComparison(data_dir, output_dir)
    
    # Load data and generate plots directly
    comparator.load_all_evaluation_data()
    
    if not comparator.evaluation_data:
        return
    
    comparator.plot_overall_metrics_comparison()
    comparator.plot_architecture_performance_heatmap()


if __name__ == "__main__":
    main()
