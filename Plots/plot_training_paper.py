#!/usr/bin/env python3
"""
Multi-Model Point Cloud Transformer Training Analysis (Paper Style)
Paper-ready plots with double column format, thick borders, and tueplots styling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes, fontsizes, cycler
import warnings
warnings.filterwarnings('ignore')

TUEBINGEN_COLORS = ['#A51E37', '#00749E', '#C4071B', '#009A93', '#E2001A', '#4B4B4D']

# Configure matplotlib for paper style with tueplots
plt.rcParams.update(fontsizes.icml2022())
plt.rcParams.update(cycler.cycler(color=TUEBINGEN_COLORS))
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular',
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.0,
    'patch.linewidth': 2.0,
    'grid.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
})


class PaperStyleTrainingPlotter:
    """Class for generating paper-style multi-model training analysis plots."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize the paper-style training plotter.
        
        Args:
            data_dir: Directory containing the data folders for each model
            output_dir: Directory to save plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(exist_ok=True)
        
        # University of Tübingen color scheme (using fallback colors)
        self.model_colors = {
            'PtV-1': TUEBINGEN_COLORS[0],      # First Tübingen color
            'PtV-3': TUEBINGEN_COLORS[1],      # Second Tübingen color  
            'OctFormer': TUEBINGEN_COLORS[2],  # Third Tübingen color
            'Superpoint': TUEBINGEN_COLORS[3]  # Fourth Tübingen color
        }
        
        self.models_data = {}
    
    def _get_step_column(self, data):
        """Get the appropriate step/epoch column from data."""
        step_columns = ['step', 'Step', 'epoch', 'Epoch']
        for col in step_columns:
            if col in data.columns:
                return col
        return data.columns[0]
    
    def _create_paper_plot(self, title, xlabel):
        """Create a standardized paper-style plot."""
        fig_width = figsizes.icml2022_half()['figure.figsize'][0] * 1.8
        fig_height = figsizes.icml2022_half()['figure.figsize'][1]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean IoU")
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        
        return fig, ax
    
    def _finalize_plot(self, ax, filename):
        """Finalize and save a plot."""
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def clean_column_name(self, col_name):
        """Clean column names by removing unwanted characters."""
        if isinstance(col_name, str):
            # Handle complex column names from different models
            if 'val/mIoU' in col_name and not ('MIN' in col_name or 'MAX' in col_name):
                return 'mIoU'
            elif 'val/miou' in col_name and not ('MIN' in col_name or 'MAX' in col_name):
                return 'mIoU'
            elif 'val/mAcc' in col_name and not ('MIN' in col_name or 'MAX' in col_name):
                return 'mAcc'
            elif 'val/macc' in col_name and not ('MIN' in col_name or 'MAX' in col_name):
                return 'mAcc'
            elif 'Epoch' in col_name:
                return 'step'
            elif 'trainer/global_step' in col_name:
                return 'step'
            else:
                return col_name.replace('Unnamed: 0', 'step').replace('val-', '').strip()
        return col_name
    
    def load_model_data(self, model_name, csv_file):
        """Load and clean data for a specific model."""
        try:
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            # Clean column names
            df.columns = [self.clean_column_name(col) for col in df.columns]
            
            # Remove duplicate columns (keep first occurrence)
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Handle different step column names
            step_columns = ['step', 'Step', 'epoch', 'Epoch']
            step_col = None
            for col in step_columns:
                if col in df.columns:
                    step_col = col
                    break
            
            if step_col is None:
                df['step'] = df.index + 1  # Use 1-based indexing for epochs
                step_col = 'step'
            
            # Ensure step column is numeric
            df[step_col] = pd.to_numeric(df[step_col], errors='coerce')
            df = df.dropna(subset=[step_col])
            
            # Check if Superpoint data needs to be converted from percentage
            if model_name == 'Superpoint' and 'mIoU' in df.columns:
                max_miou = df['mIoU'].max()
                if max_miou > 1.0:  # Likely in percentage
                    df['mIoU'] = df['mIoU'] / 100.0
            
            return df
            
        except Exception as e:
            return None
    
    def load_all_training_data(self):
        """Load training data for all available models."""
        models_info = {
            'PtV-1': 'PtV-1/val-mIoU.csv',
            'PtV-3': 'PtV-3/val-mIoU.csv', 
            'OctFormer': 'OctFormer/val-mIoU.csv',
            'Superpoint': 'Superpoint/val-mIoU.csv'
        }
        
        for model_name, csv_path in models_info.items():
            full_path = self.data_dir / csv_path
            if not full_path.exists():
                continue
            
            model_data = self.load_model_data(model_name, full_path)
            
            if model_data is not None:
                self.models_data[model_name] = model_data
        
    def plot_training_convergence_comparison(self):
        """Plot training convergence for all models - paper style."""
        if not self.models_data:
            return
        
        # 1. Pointcept models (epoch-based)
        pointcept_models = {k: v for k, v in self.models_data.items() if k in ['PtV-1', 'PtV-3', 'OctFormer']}
        
        if pointcept_models:
            fig, ax = self._create_paper_plot("Training Convergence - Pointcept Models", "Epochs")
            
            for model_name, data in pointcept_models.items():
                step_col = self._get_step_column(data)
                if 'mIoU' in data.columns:
                    ax.plot(data[step_col], data['mIoU'], 
                           label=model_name, 
                           color=self.model_colors[model_name],
                           linewidth=2, alpha=0.8)
            
            self._finalize_plot(ax, "paper_pointcept_training_convergence.pdf")
        
        # 2. Superpoint model (step-based)
        if 'Superpoint' in self.models_data:
            fig, ax = self._create_paper_plot("Training Convergence - Superpoint Model", "Training Steps")
            
            data = self.models_data['Superpoint']
            step_col = self._get_step_column(data)
            
            if 'mIoU' in data.columns:
                ax.plot(data[step_col], data['mIoU'], 
                       label='Superpoint', 
                       color=self.model_colors['Superpoint'],
                       linewidth=2, alpha=0.8)
            
            self._finalize_plot(ax, "paper_superpoint_training_convergence.pdf")
    
    
def main():
    """Main function to run the paper-style training analysis."""
    # Configuration
    data_dir = "Data"
    output_dir = "output"
    
    # Create plotter instance
    plotter = PaperStyleTrainingPlotter(data_dir, output_dir)
    
    # Load data and generate plots directly
    plotter.load_all_training_data()
    
    if not plotter.models_data:
        return
    
    plotter.plot_training_convergence_comparison()


if __name__ == "__main__":
    main()
