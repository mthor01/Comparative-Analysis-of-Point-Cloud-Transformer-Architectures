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

# Try to import palettes, use fallback if not available
try:
    from tueplots import palettes
    TUEBINGEN_COLORS = palettes.tue_plot
except ImportError:
    try:
        from tueplots.constants import palettes
        TUEBINGEN_COLORS = palettes.tue_plot
    except ImportError:
        # Fallback Tübingen-style colors if palettes not available
        TUEBINGEN_COLORS = ['#A51E37', '#00749E', '#C4071B', '#009A93', '#E2001A', '#4B4B4D']

# Configure matplotlib for paper style with tueplots
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_half())
plt.rcParams.update(fontsizes.icml2022())

# Use University of Tübingen-style colors (fallback implementation)
plt.rcParams.update(cycler.cycler(color=TUEBINGEN_COLORS))

# Disable LaTeX to avoid system dependency issues
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular'
})

# Additional paper-style configuration
PAPER_CONFIG = {
    'axes.linewidth': 2.0,       # Thick borders
    'lines.linewidth': 2.0,      # Thick lines
    'patch.linewidth': 2.0,      # Thick patch borders
    'grid.linewidth': 1.5,       # Thick grid
    'xtick.major.width': 1.5,    # Thick tick marks
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'axes.edgecolor': 'black',   # Black borders
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    # Serif fonts for academic papers
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
}

plt.rcParams.update(PAPER_CONFIG)


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
            print(f"  Original columns: {list(df.columns)}")
            
            # Clean column names
            df.columns = [self.clean_column_name(col) for col in df.columns]
            print(f"  Cleaned columns: {list(df.columns)}")
            
            # Remove duplicate columns (keep first occurrence)
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"  After removing duplicates: {list(df.columns)}")
            
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
                    print(f"  Converting Superpoint mIoU from percentage to decimal")
                    df['mIoU'] = df['mIoU'] / 100.0
            
            print(f"  Loaded {len(df)} data points")
            print(f"  Available columns: {list(df.columns)}")
            if 'mIoU' in df.columns:
                miou_min = df['mIoU'].min()
                miou_max = df['mIoU'].max()
                print(f"  mIoU range: {miou_min:.4f} - {miou_max:.4f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
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
                print(f"Training data not found for {model_name}: {full_path}")
                continue
            
            print(f"Loading training data for {model_name}...")
            model_data = self.load_model_data(model_name, full_path)
            
            if model_data is not None:
                self.models_data[model_name] = model_data
        
        print(f"Successfully loaded training data for {len(self.models_data)} models")
    
    def plot_training_convergence_comparison(self):
        """Plot training convergence for all models - paper style."""
        if not self.models_data:
            print("No training data available!")
            return
        
        # Create two separate plots due to different scales
        
        # 1. Pointcept models (epoch-based)
        pointcept_models = {k: v for k, v in self.models_data.items() if k in ['PtV-1', 'PtV-3', 'OctFormer']}
        
        if pointcept_models:
            # Use wider figure for proper double column layout
            fig_width = figsizes.icml2022_half()['figure.figsize'][0] * 1.8
            fig_height = figsizes.icml2022_half()['figure.figsize'][1]
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            for model_name, data in pointcept_models.items():
                step_col = 'step' if 'step' in data.columns else data.columns[0]
                
                # Plot mIoU if available
                if 'mIoU' in data.columns:
                    ax.plot(data[step_col], data['mIoU'], 
                           label=f'{model_name}', 
                           color=self.model_colors[model_name],
                           linewidth=2, alpha=0.8)
            
            ax.set_title("Training Convergence - Pointcept Models")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Mean IoU")
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            ax.grid(True, alpha=0.3)
            
            # Ensure all spines are visible
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "paper_pointcept_training_convergence.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Superpoint model (step-based)
        if 'Superpoint' in self.models_data:
            # Use wider figure for proper double column layout
            fig_width = figsizes.icml2022_half()['figure.figsize'][0] * 1.8
            fig_height = figsizes.icml2022_half()['figure.figsize'][1]
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            data = self.models_data['Superpoint']
            step_col = 'step' if 'step' in data.columns else data.columns[0]
            
            if 'mIoU' in data.columns:
                ax.plot(data[step_col], data['mIoU'], 
                       label='Superpoint', 
                       color=self.model_colors['Superpoint'],
                       linewidth=2, alpha=0.8)
            
            ax.set_title("Training Convergence - Superpoint Model")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Mean IoU")
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            ax.grid(True, alpha=0.3)
            
            # Ensure all spines are visible
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "paper_superpoint_training_convergence.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_final_performance_summary(self):
        """Create a final performance comparison - paper style."""
        if not self.models_data:
            print("No training data available!")
            return
        
        # Use wider figure for proper double column layout
        fig_width = figsizes.icml2022_half()['figure.figsize'][0] * 2.2
        fig_height = figsizes.icml2022_half()['figure.figsize'][1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        
        # Extract final performance metrics
        final_mious = []
        final_maccs = []
        models = []
        colors = []
        
        for model_name, data in self.models_data.items():
            models.append(model_name)
            colors.append(self.model_colors[model_name])
            
            # Get final mIoU
            if 'mIoU' in data.columns:
                final_mious.append(data['mIoU'].iloc[-1])
            else:
                final_mious.append(0)
            
            # Get final mAcc
            if 'mAcc' in data.columns:
                final_maccs.append(data['mAcc'].iloc[-1])
            else:
                final_maccs.append(0)
        
        # Final mIoU comparison
        bars1 = ax1.bar(models, final_mious, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=2, width=0.6)
        for bar, value in zip(bars1, final_mious):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=8)
        
        ax1.set_title("Final mIoU Performance")
        ax1.set_ylabel("Mean IoU")
        ax1.set_ylim(0, max(final_mious) * 1.15 if final_mious else 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Ensure right axis is visible
        ax1.spines['right'].set_visible(True)
        ax1.spines['top'].set_visible(True)
        
        # Final mAcc comparison (if available)
        if any(final_maccs):
            bars2 = ax2.bar(models, final_maccs, color=colors, alpha=0.7, 
                           edgecolor='black', linewidth=2, width=0.6)
            for bar, value in zip(bars2, final_maccs):
                height = bar.get_height()
                if value > 0:  # Only show label if value exists
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{value:.3f}', ha='center', va='bottom', 
                            fontsize=8)
            
            ax2.set_title("Final mAcc Performance")
            ax2.set_ylabel("Mean Accuracy")
            ax2.set_ylim(0, max(final_maccs) * 1.15 if max(final_maccs) > 0 else 1)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='x', rotation=45)
        else:
            # If no mAcc data, show final IoU in a different way
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'mAcc data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12)
        
        # Ensure right axis is visible for both subplots
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(True)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "paper_final_training_performance.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_all_paper_training_plots(self):
        """Generate all paper-style training comparison plots."""
        print("=" * 50)
        print("PAPER-STYLE TRAINING ANALYSIS")
        print("=" * 50)
        
        self.load_all_training_data()
        
        if not self.models_data:
            print("No training data found!")
            return
        
        print(f"\nGenerating paper-style training plots for {len(self.models_data)} models...")
        
        self.plot_training_convergence_comparison()
        print("✓ Paper-style training convergence plots saved")
        
        self.plot_final_performance_summary()
        print("✓ Paper-style final performance summary saved")
        
        print(f"\nAll paper-style training plots saved to: {self.output_dir}")
        print("\nGenerated plots:")
        print("  - paper_pointcept_training_convergence.pdf")
        print("  - paper_superpoint_training_convergence.pdf")
        print("  - paper_final_training_performance.pdf")


def main():
    """Main function to run the paper-style training analysis."""
    # Configuration
    data_dir = "Data"
    output_dir = "output"
    
    # Create plotter instance
    plotter = PaperStyleTrainingPlotter(data_dir, output_dir)
    
    # Generate all training plots
    plotter.generate_all_paper_training_plots()
    
    print("\nPaper-style multi-model training analysis complete!")


if __name__ == "__main__":
    main()
