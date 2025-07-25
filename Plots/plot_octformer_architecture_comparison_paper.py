#!/usr/bin/env python3
"""
OctFormer Architecture Comparison (Paper Style)
Comparing big and small OctFormer architectures for training convergence.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes, fontsizes, cycler

# Fallback Tübingen-style colors
TUEBINGEN_COLORS = ['#A51E37', '#00749E', '#C4071B', '#009A93', '#E2001A', '#4B4B4D']

# Configure matplotlib for paper style
plt.rcParams.update(fontsizes.icml2022())
plt.rcParams.update(cycler.cycler(color=TUEBINGEN_COLORS))
plt.rcParams.update({
    'text.usetex': False,
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.0,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
})


class OctFormerComparison:
    """Class for comparing OctFormer big vs small architecture training."""
    
    def __init__(self):
        self.data_dir = Path("Data")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.colors = {'Big': TUEBINGEN_COLORS[0], 'Small': TUEBINGEN_COLORS[1]}
    
    def load_data(self, filename):
        """Load and clean data for OctFormer architecture."""
        df = pd.read_csv(self.data_dir / "OctFormer" / filename)
        
        # Find required columns
        miou_col = next((col for col in df.columns 
                        if 'val/mIoU' in str(col) and 'MIN' not in str(col) and 'MAX' not in str(col)), None)
        epoch_col = next((col for col in df.columns if 'Epoch' in str(col)), None)
        
        if not miou_col or not epoch_col:
            raise ValueError(f"Could not find required columns in {filename}")
        
        # Return filtered data up to 32 epochs
        return pd.DataFrame({
            'Epoch': df[epoch_col],
            'mIoU': df[miou_col]
        }).query('Epoch <= 32')
    
    def plot_comparison(self):
        """Plot training convergence comparison between big and small OctFormer."""
        # Load data
        big_data = self.load_data("val-mIoU.csv")
        small_data = self.load_data("small_val-mIoU.csv")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsizes.icml2022_half()['figure.figsize'])
        
        # Plot both architectures
        ax.plot(big_data['Epoch'], big_data['mIoU'], 
               label='OctFormer (Big)', color=self.colors['Big'],
               linewidth=3, alpha=0.8, marker='o', markersize=4)
        
        ax.plot(small_data['Epoch'], small_data['mIoU'], 
               label='OctFormer (Small)', color=self.colors['Small'],
               linewidth=3, alpha=0.8, marker='s', markersize=4)
        
        # Configure plot
        ax.set(title="OctFormer Architecture Comparison", xlabel="Epochs", ylabel="Mean IoU", xlim=(1, 32))
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "paper_octformer_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the OctFormer architecture comparison."""
    comparator = OctFormerComparison()
    comparator.plot_comparison()


if __name__ == "__main__":
    main()
