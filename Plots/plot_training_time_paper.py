#!/usr/bin/env python3
"""
Training Time Comparison Script (Paper Style)
Paper-ready plots with double column format, thick borders, and tueplots styling.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tueplots import bundles, figsizes, fontsizes, cycler
import warnings
warnings.filterwarnings('ignore')

TUEBINGEN_COLORS = ['#A51E37', '#00749E', '#C4071B', '#009A93', '#E2001A', '#4B4B4D']

# Configure matplotlib for paper style with tueplots
plt.rcParams.update({
    **bundles.icml2022(),
    **figsizes.icml2022_half(),
    **fontsizes.icml2022(),
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
plt.rcParams.update(cycler.cycler(color=TUEBINGEN_COLORS))


class PaperStyleTrainingTimeAnalyzer:
    """Class for analyzing and plotting training times in paper style."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize the paper-style training time analyzer.
        
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
        
        self.training_times = {}
    
    def parse_time_string(self, time_str):
        """Parse a time string in format 'Xh Ym Zs' and return total seconds."""
        if not time_str.strip():
            return 0
        
        total_seconds = 0
        for pattern, multiplier in [(r'(\d+)h', 3600), (r'(\d+)m', 60), (r'(\d+)s', 1)]:
            match = re.search(pattern, time_str)
            if match:
                total_seconds += int(match.group(1)) * multiplier
        
        return total_seconds
    
    def format_time_display(self, total_seconds):
        """Format seconds back to readable time format for display."""
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def load_training_times(self):
        """Load and parse training times for all models."""
        for model_name in ['PtV-1', 'PtV-3', 'OctFormer', 'Superpoint']:
            traintime_file = self.data_dir / model_name / 'traintime.txt'
            
            if traintime_file.exists():
                try:
                    total_seconds = sum(
                        self.parse_time_string(line.strip())
                        for line in traintime_file.read_text().splitlines()
                        if line.strip()
                    )
                    self.training_times[model_name] = total_seconds
                except Exception:
                    pass
        
    def plot_training_time_comparison(self):
        """Create a bar plot comparing total training times - paper style."""
        if not self.training_times:
            return
        
        fig, ax = plt.subplots(figsize=figsizes.icml2022_half()['figure.figsize'])
        
        # Prepare data
        models = list(self.training_times.keys())
        training_hours = [seconds / 3600.0 for seconds in self.training_times.values()]
        colors = [self.model_colors[model] for model in models]
        
        # Create bars
        bars = ax.bar(models, training_hours, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on bars
        for bar, model in zip(bars, models):
            height = bar.get_height()
            total_seconds = self.training_times[model]
            ax.text(bar.get_x() + bar.get_width()/2., height + max(training_hours) * 0.01,
                   f'{height:.1f}h\n({self.format_time_display(total_seconds)})', 
                   ha='center', va='bottom', fontsize=8)
        
        # Configure plot
        ax.set(title="Total Training Time Comparison", 
               xlabel="Point Cloud Transformer Architectures",
               ylabel="Total Training Time (Hours)",
               ylim=(0, max(training_hours) * 1.25))
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "paper_training_time_comparison.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
def main():
    """Main function to run the paper-style training time analysis."""
    # Configuration
    data_dir = "Data"
    output_dir = "output"
    
    # Create analyzer instance
    analyzer = PaperStyleTrainingTimeAnalyzer(data_dir, output_dir)
    
    # Load data and generate plot directly
    analyzer.load_training_times()
    
    if not analyzer.training_times:
        return
    
    analyzer.plot_training_time_comparison()


if __name__ == "__main__":
    main()
