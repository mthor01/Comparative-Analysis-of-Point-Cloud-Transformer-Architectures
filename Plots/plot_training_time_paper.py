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
        """
        Parse a time string in format 'Xh Ym Zs' and return total seconds.
        
        Args:
            time_str: Time string like '10h 10m 24s' or '1h 21m'
            
        Returns:
            Total seconds as integer
        """
        time_str = time_str.strip()
        if not time_str:
            return 0
        
        total_seconds = 0
        
        # Extract hours
        hours_match = re.search(r'(\d+)h', time_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600
        
        # Extract minutes
        minutes_match = re.search(r'(\d+)m', time_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60
        
        # Extract seconds
        seconds_match = re.search(r'(\d+)s', time_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))
        
        return total_seconds
    
    def seconds_to_hours(self, seconds):
        """Convert seconds to hours with decimal precision."""
        return seconds / 3600.0
    
    def format_time_display(self, total_seconds):
        """Format seconds back to readable time format for display."""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def load_training_times(self):
        """Load and parse training times for all models."""
        models = ['PtV-1', 'PtV-3', 'OctFormer', 'Superpoint']
        
        for model_name in models:
            traintime_file = self.data_dir / model_name / 'traintime.txt'
            
            if not traintime_file.exists():
                print(f"Training time file not found for {model_name}: {traintime_file}")
                continue
            
            print(f"Loading training times for {model_name}...")
            
            try:
                with open(traintime_file, 'r') as f:
                    lines = f.readlines()
                
                total_seconds = 0
                session_count = 0
                
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        session_seconds = self.parse_time_string(line)
                        total_seconds += session_seconds
                        session_count += 1
                
                self.training_times[model_name] = total_seconds
                print(f"  Total training time: {self.format_time_display(total_seconds)} ({self.seconds_to_hours(total_seconds):.1f} hours)")
                
            except Exception as e:
                print(f"Error parsing training times for {model_name}: {e}")
        
        print(f"\nSuccessfully loaded training times for {len(self.training_times)} models")
    
    def plot_training_time_comparison(self):
        """Create a bar plot comparing total training times - paper style."""
        if not self.training_times:
            print("No training time data available!")
            return
        
        fig, ax = plt.subplots(figsize=figsizes.icml2022_half()['figure.figsize'])
        
        # Prepare data
        models = list(self.training_times.keys())
        training_hours = [self.seconds_to_hours(self.training_times[model]) for model in models]
        colors = [self.model_colors[model] for model in models]
        
        # Create bars
        bars = ax.bar(models, training_hours, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on bars
        for bar, model in zip(bars, models):
            height = bar.get_height()
            total_seconds = self.training_times[model]
            
            # Show hours and formatted time
            ax.text(bar.get_x() + bar.get_width()/2., height + max(training_hours) * 0.01,
                   f'{height:.1f}h\n({self.format_time_display(total_seconds)})', 
                   ha='center', va='bottom', 
                   fontsize=8)
        
        # Formatting
        ax.set_title("Total Training Time Comparison")
        ax.set_xlabel("Point Cloud Transformer Architectures")
        ax.set_ylabel("Total Training Time (Hours)")
        ax.set_ylim(0, max(training_hours) * 1.25)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "paper_training_time_comparison.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_paper_training_time_report(self):
        """Generate paper-style training time analysis."""
        print("=" * 50)
        print("PAPER-STYLE TRAINING TIME ANALYSIS")
        print("=" * 50)
        
        self.load_training_times()
        
        if not self.training_times:
            print("No training time data found!")
            return
        
        print(f"\nGenerating paper-style training time comparison plot...")
        self.plot_training_time_comparison()
        print("✓ Paper-style training time comparison plot saved")
        
        print(f"\nPaper-style training time analysis report saved to: {self.output_dir}")
        print("\nGenerated plots:")
        print("  - paper_training_time_comparison.pdf")
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        
        sorted_models = sorted(self.training_times.items(), key=lambda x: x[1], reverse=True)
        
        print("\nModels ranked by training time (longest to shortest):")
        for i, (model, seconds) in enumerate(sorted_models, 1):
            hours = self.seconds_to_hours(seconds)
            formatted_time = self.format_time_display(seconds)
            print(f"{i}. {model}: {hours:.1f} hours ({formatted_time})")


def main():
    """Main function to run the paper-style training time analysis."""
    # Configuration
    data_dir = "Data"
    output_dir = "output"
    
    # Create analyzer instance
    analyzer = PaperStyleTrainingTimeAnalyzer(data_dir, output_dir)
    
    # Generate training time analysis
    analyzer.generate_paper_training_time_report()
    
    print("\nPaper-style training time analysis complete!")


if __name__ == "__main__":
    main()
