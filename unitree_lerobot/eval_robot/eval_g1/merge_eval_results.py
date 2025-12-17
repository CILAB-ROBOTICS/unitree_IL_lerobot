"""
Merge and visualize evaluation results from multiple checkpoint evaluation files
Combines results from different runs and creates comparison plots across architectures and tasks

Usage:
    Basic usage (3-part format):
    python unitree_lerobot/eval_robot/eval_g1/merge_eval_results.py \
        --tasks towel_weak_train \
        --configs \
            "path1.json:act:ACT-Normal" \
            "path2.json:act:ACT-NoTactile" \
            "path1.json:mlp:MLP-Baseline"

    Extended format with custom colors, linestyles, and markers (6-part format):
    python unitree_lerobot/eval_robot/eval_g1/merge_eval_results.py \
        --tasks towel_weak_train towel_strong_train \
        --configs \
            "file1.json:act:ACT-Train:#007737:-:o" \
            "file2.json:act:ACT-Test:#DD5E1D:--:s" \
        --output_dir merged_results

Arguments:
    --tasks: Task names to analyze (required, e.g., towel_weak_train towel_strong_train)
    --configs: Model configurations in format "file:arch:label" or "file:arch:label:color:linestyle:marker" (required)
    --output_dir: Output directory for merged results (default: merged_results)

Features:
    - Task-specific analysis with custom architecture labeling
    - Flexible configuration formats:
        * Basic: file:architecture:label
        * Extended: file:architecture:label:color:linestyle:marker
    - Creates comparison plots for MSE, L1, and Huber losses
    - Handles different checkpoint steps automatically
    - Generates publication-ready plots with error bars
    - Custom styling: colors (hex/name), linestyles (-, --, -., :), markers (o, s, ^, etc.)
    - Supports glob patterns in file paths (e.g., "run_*/checkpoint_evaluation_stats.json")
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import os
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    file_path: str
    architecture: str  # Original architecture name in JSON
    custom_label: str  # Custom label for plotting
    color: Optional[str] = None  # Custom color (hex or name)
    linestyle: Optional[str] = None  # Line style ('-', '--', '-.', ':')
    marker: Optional[str] = None  # Marker style ('o', 's', '^', etc.)


class EvalResultsMerger:
    """Class to merge and visualize evaluation results from multiple files"""
    
    def __init__(self, tasks: List[str], configs: List[str], output_dir: str = "merged_results"):
        self.tasks = tasks
        
        # Generate timestamp with process ID for concurrent execution
        import os
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        
        # Create unique output directory with timestamp
        self.unique_output_dir = Path(output_dir) / f"merge_{self.timestamp}"
        self.unique_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.unique_output_dir  # Keep compatibility with existing code
        
        # Parse configs: "file:arch:label" format
        self.model_configs = self._parse_configs(configs)
        
        # Load all data
        self.data = self._load_all_data()
        
        # Extract checkpoint steps from all relevant data
        self.checkpoint_steps = self._extract_checkpoint_steps()
        
        print(f"Target tasks: {self.tasks}")
        print(f"Output directory: {self.unique_output_dir}")
        print(f"Model configurations:")
        for config in self.model_configs:
            print(f"  {config.custom_label}: {config.file_path} -> {config.architecture}")
        print(f"Found checkpoint steps: {self.checkpoint_steps}")
    
    def _parse_configs(self, configs: List[str]) -> List[ModelConfig]:
        """Parse configuration strings in format 'file:arch:label' or 'file:arch:label:color:linestyle:marker'"""
        model_configs = []
        for config_str in configs:
            try:
                parts = config_str.split(':')
                if len(parts) == 3:
                    # Basic format: file:arch:label
                    file_path, architecture, custom_label = parts
                    model_configs.append(ModelConfig(
                        file_path=file_path.strip(),
                        architecture=architecture.strip(),
                        custom_label=custom_label.strip()
                    ))
                elif len(parts) == 6:
                    # Extended format: file:arch:label:color:linestyle:marker
                    file_path, architecture, custom_label, color, linestyle, marker = parts
                    model_configs.append(ModelConfig(
                        file_path=file_path.strip(),
                        architecture=architecture.strip(),
                        custom_label=custom_label.strip(),
                        color=color.strip() if color.strip() else None,
                        linestyle=linestyle.strip() if linestyle.strip() else None,
                        marker=marker.strip() if marker.strip() else None
                    ))
                else:
                    raise ValueError(f"Config must be in format 'file:arch:label' or 'file:arch:label:color:linestyle:marker', got {len(parts)} parts")
            except Exception as e:
                raise ValueError(f"Invalid config format '{config_str}': {e}")
        
        return model_configs
    
    def _resolve_latest_file(self, pattern: str) -> Optional[str]:
        """Resolve a file path pattern (supports glob like run_*/file.json) to the latest file by mtime.

        If the pattern contains wildcards, expand and pick the most recent. If no wildcard, return as-is.
        """
        # Detect simple glob characters
        if any(ch in pattern for ch in ['*', '?', '[']):
            matches = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                print(f"✗ No files matched pattern: {pattern}")
                return None
            chosen = str(matches[0])
            # Log a short summary
            print(f"Pattern '{pattern}' matched {len(matches)} file(s). Using latest: {chosen}")
            return chosen
        # No wildcard, return as-is
        return pattern

    def _load_all_data(self) -> Dict[str, Dict[str, Any]]:
        """Load data from all input files, indexed by resolved file path (after wildcard expansion)."""
        data = {}
        original_to_resolved: Dict[str, str] = {}
        unique_files = set(config.file_path for config in self.model_configs)

        for file_pattern in unique_files:
            resolved = self._resolve_latest_file(file_pattern)
            if resolved is None:
                continue
            original_to_resolved[file_pattern] = resolved
            try:
                with open(resolved, 'r') as f:
                    file_data = json.load(f)
                    data[resolved] = file_data
                    print(f"✓ Loaded {resolved}")
            except Exception as e:
                print(f"✗ Failed to load {resolved}: {e}")
                raise

        # Update model configs to point to resolved paths so downstream lookups work
        for cfg in self.model_configs:
            if cfg.file_path in original_to_resolved:
                cfg.file_path = original_to_resolved[cfg.file_path]

        return data
    
    def _extract_checkpoint_steps(self) -> List[str]:
        """Extract all checkpoint steps from relevant task data"""
        steps = set()
        
        for config in self.model_configs:
            file_data = self.data[config.file_path]
            
            if config.architecture in file_data:
                arch_data = file_data[config.architecture]
                
                for task in self.tasks:
                    if task in arch_data and isinstance(arch_data[task], dict):
                        steps.update(arch_data[task].keys())
        
        return sorted(list(steps))
    
    def create_task_comparison_plots(self):
        """Create comparison plots for each specified task"""
        metrics = ['eval_mse', 'eval_l1', 'eval_huber']
        metric_names = {'eval_mse': 'MSE Loss', 'eval_l1': 'L1 Loss', 'eval_huber': 'Huber Loss'}
        
        # Create plots for each task
        for task in self.tasks:
            print(f"Creating plots for task: {task}")
            
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Convert checkpoint steps to integers for plotting
                x_steps = [int(step) for step in self.checkpoint_steps]
                
                # Default colors, linestyles and markers for different models
                default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                default_markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*']
                default_linestyle = '-'
                
                plot_idx = 0
                
                # Plot each configured model
                for config in self.model_configs:
                    file_data = self.data[config.file_path]
                    
                    # Check if this configuration has data for current task
                    if config.architecture not in file_data:
                        print(f"  Warning: Architecture '{config.architecture}' not found in {config.file_path}")
                        continue
                    
                    arch_data = file_data[config.architecture]
                    if task not in arch_data:
                        print(f"  Warning: Task '{task}' not found for {config.custom_label}")
                        continue
                    
                    task_data = arch_data[task]
                    
                    means = []
                    stds = []
                    
                    for step in self.checkpoint_steps:
                        if step in task_data:
                            means.append(task_data[step][f'{metric}_mean'])
                            stds.append(task_data[step][f'{metric}_std'])
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    # Skip if no valid data
                    if np.all(np.isnan(means)):
                        print(f"  Warning: No valid data for {config.custom_label} on {task}")
                        continue
                    
                    # Use custom or default styles
                    color = config.color if config.color else default_colors[plot_idx % len(default_colors)]
                    marker = config.marker if config.marker else default_markers[plot_idx % len(default_markers)]
                    linestyle = config.linestyle if config.linestyle else default_linestyle
                    
                    # Main line plot
                    ax.plot(x_steps, means, 
                           label=config.custom_label, 
                           color=color, 
                           marker=marker,
                           linestyle=linestyle,
                           linewidth=3, 
                           markersize=8,
                           markerfacecolor='white',
                           markeredgewidth=2,
                           markeredgecolor=color)
                    
                    # Error bars (if std > 0)
                    valid_mask = ~np.isnan(means) & (stds > 0)
                    if np.any(valid_mask):
                        ax.fill_between(np.array(x_steps)[valid_mask], 
                                      (means - stds)[valid_mask], 
                                      (means + stds)[valid_mask], 
                                      alpha=0.2, color=color)
                    
                    plot_idx += 1
                
                if plot_idx == 0:
                    print(f"  Warning: No data plotted for {task} - {metric}")
                    plt.close()
                    continue
                
                # Formatting
                ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric_names[metric], fontsize=14, fontweight='bold')
                ax.set_title(f'{metric_names[metric]} vs Training Steps - {task}', 
                           fontsize=16, fontweight='bold', pad=20)
                
                # Grid and legend
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                
                # Formatting
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                plt.tight_layout()
                
                # Save plot
                filename = f'{task}_{metric}_comparison.png'
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"  ✓ Saved {filename}")
    
    def create_multi_task_overview(self):
        """Create overview plots comparing all models across all tasks"""
        if len(self.tasks) <= 1:
            return  # Skip if only one task
            
        metrics = ['eval_mse', 'eval_l1', 'eval_huber']
        metric_names = {'eval_mse': 'MSE Loss', 'eval_l1': 'L1 Loss', 'eval_huber': 'Huber Loss'}
        
        for metric in metrics:
            fig, axes = plt.subplots(1, len(self.tasks), figsize=(6*len(self.tasks), 6))
            if len(self.tasks) == 1:
                axes = [axes]
            
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            default_markers = ['o', 's', '^', 'v', 'D', 'p']
            default_linestyle = '-'
            
            for task_idx, task in enumerate(self.tasks):
                ax = axes[task_idx]
                x_steps = [int(step) for step in self.checkpoint_steps]
                
                plot_idx = 0
                
                # Plot each configured model for this task
                for config in self.model_configs:
                    file_data = self.data[config.file_path]
                    
                    if (config.architecture not in file_data or 
                        task not in file_data[config.architecture]):
                        continue
                    
                    task_data = file_data[config.architecture][task]
                    
                    means = []
                    stds = []
                    
                    for step in self.checkpoint_steps:
                        if step in task_data:
                            means.append(task_data[step][f'{metric}_mean'])
                            stds.append(task_data[step][f'{metric}_std'])
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    if np.all(np.isnan(means)):
                        continue
                    
                    # Use custom or default styles
                    color = config.color if config.color else default_colors[plot_idx % len(default_colors)]
                    marker = config.marker if config.marker else default_markers[plot_idx % len(default_markers)]
                    linestyle = config.linestyle if config.linestyle else default_linestyle
                    
                    ax.plot(x_steps, means, 
                           label=config.custom_label, 
                           color=color, 
                           marker=marker,
                           linestyle=linestyle,
                           linewidth=2.5, 
                           markersize=6,
                           markerfacecolor='white',
                           markeredgewidth=2,
                           markeredgecolor=color)
                    
                    # Error bars
                    valid_mask = ~np.isnan(means) & (stds > 0)
                    if np.any(valid_mask):
                        ax.fill_between(np.array(x_steps)[valid_mask], 
                                      (means - stds)[valid_mask], 
                                      (means + stds)[valid_mask], 
                                      alpha=0.2, color=color)
                    
                    plot_idx += 1
                
                ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
                if task_idx == 0:
                    ax.set_ylabel(metric_names[metric], fontsize=12, fontweight='bold')
                ax.set_title(f'{task}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                if task_idx == len(self.tasks) - 1:  # Legend on last subplot
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            plt.suptitle(f'{metric_names[metric]} Comparison Across All Tasks', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            filename = f'multi_task_overview_{metric}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Saved {filename}")
    
    def create_summary_table(self):
        """Create a summary table of best results"""
        summary = {}
        
        for task in self.tasks:
            summary[task] = {}
            
            for config in self.model_configs:
                file_data = self.data[config.file_path]
                
                if (config.architecture not in file_data or 
                    task not in file_data[config.architecture]):
                    continue
                
                task_data = file_data[config.architecture][task]
                
                # Find best checkpoint for each metric
                best_mse = float('inf')
                best_l1 = float('inf')
                best_huber = float('inf')
                best_mse_step = None
                best_l1_step = None
                best_huber_step = None
                
                for step, metrics in task_data.items():
                    if metrics['eval_mse_mean'] < best_mse:
                        best_mse = metrics['eval_mse_mean']
                        best_mse_step = step
                    if metrics['eval_l1_mean'] < best_l1:
                        best_l1 = metrics['eval_l1_mean']
                        best_l1_step = step
                    if metrics['eval_huber_mean'] < best_huber:
                        best_huber = metrics['eval_huber_mean']
                        best_huber_step = step
                
                summary[task][config.custom_label] = {
                    'best_mse': (best_mse, best_mse_step),
                    'best_l1': (best_l1, best_l1_step),
                    'best_huber': (best_huber, best_huber_step)
                }
        
        # Save summary as JSON
        summary_path = self.output_dir / 'summary_best_results.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create readable summary text
        summary_text_path = self.output_dir / 'summary_best_results.txt'
        with open(summary_text_path, 'w') as f:
            f.write("=== BEST RESULTS SUMMARY ===\n\n")
            
            for task in self.tasks:
                f.write(f"Task: {task}\n")
                f.write("-" * (len(task) + 6) + "\n")
                
                for model_key, results in summary[task].items():
                    f.write(f"\n{model_key}:\n")
                    f.write(f"  Best MSE:   {results['best_mse'][0]:.6f} (step {results['best_mse'][1]})\n")
                    f.write(f"  Best L1:    {results['best_l1'][0]:.6f} (step {results['best_l1'][1]})\n")
                    f.write(f"  Best Huber: {results['best_huber'][0]:.6f} (step {results['best_huber'][1]})\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"✓ Saved summary files")
        return summary
    
    def run_analysis(self):
        """Run complete analysis and generate all plots"""
        print(f"Starting analysis with {len(self.model_configs)} model configurations...")
        
        # Create all visualizations
        self.create_task_comparison_plots()
        self.create_multi_task_overview()
        summary = self.create_summary_table()
        
        print(f"\n✓ Analysis complete! Results saved to: {self.unique_output_dir}")
        
        return summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Merge and visualize evaluation results with custom model labeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Format options for --configs:
  Basic (3-part):    file:architecture:label
  Extended (6-part): file:architecture:label:color:linestyle:marker

Examples:
  Basic:    "results.json:act:ACT-Normal"
  Extended: "results.json:act:ACT-Train:#007737:-:o"
  
Linestyles: - (solid), -- (dashed), -. (dashdot), : (dotted)
Markers: o (circle), s (square), ^ (triangle), v (triangle_down), D (diamond), etc.
"""
    )
    
    parser.add_argument('--tasks', nargs='+', required=True,
                       help='Task names to analyze (e.g., towel_weak_train towel_strong_train)')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Model configurations in format "file:arch:label" or "file:arch:label:color:linestyle:marker"')
    parser.add_argument('--output_dir', default='merged_results',
                       help='Output directory for merged results (default: merged_results)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create merger and run analysis
    merger = EvalResultsMerger(
        tasks=args.tasks,
        configs=args.configs,
        output_dir=args.output_dir
    )
    
    summary = merger.run_analysis()
    
    # Print quick summary to console
    print("\n=== QUICK SUMMARY ===")
    for task in merger.tasks:
        print(f"\n{task}:")
        if task in summary:
            for model_key, results in summary[task].items():
                print(f"  {model_key}: MSE={results['best_mse'][0]:.4f}, L1={results['best_l1'][0]:.4f}, Huber={results['best_huber'][0]:.4f}")
        else:
            print(f"  No results found for {task}")


if __name__ == "__main__":
    main()
