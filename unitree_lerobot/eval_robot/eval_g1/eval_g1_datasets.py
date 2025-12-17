"""
Comprehensive evaluation script for comparing checkpoints across training steps
Compares MLP vs ACT architectures with multiple seeds across different checkpoint steps
Uses shell commands to call eval_g1_dataset.py for each evaluation

Usage:
    Basic usage with repo IDs:
    python unitree_lerobot/eval_robot/eval_g1/eval_g1_datasets.py --repo_ids user/towel_strong_train user/towel_weak_train

    With custom architectures and checkpoint steps:
    python unitree_lerobot/eval_robot/eval_g1/eval_g1_datasets.py \
        --repo_ids user/towel_strong_train user/towel_weak_train \
        --architectures mlp act \
        --checkpoint_steps 020000 040000 060000 080000 100000

    With custom dataset names (matching directory names):
    python unitree_lerobot/eval_robot/eval_g1/eval_g1_datasets.py \
        --repo_ids user/towel_strong_train user/towel_weak_train \
        --dataset_names towel_strong_train towel_weak_train \
        --models_base_path outputs/models \
        --output_dir outputs/checkpoint_evaluation_results \
        --device cuda

Arguments:
    --repo_ids: List of HuggingFace repo IDs to evaluate against (required)
    --dataset_names: Custom names for datasets (optional, defaults to repo names).
                    Order must match repo_ids (i.e., dataset_names[i] corresponds to repo_ids[i])
    --architectures: Model architectures to compare (default: mlp act)
    --checkpoint_steps: Training steps to evaluate (default: 020000 040000 060000 080000 100000)
    --models_base_path: Base path for model checkpoints (default: outputs/models)
    --output_dir: Output directory for results (default: outputs/checkpoint_evaluation_results)
    --device: Device to use for evaluation (default: cuda)

한글 설명:
    --repo_ids: 평가에 사용할 HuggingFace repo ID 리스트 (필수)
    --dataset_names: 각 repo에 대응하는 데이터셋 이름 (선택, 기본값은 repo 이름의 마지막 부분).
                    repo_ids와 순서를 반드시 맞춰야 하며, dataset_names[i]는 repo_ids[i]에 대응함
    --architectures: 비교할 모델 아키텍처 리스트 (기본값: mlp act)
    --checkpoint_steps: 평가할 학습 step 리스트 (기본값: 020000 040000 060000 080000 100000)
    --models_base_path: 모델 체크포인트가 저장된 기본 경로 (기본값: outputs/models)
    --output_dir: 결과를 저장할 출력 디렉토리 (기본값: outputs/checkpoint_evaluation_results)
    --device: 평가에 사용할 디바이스 (기본값: cuda)

Output:
    - Statistics JSON file with mean/std for each architecture-dataset-step combination
    - Raw results JSON file with all individual evaluation results
    - Line plots showing training progress for MSE, L1, and Huber losses
    - All files saved to timestamped subdirectory for concurrent execution safety

Example directory structure expected:
    outputs/models/
    ├── towel_strong_train/
    │   ├── 13-16-50_mlp/checkpoints/020000/pretrained_model/
    │   ├── 14-25-47_act/checkpoints/020000/pretrained_model/
    │   └── ...
    └── towel_weak_train/
        ├── 13-16-29_mlp/checkpoints/020000/pretrained_model/
        └── ...
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import re
from datetime import datetime


@dataclass
class CheckpointEvalConfig:
    """Configuration for checkpoint-based evaluation"""
    architectures: List[str]  # e.g., ['mlp', 'act']
    datasets: List[str]      # e.g., ['strong_train', 'weak_train']
    repo_ids: List[str]      # e.g., ['user/strong_train', 'user/weak_train']
    checkpoint_steps: List[str]  # e.g., ['020000', '040000', '060000', '080000', '100000']
    models_base_path: str    # Base path where model checkpoints are stored
    output_dir: str          # Directory to save results and plots
    device: str = "cuda"


class CheckpointEvaluator:
    """Evaluator for comparing model checkpoints across training steps using shell commands"""
    
    def __init__(self, config: CheckpointEvalConfig):
        self.config = config
        # Structure: arch -> dataset -> checkpoint_step -> [seed_results]
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # Generate single timestamp with process ID for concurrent execution
        import os
        # Use timestamp without process ID for cleaner output paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate that datasets and repo_ids have same length
        if len(config.datasets) != len(config.repo_ids):
            raise ValueError(f"datasets ({len(config.datasets)}) and repo_ids ({len(config.repo_ids)}) must have same length")
        
        # Create dataset to repo_id mapping
        self.dataset_to_repo_id = dict(zip(config.datasets, config.repo_ids))
        
        # Create unique output directory for concurrent execution
        self.unique_output_dir = os.path.join(config.output_dir, f"run_{self.timestamp}")
        os.makedirs(self.unique_output_dir, exist_ok=True)
        print(f"Output directory: {self.unique_output_dir}")
        
    def fix_config_type(self, pretrained_model_path: str, architecture: str):
        """
        Check and fix config.json in pretrained_model directory to ensure 'type' field exists at the top
        
        Args:
            pretrained_model_path: Path to pretrained_model directory
            architecture: Architecture type ('mlp' or 'act') to add if missing
        """
        config_path = Path(pretrained_model_path) / "config.json"
        
        if not config_path.exists():
            print(f"Warning: config.json not found in {pretrained_model_path}")
            return
            
        try:
            # Read current config
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Parse JSON
            config_data = json.loads(config_content)
            
            # Check if 'type' field exists
            if 'type' not in config_data:
                print(f"Adding missing 'type': '{architecture}' to {config_path}")
                
                # Create new config with 'type' at the top
                new_config = {'type': architecture}
                new_config.update(config_data)
                
                # Write back with proper formatting
                with open(config_path, 'w') as f:
                    json.dump(new_config, f, indent=2)
                    
            elif config_data.get('type') != architecture:
                print(f"Warning: Config type '{config_data.get('type')}' doesn't match expected '{architecture}' in {config_path}")
            else:
                print(f"Config type '{architecture}' already correct in {config_path}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing config.json in {pretrained_model_path}: {e}")
        except Exception as e:
            print(f"Error fixing config in {pretrained_model_path}: {e}")
        
    def find_checkpoint_paths(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Find checkpoint paths for each architecture, dataset, and checkpoint step
        Returns: {arch: {dataset: {checkpoint_step: [seed_path1, seed_path2, seed_path3]}}}
        """
        checkpoint_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        base_path = Path(self.config.models_base_path)
        
        for arch in self.config.architectures:
            for dataset in self.config.datasets:
                # Find all directories for this dataset and architecture
                # Use exact dataset name matching to avoid confusion between similar names
                dataset_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name == dataset]
                if not dataset_dirs:
                    # Fallback to partial matching if exact match not found
                    dataset_dirs = [d for d in base_path.iterdir() if d.is_dir() and dataset in d.name]
                
                pattern_dirs = []
                for dataset_dir in dataset_dirs:
                    arch_dirs = list(dataset_dir.glob(f"*{arch}*"))
                    pattern_dirs.extend(arch_dirs)
                
                # Sort by modification time (consistent seed ordering)
                pattern_dirs.sort(key=lambda x: x.stat().st_mtime)
                
                print(f"Found {len(pattern_dirs)} seed directories for {arch}-{dataset}")
                
                # For each checkpoint step
                for checkpoint_step in self.config.checkpoint_steps:
                    for seed_dir in pattern_dirs:
                        checkpoint_path = seed_dir / "checkpoints" / checkpoint_step / "pretrained_model"
                        if checkpoint_path.exists():
                            # Fix config.json type field before adding to paths
                            self.fix_config_type(str(checkpoint_path), arch)
                            checkpoint_paths[arch][dataset][checkpoint_step].append(str(checkpoint_path))
                        else:
                            print(f"Warning: Checkpoint {checkpoint_step} not found in {seed_dir}")
                            
        return checkpoint_paths
    
    def run_single_evaluation(self, pretrained_model_path: str, repo_id: str) -> Dict[str, Any]:
        """Run a single evaluation using shell command to eval_g1_dataset.py"""
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_result_file = temp_file.name
        
        try:
            # Build command with device specification
            cmd = [
                "python", "unitree_lerobot/eval_robot/eval_g1/eval_g1_dataset.py",
                f"--policy.path={pretrained_model_path}",
                f"--repo_id={repo_id}",
                f"--policy.device={self.config.device}"
            ]
            
            print(f"[DEBUG] Running command: {' '.join(cmd)}")
            
            # Set environment variables for automated execution
            env = os.environ.copy()
            env['EVAL_AUTO_MODE'] = 'true'
            env['EVAL_OUTPUT_FILE'] = temp_result_file
            
            # Run command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd(),
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            print(f"[DEBUG] Command return code: {result.returncode}")
            print(f"[DEBUG] STDOUT: {result.stdout}")
            print(f"[DEBUG] STDERR: {result.stderr}")
            
            if result.returncode != 0:
                print(f"[ERROR] Command failed with return code {result.returncode}")
                return None
            
            # Load results from temporary file
            if os.path.exists(temp_result_file):
                # Check if file has content
                file_size = os.path.getsize(temp_result_file)
                print(f"[DEBUG] Result file size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"[ERROR] Result file is empty: {temp_result_file}")
                    return None
                
                with open(temp_result_file, 'r') as f:
                    content = f.read()
                    print(f"[DEBUG] File content preview: {content[:200]}...")
                    
                try:
                    results = json.loads(content)
                    print(f"[DEBUG] Successfully loaded results from {temp_result_file}")
                    return results
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON decode error: {e}")
                    print(f"[ERROR] File content: {content}")
                    return None
            else:
                print(f"[ERROR] Result file not created: {temp_result_file}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"[ERROR] Exception during evaluation: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_result_file):
                os.unlink(temp_result_file)
    
    def run_evaluation(self):
        """Run evaluation for all configurations"""
        checkpoint_paths = self.find_checkpoint_paths()
        
        print(f"Found checkpoint paths structure:")
        for arch in checkpoint_paths:
            for dataset in checkpoint_paths[arch]:
                for step in checkpoint_paths[arch][dataset]:
                    n_seeds = len(checkpoint_paths[arch][dataset][step])
                    print(f"  {arch}-{dataset}-{step}: {n_seeds} seeds")
        
        for arch in self.config.architectures:
            for dataset in self.config.datasets:
                if dataset not in checkpoint_paths[arch]:
                    print(f"Warning: No checkpoints found for {arch}-{dataset}")
                    continue
                    
                print(f"Evaluating {arch} on {dataset}...")
                repo_id = self.dataset_to_repo_id[dataset]
                
                for checkpoint_step in self.config.checkpoint_steps:
                    if checkpoint_step not in checkpoint_paths[arch][dataset]:
                        print(f"  Warning: No checkpoint {checkpoint_step} found")
                        continue
                        
                    print(f"  Checkpoint {checkpoint_step}:")
                    seed_paths = checkpoint_paths[arch][dataset][checkpoint_step]
                    
                    for i, pretrained_model_path in enumerate(seed_paths):
                        print(f"    Seed {i+1}: {pretrained_model_path}")
                        
                        results = self.run_single_evaluation(pretrained_model_path, repo_id)
                        
                        if results:
                            self.results[arch][dataset][checkpoint_step].append(results)
                            print(f"    ✓ Success - MSE: {results.get('eval_mse_loss', 'N/A'):.4f}")
                        else:
                            print(f"    ✗ Failed")
    
    def compute_statistics(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """Compute mean and std for each architecture-dataset-checkpoint combination"""
        # Structure: arch -> dataset -> checkpoint_step -> {metric_mean, metric_std}
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        print(f"[DEBUG] Computing statistics...")
        
        for arch in self.results:
            for dataset in self.results[arch]:
                for checkpoint_step in self.results[arch][dataset]:
                    results_list = self.results[arch][dataset][checkpoint_step]
                    
                    print(f"[DEBUG] {arch}-{dataset}-{checkpoint_step}: {len(results_list)} results")
                    
                    if not results_list:
                        continue
                    
                    # Extract standardized evaluation metrics from all seeds
                    eval_mse_values = [r['eval_mse_loss'] for r in results_list if 'eval_mse_loss' in r]
                    eval_l1_values = [r['eval_l1_loss'] for r in results_list if 'eval_l1_loss' in r]
                    eval_huber_values = [r['eval_huber_loss'] for r in results_list if 'eval_huber_loss' in r]
                    
                    if not eval_mse_values:
                        print(f"[WARNING] No valid results for {arch}-{dataset}-{checkpoint_step}")
                        continue
                    
                    print(f"[DEBUG] MSE values: {eval_mse_values}")
                    print(f"[DEBUG] L1 values: {eval_l1_values}")
                    print(f"[DEBUG] Huber values: {eval_huber_values}")
                    
                    # Compute statistics for fair comparison
                    stats[arch][dataset][checkpoint_step] = {
                        'eval_mse_mean': np.mean(eval_mse_values),
                        'eval_mse_std': np.std(eval_mse_values),
                        'eval_l1_mean': np.mean(eval_l1_values),
                        'eval_l1_std': np.std(eval_l1_values),
                        'eval_huber_mean': np.mean(eval_huber_values),
                        'eval_huber_std': np.std(eval_huber_values),
                        'n_seeds': len(results_list)
                    }
                    
        return stats
    
    def plot_results(self, stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]):
        """Create line plots showing training progress for each metric"""
        
        metrics = ['eval_mse', 'eval_l1', 'eval_huber']
        
        for dataset in self.config.datasets:
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Convert checkpoint steps to integers for plotting
                x_steps = [int(step) for step in self.config.checkpoint_steps]
                
                for arch in self.config.architectures:
                    if dataset not in stats[arch]:
                        continue
                        
                    means = []
                    stds = []
                    
                    for checkpoint_step in self.config.checkpoint_steps:
                        if checkpoint_step in stats[arch][dataset]:
                            means.append(stats[arch][dataset][checkpoint_step][f'{metric}_mean'])
                            stds.append(stats[arch][dataset][checkpoint_step][f'{metric}_std'])
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    # Plot line with error bars
                    ax.plot(x_steps, means, label=arch.upper(), marker='o', linewidth=2, markersize=8)
                    ax.fill_between(x_steps, means - stds, means + stds, alpha=0.2)
                
                ax.set_xlabel('Training Steps')
                if metric == 'eval_mse':
                    ax.set_ylabel('MSE Loss')
                    ax.set_title(f'MSE Loss vs Training Steps - {dataset}')
                elif metric == 'eval_l1':
                    ax.set_ylabel('L1 Loss')
                    ax.set_title(f'L1 Loss vs Training Steps - {dataset}')
                elif metric == 'eval_huber':
                    ax.set_ylabel('Huber Loss')
                    ax.set_title(f'Huber Loss vs Training Steps - {dataset}')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Use the shared timestamp for all plots
                filename = f'{self.timestamp}_{dataset}_{metric}_vs_steps.png'
                plt.savefig(os.path.join(self.unique_output_dir, filename), dpi=300)
                plt.close()  # Close figure to save memory
        
        print(f"Plots saved to {self.unique_output_dir}")
    
    def save_results(self, stats):
        """Save detailed results to JSON file"""
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_dict = {}
        for arch in stats:
            stats_dict[arch] = {}
            for dataset in stats[arch]:
                stats_dict[arch][dataset] = dict(stats[arch][dataset])
        
        # Use the shared timestamp for all files
        
        # Save statistics
        stats_filename = f'checkpoint_evaluation_stats.json'
        stats_path = os.path.join(self.unique_output_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        # Save raw results
        raw_results = {}
        for arch in self.results:
            raw_results[arch] = {}
            for dataset in self.results[arch]:
                raw_results[arch][dataset] = {}
                for checkpoint_step in self.results[arch][dataset]:
                    raw_results[arch][dataset][checkpoint_step] = self.results[arch][dataset][checkpoint_step]
        
        raw_filename = f'raw_checkpoint_results.json'
        raw_path = os.path.join(self.unique_output_dir, raw_filename)
        with open(raw_path, 'w') as f:
            json.dump(raw_results, f, indent=2)
        
        print(f"Results saved to {self.unique_output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Checkpoint-based Multi-Architecture Evaluation')
    
    parser.add_argument('--repo_ids', nargs='+', required=True,
                       help='List of repo IDs to evaluate (e.g., user/strong_train user/weak_train)')
    parser.add_argument('--dataset_names', nargs='+', 
                       help='Dataset names (if not provided, will use repo names)')
    parser.add_argument('--architectures', nargs='+', default=['mlp', 'act'],
                       help='Architectures to evaluate (default: mlp act)')
    parser.add_argument('--checkpoint_steps', nargs='+', default=['020000', '040000', '060000', '080000', '100000'],
                       help='Checkpoint steps to evaluate (default: 020000 040000 060000 080000 100000)')
    parser.add_argument('--models_base_path', default='outputs/models',
                       help='Base path for model checkpoints (default: outputs/models)')
    parser.add_argument('--output_dir', default='outputs/checkpoint_evaluation_results',
                       help='Output directory for results (default: outputs/checkpoint_evaluation_results)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (default: cuda)')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    
    # Parse command line arguments
    args = parse_args()
    
    # If dataset names not provided, extract from repo_ids
    if args.dataset_names is None:
        dataset_names = [repo_id.split('/')[-1] for repo_id in args.repo_ids]
    else:
        dataset_names = args.dataset_names
        if len(dataset_names) != len(args.repo_ids):
            raise ValueError(f"Number of dataset names ({len(dataset_names)}) must match number of repo IDs ({len(args.repo_ids)})")
    
    # Configuration
    config = CheckpointEvalConfig(
        architectures=args.architectures,
        datasets=dataset_names,
        repo_ids=args.repo_ids,
        checkpoint_steps=args.checkpoint_steps,
        models_base_path=args.models_base_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print(f"Evaluating architectures: {config.architectures}")
    print(f"Datasets: {config.datasets}")
    print(f"Repo IDs: {config.repo_ids}")
    print(f"Checkpoint steps: {config.checkpoint_steps}")
    
    # Run evaluation
    evaluator = CheckpointEvaluator(config)
    evaluator.run_evaluation()
    
    # Compute statistics and create plots
    stats = evaluator.compute_statistics()
    evaluator.plot_results(stats)
    evaluator.save_results(stats)
    
    # Print summary
    print("\n=== Checkpoint Evaluation Summary ===")
    for arch in stats:
        print(f"\n{arch.upper()} Architecture:")
        for dataset in stats[arch]:
            print(f"  {dataset}:")
            for checkpoint_step in sorted(stats[arch][dataset].keys()):
                s = stats[arch][dataset][checkpoint_step]
                print(f"    Step {checkpoint_step}: MSE={s['eval_mse_mean']:.4f}±{s['eval_mse_std']:.4f}, "
                      f"L1={s['eval_l1_mean']:.4f}±{s['eval_l1_std']:.4f}, "
                      f"Huber={s['eval_huber_mean']:.4f}±{s['eval_huber_std']:.4f} (n={s['n_seeds']})")


if __name__ == "__main__":
    main()