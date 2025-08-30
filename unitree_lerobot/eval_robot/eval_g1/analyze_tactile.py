from typing import Literal
import tqdm
import logging
from dataclasses import dataclass
from pprint import pformat
from dataclasses import asdict

import torch

from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import RobotConfig, ROBOT_CONFIGS

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns


@dataclass
class AnalysisConfig:
    repo_id: str
    arm_type: str = "g1"
    hand_type: str = "inspire"
    tactile_enc_type: Literal["image"] = "image"


def plot_tactile_over_time(tactile_signals: dict, metric="Average", fps=None):
    N = len(tactile_signals.keys())
    rows = int(round(N ** 0.5))
    rows = max(rows, 1)
    cols = int(math.ceil(N / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), sharex=True)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, (name, arr) in enumerate(tactile_signals.items()):
        ax = axes[i]
        assert arr.ndim == 3, f"{name} must be (N_STEP, W, H), got {arr.shape}"

        if metric == "Average":
            ts = arr.mean(axis=(1, 2))  # (N_STEP,)
        elif metric == "Max":
            ts = arr.max(axis=(1, 2))
        elif metric == "Min":
            ts = arr.min(axis=(1, 2))
        x = np.arange(len(ts))
        if fps is not None and fps > 0:
            x = x / float(fps)

        ax.plot(x, ts)
        ax.set_title(name, fontsize=9)
        if i % cols == 0:
            ax.set_ylabel("mean over (W,H)")
        if i // cols == rows - 1:
            ax.set_xlabel("time (s)" if fps else "step")

    # Hide unused axes
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle(f"{metric} Tactile Signals Over Time", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"tactile_signals_{metric}.png", dpi=300, bbox_inches='tight')


def plot_tactile_confusion_matrix(tactile_signals_array: dict):
    """
    Plot confusion matrix for tactile signals.
    :param tactile_signals_array: dict[name] = (N_STEP, W, H) tactile signals
    :param tactile_names: list of tactile signal names
    :param metric: "Average", "Min", or "Max"
    """
    names = list(tactile_signals_array.keys())
    X = np.stack([tactile_signals_array[n].mean(axis=(1, 2)) for n in names], axis=1)  # (N_STEP, N_PATCH)
    corr = np.corrcoef(X.T)  # (N_PATCH, N_PATCH)

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, xticklabels=names, yticklabels=names, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Tactile patch correlation matrix")
    plt.savefig("tactile_correlation_matrix.png", dpi=300, bbox_inches='tight')


def analyze_tactile(robot_cfg: RobotConfig, dataset: LeRobotDataset):
    logging.info("Analyzing tactile data...")

    camera_names = robot_cfg.cameras
    tactile_names = getattr(robot_cfg, "tactiles", [])

    from collections import defaultdict
    tactile_signals = defaultdict(list)

    print("Loading tactile data from dataset...")
    if tactile_names:
        for epi_idx in range(dataset.num_episodes):
            # init pose
            from_idx = dataset.episode_data_index["from"][epi_idx].item()
            to_idx = dataset.episode_data_index["to"][epi_idx].item()

            tactile_signals_in_episode = defaultdict(list)
            for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc=f"Episode {epi_idx+1}/{dataset.num_episodes}"):
                step = dataset[step_idx]

                # Process tactile data
                for tac_name in tactile_names:
                    # Case where tactile data is stored as images
                    tactile_img = step.get(f"observation.images.{tac_name}", None)
                    if tactile_img is not None:
                        tactile_signals_in_episode[tac_name].append(tactile_img[0])  # first dim is dummy channel

            for tac_name in tactile_names:
                tactile_signals_in_episode_array = np.stack(tactile_signals_in_episode[tac_name], axis=0)
                tactile_signals[tac_name].append(tactile_signals_in_episode_array)

    tactile_array_flat = {
        tac_name: np.concat(tactile_signals[tac_name], axis=0)
        for tac_name in tactile_signals
    }

    logging.info(f"Tactile signals collected: {len(tactile_signals)}")

    # visualize signals over time
    for metric in ["Average", "Min", "Max"]:
        plot_tactile_over_time(
            tactile_array_flat,
            metric=metric,
            fps=None,  # If fps=dataset.fps, the x-axis will be in seconds
        )

    # visualize confusion matrix
    plot_tactile_confusion_matrix(tactile_array_flat)

    logging.info("End of tactile analysis")


@parser.wrap()
def analyze_main(cfg: AnalysisConfig):
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    # Load robot configuration based on arm and hand type
    robot_config = ROBOT_CONFIGS[
        f"Unitree"
        f"_{cfg.arm_type.capitalize()}"
        f"_{cfg.hand_type.capitalize()}"
    ]

    # Analyze tactile data
    analyze_tactile(robot_config, dataset)

    logging.info("End of analysis")


if __name__ == "__main__":
    init_logging()
    analyze_main()

"""
Usage:
python analyze_tactile.py --repo_id <your_dataset_repo_id> --arm_type g1 --hand_type inspire --tactile_enc_type image
"""
