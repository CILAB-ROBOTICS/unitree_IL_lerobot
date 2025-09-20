from typing import Literal, List
import tqdm
import logging
from dataclasses import dataclass
from pprint import pformat
from dataclasses import asdict

from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import RobotConfig, ROBOT_CONFIGS

import numpy as np


@dataclass
class AnalysisConfig:
    repo_id: str
    arm_type: str = "g1"
    hand_type: str = "inspire"
    tactile_enc_type: Literal["image", "state"] = "image"
    save_dir: str = "./results"


def parse_tactile_signals(
    robot_cfg: RobotConfig,
    tactile_enc_type: str,
    step: dict,
    prefixs: List[str] = ["left_tactile", "right_tactile"],
):
    tactile_signals = {}

    if tactile_enc_type == "image":
        for tac_name in robot_cfg.tactiles:
            if any(tac_name.startswith(p) for p in prefixs):
                tactile_img = step.get(f"observation.images.{tac_name}", None)
                if tactile_img is not None:
                    tactile_signal = tactile_img[0, :, :].numpy()  # (H, W) shape, use first channel
                    tactile_signals[tac_name] = tactile_signal

    elif tactile_enc_type == "state":
        total_pixels = sum([
            h * w for tac_name, (c, h, w) in robot_cfg.tactile_to_image_shape.items()
            if any(tac_name.startswith(p) for p in prefixs)
        ])
        empty_data = np.zeros((total_pixels,), dtype=np.float32).reshape(len(prefixs), -1)

        # Reconstruct raw tactile data from state
        start_idx = len(robot_cfg.motors)
        tactile_state = step["observation.state"][start_idx:]
        for i, (tac_name, pixel_indices) in enumerate(robot_cfg.tactile_to_state_indices.items()):
            tactile_data = tactile_state[i].item()
            tac_channel = 0 if tac_name.startswith(prefixs[0]) else 1
            empty_data[tac_channel, pixel_indices] = tactile_data

        # indexing and reshaping into images
        flatten_data = empty_data.flatten()
        idx = 0
        for tac_name, (channel, height, width) in robot_cfg.tactile_to_image_shape.items():
            if any(tac_name.startswith(p) for p in prefixs):
                size = height * width
                tactile_signal = flatten_data[idx:idx + size].reshape((height, width))
                tactile_signals[tac_name] = tactile_signal
                idx += size

    return tactile_signals


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

    logging.info("Analyzing tactile data...")

    infos = []
    for epi_idx in range(dataset.num_episodes):
        # init pose
        from_idx = dataset.episode_data_index["from"][epi_idx].item()
        to_idx = dataset.episode_data_index["to"][epi_idx].item()

        # Cumulate tactile signals over the episode
        cum_tactile_signal = {}
        for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc=f"Episode {epi_idx + 1}/{dataset.num_episodes}"):
            step = dataset[step_idx]
            tactile_signal = parse_tactile_signals(robot_config, cfg.tactile_enc_type, step)
            for tac_name, tactile_data in tactile_signal.items():
                if tac_name not in cum_tactile_signal:
                    cum_tactile_signal[tac_name] = tactile_data
                else:
                    cum_tactile_signal[tac_name] += tactile_data

        # Extract tactile info
        info = {'repo_id': dataset.repo_id, 'episode_index': epi_idx}
        for tac_name, tactile_data in cum_tactile_signal.items():
            infos.append(
                {
                    'name': tac_name,
                    'shape': tactile_data.shape,
                    'sum': float(np.sum(tactile_data)),
                    **info,
                }
            )

    # Save infos as csv
    import os
    import pandas as pd
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, f"{cfg.repo_id.split('/')[1]}.csv")
    df = pd.DataFrame(infos)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved analysis results to {save_path}")


if __name__ == "__main__":
    init_logging()
    analyze_main()

"""
Usage:
python extract_tactile_info.py \
    --repo_id <your_dataset_repo_id> \
    --arm_type g1 \
    --hand_type inspire \
    --tactile_enc_type image \
    --save_dir ./results
"""
