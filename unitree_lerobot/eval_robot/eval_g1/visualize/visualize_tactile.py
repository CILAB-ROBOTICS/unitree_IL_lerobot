from typing import Literal, List, Tuple
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
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

from unitree_lerobot.eval_robot.eval_g1.visualize.const import IMAGE_PATH, VERTICES


@dataclass
class VisualizeConfig:
    repo_id: str
    arm_type: str = "g1"
    hand_type: str = "inspire"
    tactile_enc_type: Literal["image"] = "image"


def split_vertice(vertices: List[Tuple[int, int]], image_shape: Tuple[int, int, int]):
    """
    vertices: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]  # axis-aligned rectangle assumed
    image_shape: (C, H, W)  # channel, height, width

    return: list of list of vertices (H x W)
    """
    _, H, W = image_shape

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_edges = [round(x_min + (x_max - x_min) * c / W) for c in range(W + 1)]
    y_edges = [round(y_min + (y_max - y_min) * r / H) for r in range(H + 1)]

    sub_rects = []
    for r in range(H):
        row = []
        for c in range(W):
            x0, x1 = x_edges[c], x_edges[c + 1]
            y0, y1 = y_edges[r], y_edges[r + 1]
            row.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        sub_rects.append(row)

    return sub_rects


def visualize_tactile(robot_cfg: RobotConfig, dataset: LeRobotDataset):
    logging.info("Visualizing tactile data...")

    camera_names = robot_cfg.cameras
    tactile_names = getattr(robot_cfg, "tactiles", [])

    if tactile_names:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            pass

        # Load background image for visualization
        bg_img = plt.imread(IMAGE_PATH)
        cmap = matplotlib.colormaps['viridis']
        amplifier = 10.0

        # Prepare figure
        plt.ion()
        n_cam = max(len(camera_names), 1)
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(nrows=2, ncols=n_cam, height_ratios=[1.1, 1.6], hspace=0.02, wspace=0.02, figure=fig)

        # Prepare axis for camera visualization
        axes_cam = {}
        cam_artists = {}
        for i, cam in enumerate(camera_names):
            axc = fig.add_subplot(gs[0, i])
            axc.set_title(cam, fontsize=10)
            axc.axis('off')
            cam_artists[cam] = axc.imshow(np.zeros((12, 16, 3), dtype=np.float32))
            axes_cam[cam] = axc
        if not camera_names:
            axc = fig.add_subplot(gs[0, 0])
            axc.set_title("No camera", fontsize=10)
            axc.axis('off')
            axc.imshow(np.zeros((12, 16, 3), dtype=np.float32))

        # Prepare axis for tactile visualization
        ax_tac = fig.add_subplot(gs[1, :])
        ax_tac.imshow(bg_img)
        ax_tac.axis('off')

        # Prepare patches for each tactile sensor
        patches = {}
        for name, verts in VERTICES.items():
            if name in tactile_names:
                image_shape = robot_cfg.tactile_to_image_shape[name]
                sub_rects = split_vertice(verts, image_shape)
                patch_grid = []
                for r in range(image_shape[1]):
                    row = []
                    for c in range(image_shape[2]):
                        polygon = Polygon(sub_rects[r][c], closed=True, edgecolor='gray', facecolor='black', alpha=0.7)
                        ax_tac.add_patch(polygon)
                        row.append(polygon)
                    patch_grid.append(row)
                patches[name] = patch_grid

        plt.show(block=False)

        # Iterate through episodes and steps
        for epi_idx in range(dataset.num_episodes):
            # init pose
            from_idx = dataset.episode_data_index["from"][epi_idx].item()
            to_idx = dataset.episode_data_index["to"][epi_idx].item()

            for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
                step = dataset[step_idx]

                # Process tactile data
                for tac_name in tactile_names:
                    # Case where tactile data is stored as images
                    tactile_img = step.get(f"observation.images.{tac_name}", None)
                    if tactile_img is not None:
                        tactile_data = tactile_img[0]

                        # Set value of heatmap
                        for r in range(tactile_data.shape[0]):
                            for c in range(tactile_data.shape[1]):
                                intensity = tactile_data[r, c].item()
                                color = cmap(intensity * amplifier)
                                if tac_name.endswith("_tactile_palm"):
                                    idx = tactile_data.shape[1] * r + c
                                    new_c = idx // tactile_data.shape[0]
                                    new_r = tactile_data.shape[0] - (idx - new_c * tactile_data.shape[0]) - 1
                                    patches[tac_name][new_r][new_c].set_facecolor(color)
                                else:
                                    patches[tac_name][r][c].set_facecolor(color)
                    else:
                        for r in range(len(patches[tac_name])):
                            for c in range(len(patches[tac_name][0])):
                                patches[tac_name][r][c].set_facecolor(cmap(0.0))

                # Process image data if needed
                for cam_name in camera_names:
                    image = step.get(f"observation.images.{cam_name}", None)
                    if image is not None:
                        frame = image.permute(1, 2, 0)  # C, H, W -> H, W, C
                        cam_artists[cam_name].set_data(frame)

                # Update visualization
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

    print("End of tactile data visualization")


@parser.wrap()
def visualize_main(cfg: VisualizeConfig):
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    # Load robot configuration based on arm and hand type
    robot_config = ROBOT_CONFIGS[
        f"Unitree"
        f"_{cfg.arm_type.capitalize()}"
        f"_{cfg.hand_type.capitalize()}"
    ]

    # Analyze tactile data
    visualize_tactile(robot_config, dataset)

    logging.info("End of analysis")


if __name__ == "__main__":
    init_logging()
    visualize_main()

"""
Usage:
python visualize_tactile.py --repo_id <your_dataset_repo_id> --arm_type g1 --hand_type inspire --tactile_enc_type image
"""
