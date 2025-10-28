"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
"""
import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str, tactile_enc_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.
        
        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.robot_type = robot_type
        self.tactile_enc_type = tactile_enc_type
        self.json_file = 'data.json'

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key


    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []
        
        for task_path in glob.glob(os.path.join(self.data_dirs, '*')):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)
        
        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))


    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)


    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, 'r', encoding='utf-8') as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached


    def _extract_data(self, episode_data: Dict, key: str, parts: List[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.
        
        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')
            
        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data['data']:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                if part in sample_data[key] and sample_data[key][part] is not None:
                    qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                    data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)


    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]['colors'].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data['data']:
                relative_path = sample_data['colors'].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images


    def _parse_tactiles(self, episode_path, episode_data: Dict) -> dict[str, list[np.ndarray]]:
        """Load and stack tactile data from the episode."""

        tactiles = defaultdict(list)

        keys = episode_data["data"][0]['tactiles'].keys()

        for key in keys:
            for sample_data in episode_data['data']:
                relative_path = sample_data['tactiles'].get(key)
                if not relative_path:
                    continue

                tactile_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(tactile_path):
                    raise FileNotFoundError(f"Tactile path does not exist: {tactile_path}")

                tactile_data = np.load(tactile_path)
                if tactile_data is None:
                    raise RuntimeError(f"Failed to load tactile data: {tactile_path}")

                # Hard coded shape for G1-Inspire
                prefix = key.split('_')[0]

                if self.tactile_enc_type == "image":
                    sub_keys = ROBOT_CONFIGS[self.robot_type].tactile_to_image_shape
                    idx = 0
                    for sub_key, (channel, height, width) in sub_keys.items():
                        if sub_key.startswith(prefix):
                            size = height * width
                            data = tactile_data[idx:idx+size].reshape((1, height, width))
                            normalized_data = (data / 4095).astype(np.float32)  # Normalize to [0, 1]
                            transposed_data = normalized_data.transpose(1, 2, 0)  # (H, W, C) format
                            image_rgb = cv2.cvtColor(transposed_data, cv2.COLOR_GRAY2RGB)
                            tactiles[sub_key].append(image_rgb)
                            idx += size
                elif self.tactile_enc_type == "state":
                    sub_keys = ROBOT_CONFIGS[self.robot_type].tactile_to_state_indices
                    state_data = []
                    for sub_key, indices in sub_keys.items():
                        if sub_key.startswith(prefix):
                            indices = np.array(indices)  # (M*N, )
                            extracted_data = tactile_data[indices]  # (M*N, )
                            average_value = np.mean(extracted_data)
                            state_data.append(average_value)
                    state_data = np.array(state_data, dtype=np.float32)  # (num_sub_keys, )
                    tactiles[prefix].append(state_data)
                else:
                    raise NotImplementedError(f"Tactile encoding type '{self.tactile_enc_type}' is not implemented.")

        return tactiles


    def _parse_external_tactiles(
        self,
        episode_path,
        episode_data: Dict,
        min_val: float = -300,
        max_val: float = 1100,
    ) -> dict[str, list[np.ndarray]]:
        """Load and stack tactile data from the episode."""

        tactiles = defaultdict(list)

        keys = episode_data["data"][0]['carpet_tactiles'].keys()

        for key in keys:
            for sample_data in episode_data['data']:
                relative_path = os.path.basename(sample_data['carpet_tactiles'].get(key))
                relative_path = os.path.join('carpet_tactiles', relative_path)
                if not relative_path:
                    continue

                tactile_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(tactile_path):
                    raise FileNotFoundError(f"Tactile path does not exist: {tactile_path}")

                tactile_data = np.load(tactile_path)
                if tactile_data is None:
                    raise RuntimeError(f"Failed to load tactile data: {tactile_path}")

                data = tactile_data[None, ...]
                normalized_data = ((data - min_val) / (max_val - min_val)).astype(np.float32)  # Normalize to [0, 1]
                transposed_data = normalized_data.transpose(1, 2, 0)  # (H, W, C) format
                image_rgb = cv2.cvtColor(transposed_data, cv2.COLOR_GRAY2RGB)
                tactiles[key].append(image_rgb)
        return tactiles


    def get_item(self, index: Optional[int] = None,) -> Dict:
        """Get a training sample from the dataset.  """
            
        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, 'actions', self.json_action_data_name)
        state = self._extract_data(episode_data, 'states', self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]
        
        # Load task description
        task = episode_data.get('text', {}).get('goal', "")
        
        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Load tactile data
        tactiles = self._parse_tactiles(file_path, episode_data)
        if self.tactile_enc_type == "state":
            # Concatenate tactile state data along the last dimension
            state_tactiles = np.concatenate([
                tactiles.pop(key) for key in sorted(tactiles.keys())
            ], axis=-1)

            # Append tactile state data to the original state
            state = np.concatenate([state, state_tactiles], axis=-1)

        # Load external tactile data if available
        external_tactiles = self._parse_external_tactiles(file_path, episode_data)
        tactiles.update(external_tactiles)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            'camera_names': list(cameras.keys()),
            'cam_height': cam_height,
            'cam_width': cam_width,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
        
        return {'episode_index': index,
                'episode_length': episode_length,
                'state': state, 
                'action': action,
                'cameras': cameras,
                'tactiles': tactiles,
                'task': task,
                'data_cfg':data_cfg}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    tactile_enc_type: Literal["image", "state"] = "image",
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    print("cameras", cameras)
    for cam in cameras:
        if cam == 'cam_left_high':
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 848),  # TODO: check whether hardcoded is required
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

        if cam == 'cam_third':
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 640),  # TODO: check whether hardcoded is required
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    tactiles = getattr(ROBOT_CONFIGS[robot_type], "tactiles", [])
    tactile_to_image_shape = getattr(ROBOT_CONFIGS[robot_type], "tactile_to_image_shape", {})
    if tactile_enc_type == "state":
        state = features["observation.state"]
        names = list(ROBOT_CONFIGS[robot_type].tactile_to_state_indices.keys())

        # Modify the state feature to include tactile state names
        features[f"observation.state"] = {
            "dtype": "float32",
            "shape": (state["shape"][0] + len(names),),
            "names": state["names"] + names,
        }

        # No need to add tactile images if using state encoding
        tactiles = ["carpet_0"] if "carpet_0" in tactiles else []  # TODO: check whether hardcoded is required

    for tactile in tactiles:
        features[f"observation.images.{tactile}"] = {
            "dtype": "image",
            "shape": tactile_to_image_shape[tactile],
            "names": [
                "channels",
                "height",
                "width",
                "channel", # tactile carpet
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    tactile_enc_type: Literal["image", "state"] = "image",
) -> LeRobotDataset:

    json_dataset = JsonDataset(raw_dir, robot_type, tactile_enc_type)
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length

        for i in range(num_frames):
            frame = {
                "observation.state": state[i], # mod
                "action": action[i],
                "task": task
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            for tactile, tactile_array in episode["tactiles"].items():
                frame[f"observation.images.{tactile}"] = tactile_array[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,        # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    tactile_enc_type: Literal["image", "state"] = "image",
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        tactile_enc_type=tactile_enc_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        tactile_enc_type=tactile_enc_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder = True)


def local_push_to_hub(
        repo_id: str,
        root_path: Path,):

    dataset = LeRobotDataset(repo_id = repo_id, root = root_path)
    dataset.push_to_hub(upload_large_folder = True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)

"""
Usage:
python convert_unitree_json_to_lerobot.py
--raw-dir <path_to_your_json_dataset> \
--repo-id <huggingface_repo_id> \
--robot_type Unitree_G1_Inspire
--tactile_enc_type state
"""
