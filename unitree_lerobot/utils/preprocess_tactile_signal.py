import cv2
import numpy as np

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


def tactile_as_image(key: str, tactile_data: np.array, robot_type: str = "Unitree_G1_Inspire") -> dict[str, np.ndarray]:
    tactile_dict = {}

    prefix = key.split('_')[0]
    sub_keys = ROBOT_CONFIGS[robot_type].tactile_to_image_shape
    idx = 0
    for sub_key, (channel, height, width) in sub_keys.items():
        if sub_key.startswith(prefix):
            size = height * width
            data = tactile_data[idx:idx + size].reshape((1, height, width))
            normalized_data = (data / 4095).astype(np.float32)  # Normalize to [0, 1]
            transposed_data = normalized_data.transpose(1, 2, 0)  # (H, W, C) format
            image_rgb = cv2.cvtColor(transposed_data, cv2.COLOR_GRAY2RGB)
            tactile_dict[sub_key] = image_rgb
            idx += size
    return tactile_dict


def tactile_as_state(key: str, tactile_data: np.array, robot_type: str = "Unitree_G1_Inspire") -> dict[str, np.ndarray]:
    state_dict = {}

    prefix = key.split('_')[0]
    sub_keys = ROBOT_CONFIGS[robot_type].tactile_to_state_indices
    state_data = []
    for sub_key, indices in sub_keys.items():
        if sub_key.startswith(prefix):
            indices = np.array(indices)  # (M*N, )
            extracted_data = tactile_data[indices]  # (M*N, )
            average_value = np.mean(extracted_data)
            state_data.append(average_value)
    state_data = np.array(state_data, dtype=np.float32)  # (num_sub_keys, )
    state_dict[prefix] = state_data
    return state_dict


def parse_tactile_as_image(tactiles: dict, robot_type: str = "Unitree_G1_Inspire") -> dict:
    image_dict = {}
    for key, tactile_data in tactiles.items():
        tactile_dict = tactile_as_image(key, tactile_data, robot_type)
        image_dict.update(tactile_dict)
    return image_dict


def parse_tactile_as_state(tactiles: dict, robot_type: str = "Unitree_G1_Inspire") -> dict:
    state_dict = {}
    for key, tactile_data in tactiles.items():
        tactile_dict = tactile_as_state(key, tactile_data, robot_type)
        state_dict.update(tactile_dict)
    return state_dict
