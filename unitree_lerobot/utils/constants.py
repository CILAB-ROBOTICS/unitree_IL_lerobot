
import dataclasses
from collections import OrderedDict
from typing import List, Dict

@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_state_data_name: List[str]
    json_action_data_name: List[str]


@dataclasses.dataclass(frozen=True)
class TactileRobotConfig(RobotConfig):
    tactiles: List[str]
    tactile_to_image_shape: Dict[str, tuple]  # Maps tactile names to their image shapes (height, width)
    tactile_to_state_indices: Dict[str, list]  # Maps tactile names to their indices


Z1_CONFIG = RobotConfig(
    motors=[
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_left_wrist' ,'color_2': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


Z1_SINGLE_CONFIG = RobotConfig(
    motors=[
        "kWaist",
        "kShoulder",
        "kElbow",
        "kForearmRoll",
        "kWristAngle",
        "kWristRotate",
        "kGripper",
    ],
    cameras=[
        "cam_high",
        "cam_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


G1_GRIPPER_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
        ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_INSPIRE_CONFIG = TactileRobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ],
    cameras=[
        "cam_left_high", "cam_third"
    ],
    tactiles=[
        "left_tactile_little_finger_tip",  # 3*3
        "left_tactile_little_finger_nail",  # 12*8
        "left_tactile_little_finger_pad",  # 10*8
        "left_tactile_ring_finger_tip",  # 3*3
        "left_tactile_ring_finger_nail",  # 12*8
        "left_tactile_ring_finger_pad",  # 10*8
        "left_tactile_middle_finger_tip",  # 3*3
        "left_tactile_middle_finger_nail",  # 12*8
        "left_tactile_middle_finger_pad",  # 10*8
        "left_tactile_index_finger_tip",  # 3*3
        "left_tactile_index_finger_nail",  # 12*8
        "left_tactile_index_finger_pad",  # 10*8
        "left_tactile_thumb_tip",  # 3*3
        "left_tactile_thumb_nail",  # 12*8
        "left_tactile_thumb_middle",  # 3*3
        "left_tactile_thumb_pad",  # 12*8
        "left_tactile_palm",  # 8*14
        "right_tactile_little_finger_tip",  # 3*3
        "right_tactile_little_finger_nail",  # 12*8
        "right_tactile_little_finger_pad",  # 10*8
        "right_tactile_ring_finger_tip",  # 3*3
        "right_tactile_ring_finger_nail",  # 12*8
        "right_tactile_ring_finger_pad",  # 10*8
        "right_tactile_middle_finger_tip",  # 3*3
        "right_tactile_middle_finger_nail",  # 12*8
        "right_tactile_middle_finger_pad",  # 10*8
        "right_tactile_index_finger_tip",  # 3*3
        "right_tactile_index_finger_nail",  # 12*8
        "right_tactile_index_finger_pad",  # 10*8
        "right_tactile_thumb_tip",  # 3*3
        "right_tactile_thumb_nail",  # 12*8
        "right_tactile_thumb_middle",  # 3*3
        "right_tactile_thumb_pad",  # 12*8
        "right_tactile_palm",  # 8*14
        "carpet_0",  # 32*32
    ],
    tactile_to_image_shape = OrderedDict({
        "left_tactile_little_finger_tip": (3, 3, 3),
        "left_tactile_little_finger_nail": (3, 12, 8),
        "left_tactile_little_finger_pad": (3, 10, 8),
        "left_tactile_ring_finger_tip": (3, 3, 3),
        "left_tactile_ring_finger_nail": (3, 12, 8),
        "left_tactile_ring_finger_pad": (3, 10, 8),
        "left_tactile_middle_finger_tip": (3, 3, 3),
        "left_tactile_middle_finger_nail": (3, 12, 8),
        "left_tactile_middle_finger_pad": (3, 10, 8),
        "left_tactile_index_finger_tip": (3, 3, 3),
        "left_tactile_index_finger_nail": (3, 12, 8),
        "left_tactile_index_finger_pad": (3, 10, 8),
        "left_tactile_thumb_tip": (3, 3, 3),
        "left_tactile_thumb_nail": (3, 12, 8),
        "left_tactile_thumb_middle": (3, 3, 3),
        "left_tactile_thumb_pad": (3, 12, 8),
        "left_tactile_palm": (3, 8, 14),
        "right_tactile_little_finger_tip": (3, 3, 3),
        "right_tactile_little_finger_nail": (3, 12, 8),
        "right_tactile_little_finger_pad": (3, 10, 8),
        "right_tactile_ring_finger_tip": (3, 3, 3),
        "right_tactile_ring_finger_nail": (3, 12, 8),
        "right_tactile_ring_finger_pad": (3, 10, 8),
        "right_tactile_middle_finger_tip": (3, 3, 3),
        "right_tactile_middle_finger_nail": (3, 12, 8),
        "right_tactile_middle_finger_pad": (3, 10, 8),
        "right_tactile_index_finger_tip": (3, 3, 3),
        "right_tactile_index_finger_nail": (3, 12, 8),
        "right_tactile_index_finger_pad": (3, 10, 8),
        "right_tactile_thumb_tip": (3, 3, 3),
        "right_tactile_thumb_nail": (3, 12, 8),
        "right_tactile_thumb_middle": (3, 3, 3),
        "right_tactile_thumb_pad": (3, 12, 8),
        "right_tactile_palm": (3, 8, 14),
        "carpet_0": (3, 32, 32),
    }),
    tactile_to_state_indices = OrderedDict({
        "left_tactile_little_finger_tip": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 3*3 -> 3*3
        "left_tactile_little_finger_nail": [43, 44, 45, 46, 51, 52, 53, 54, 59, 60, 61, 62, 67, 68, 69, 70],  # 12*8 -> 4*4
        "left_tactile_little_finger_pad": [131, 132, 133, 134, 139, 140, 141, 142, 147, 148, 149, 150, 155, 156, 157, 158],  # 10*8 -> 4*4
        "left_tactile_ring_finger_tip": [185, 186, 187, 188, 189, 190, 191, 192, 193],  # 3*3 -> 3*3
        "left_tactile_ring_finger_nail": [228, 229, 230, 231, 236, 237, 238, 239, 244, 245, 246, 247, 252, 253, 254, 255],  # 12*8 -> 4*4
        "left_tactile_ring_finger_pad": [316, 317, 318, 319, 324, 325, 326, 327, 332, 333, 334, 335, 340, 341, 342, 343],  # 10*8 -> 4*4
        "left_tactile_middle_finger_tip": [370, 371, 372, 373, 374, 375, 376, 377, 378],  # 3*3 -> 3*3
        "left_tactile_middle_finger_nail": [413, 414, 415, 416, 421, 422, 423, 424, 429, 430, 431, 432, 437, 438, 439, 440],  # 12*8 -> 4*4
        "left_tactile_middle_finger_pad": [501, 502, 503, 504, 509, 510, 511, 512, 517, 518, 519, 520, 525, 526, 527, 528],  # 10*8 -> 4*4
        "left_tactile_index_finger_tip": [555, 556, 557, 558, 559, 560, 561, 562, 563],  # 3*3 -> 3*3
        "left_tactile_index_finger_nail": [598, 599, 600, 601, 606, 607, 608, 609, 614, 615, 616, 617, 622, 623, 624, 625],  # 12*8 -> 4*4
        "left_tactile_index_finger_pad": [686, 687, 688, 689, 694, 695, 696, 697, 702, 703, 704, 705, 710, 711, 712, 713],  # 10*8 -> 4*4
        "left_tactile_thumb_tip": [740, 741, 742, 743, 744, 745, 746, 747, 748],  # 3*3 -> 3*3
        "left_tactile_thumb_nail": [783, 784, 785, 786, 791, 792, 793, 794, 799, 800, 801, 802, 807, 808, 809, 810],  # 12*8 -> 4*4
        "left_tactile_thumb_middle": [845, 846, 847, 848, 849, 850, 851, 852, 853],  # 3*3 -> 3*3
        "left_tactile_thumb_pad": [888, 889, 890, 891, 896, 897, 898, 899, 904, 905, 906, 907, 912, 913, 914, 915],  # 12*8 -> 4*4
        "left_tactile_little_knuckle": [963, 964, 971, 972],  # 8*14 -> 2*2
        "left_tactile_ring_knuckle": [987, 988, 995, 996],  # 8*14 -> 2*2
        "left_tactile_middle_knuckle": [1019, 1020, 1027, 1028],  # 8*14 -> 2*2
        "left_tactile_index_knuckle": [1043, 1044, 1051, 1052],  # 8*14 -> 2*2
        "left_tactile_palm": [1039, 1040, 1047, 1048],  # 8*14 -> 2*2
        "right_tactile_little_finger_tip": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 3*3 -> 3*3
        "right_tactile_little_finger_nail": [43, 44, 45, 46, 51, 52, 53, 54, 59, 60, 61, 62, 67, 68, 69, 70],  # 12*8 -> 4*4
        "right_tactile_little_finger_pad": [131, 132, 133, 134, 139, 140, 141, 142, 147, 148, 149, 150, 155, 156, 157, 158],  # 10*8 -> 4*4
        "right_tactile_ring_finger_tip": [185, 186, 187, 188, 189, 190, 191, 192, 193],  # 3*3 -> 3*3
        "right_tactile_ring_finger_nail": [228, 229, 230, 231, 236, 237, 238, 239, 244, 245, 246, 247, 252, 253, 254, 255],  # 12*8 -> 4*4
        "right_tactile_ring_finger_pad": [316, 317, 318, 319, 324, 325, 326, 327, 332, 333, 334, 335, 340, 341, 342, 343],  # 10*8 -> 4*4
        "right_tactile_middle_finger_tip": [370, 371, 372, 373, 374, 375, 376, 377, 378],  # 3*3 -> 3*3
        "right_tactile_middle_finger_nail": [413, 414, 415, 416, 421, 422, 423, 424, 429, 430, 431, 432, 437, 438, 439, 440],  # 12*8 -> 4*4
        "right_tactile_middle_finger_pad": [501, 502, 503, 504, 509, 510, 511, 512, 517, 518, 519, 520, 525, 526, 527, 528],  # 10*8 -> 4*4
        "right_tactile_index_finger_tip": [555, 556, 557, 558, 559, 560, 561, 562, 563],  # 3*3 -> 3*3
        "right_tactile_index_finger_nail": [598, 599, 600, 601, 606, 607, 608, 609, 614, 615, 616, 617, 622, 623, 624, 625],  # 12*8 -> 4*4
        "right_tactile_index_finger_pad": [686, 687, 688, 689, 694, 695, 696, 697, 702, 703, 704, 705, 710, 711, 712, 713],  # 10*8 -> 4*4
        "right_tactile_thumb_tip": [740, 741, 742, 743, 744, 745, 746, 747, 748],  # 3*3 -> 3*3
        "right_tactile_thumb_nail": [783, 784, 785, 786, 791, 792, 793, 794, 799, 800, 801, 802, 807, 808, 809, 810],  # 12*8 -> 4*4
        "right_tactile_thumb_middle": [845, 846, 847, 848, 849, 850, 851, 852, 853],  # 3*3 -> 3*3
        "right_tactile_thumb_pad": [888, 889, 890, 891, 896, 897, 898, 899, 904, 905, 906, 907, 912, 913, 914, 915],  # 12*8 -> 4*4
        "right_tactile_little_knuckle": [963, 964, 971, 972],  # 8*14 -> 2*2
        "right_tactile_ring_knuckle": [987, 988, 995, 996],  # 8*14 -> 2*2
        "right_tactile_middle_knuckle": [1019, 1020, 1027, 1028],  # 8*14 -> 2*2
        "right_tactile_index_knuckle": [1043, 1044, 1051, 1052],  # 8*14 -> 2*2
        "right_tactile_palm": [1039, 1040, 1047, 1048],  # 8*14 -> 2*2
    }),
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_3':'cam_third',},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


ROBOT_CONFIGS = {
    "Unitree_Z1_Single": Z1_SINGLE_CONFIG,
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Gripper": G1_GRIPPER_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
    "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
}