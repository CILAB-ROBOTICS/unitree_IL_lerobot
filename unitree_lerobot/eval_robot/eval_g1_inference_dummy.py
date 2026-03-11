"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
#from unitree_lerobot.eval_robot.make_robot import (
#    setup_image_client,
#    setup_robot_interface,
#    process_images_and_observations,
#) --- IGNORE ---
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.utils.constants import ROBOT_CONFIGS
from unitree_lerobot.utils.preprocess_tactile_signal import parse_tactile_as_image, parse_tactile_as_state
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

#logging_mp.basicConfig(level=logging_mp.INFO)
#logger_mp = logging_mp.getLogger(__name__)


def keep_single_camera_observation(
    observation: dict[str, Any],
    camera_key: str = "observation.images.cam_left_high",
) -> dict[str, Any]:
    """Keep only one head/wrist camera image key and preserve non-camera keys."""
    filtered_observation: dict[str, Any] = {}
    for key, value in observation.items():
        if key.startswith("observation.images.cam_") and key != camera_key:
            continue
        filtered_observation[key] = value
    return filtered_observation


def apply_rename_map_to_observation(
    observation: dict[str, Any],
    rename_map: dict[str, str],
) -> dict[str, Any]:
    """Rename observation keys at runtime so policy expected feature names are present."""
    renamed_observation = dict(observation)
    for source_key, target_key in rename_map.items():
        if source_key in renamed_observation:
            renamed_observation[target_key] = renamed_observation.pop(source_key)
    return renamed_observation


def align_action_stats_to_policy(
    dataset_stats: dict[str, dict[str, Any]],
    target_action_dim: int,
) -> dict[str, dict[str, Any]]:
    """Slice action stats to match policy action dimension when they differ."""
    action_stats = dataset_stats.get("action")
    if action_stats is None:
        return dataset_stats

    for stat_name, stat_value in action_stats.items():
        if hasattr(stat_value, "shape") and len(stat_value.shape) == 1 and stat_value.shape[0] > target_action_dim:
            action_stats[stat_name] = stat_value[:target_action_dim]

    return dataset_stats


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    #logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # Tactile configuration
    eval_config = {
        'tactile_enc_type': "image",  # or "state"
    } if cfg.tactile_use else {}

    robot_config = ROBOT_CONFIGS[
        f"Unitree"
        f"_G1"
        f"_Inspire"
    ]

    tactile_names = getattr(robot_config, "tactiles", []) if cfg.tactile_use else []

    # Tactile preprocessing function
    tactile_preprocess_fn = lambda x: x  # Placeholder
    if cfg.tactile_use:
        if eval_config['tactile_enc_type'] == "image":
            tactile_preprocess_fn = parse_tactile_as_image
        elif eval_config['tactile_enc_type'] == "state":
            tactile_preprocess_fn = parse_tactile_as_state

    # Dummy mode: simulate without real hardware
    if getattr(cfg, 'dummy', False):
        #logger_mp.info("Running in dummy mode - no real hardware interaction")
        # Simulate setup
        arm_dof = 14
        ee_dof = 6 if cfg.ee == "inspire1" else 0
        # Get initial pose from dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = "s"  # Auto-start in dummy mode
        idx = 0
        print(f"user_input: {user_input}")
        if user_input.lower() == "s":
            #logger_mp.info("Dummy: Initializing robot to starting pose...")
            # Skip real initialization
            #logger_mp.info("Dummy: Starting evaluation loop.")
            for idx in range(10):  # Run 10 dummy steps
                loop_start_time = time.perf_counter()
                # 1. Generate dummy observations
                # Dummy images (random)
                dummy_tv_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                #dummy_wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) if True else None  # Assume has_wrist_cam

                observation = {
                    "observation.images.cam_left_high": torch.from_numpy(dummy_tv_image),
                    #"observation.images.cam_right_high": torch.from_numpy(dummy_tv_image) if True else None,  # Assume binocular
                    #"observation.images.cam_left_wrist": torch.from_numpy(dummy_wrist_image) if dummy_wrist_image is not None else None,
                    #"observation.images.cam_right_wrist": torch.from_numpy(dummy_wrist_image) if dummy_wrist_image is not None else None,
                }

                # Dummy arm state
                current_arm_q = np.random.randn(arm_dof).astype(np.float32)
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    left_ee_state = np.random.randn(ee_dof).astype(np.float32)
                    right_ee_state = np.random.randn(ee_dof).astype(np.float32)
                    # Dummy tactile data
                    if cfg.tactile_use and cfg.ee == "inspire1":
                        left_touch = np.random.randn(1062).astype(np.float32)
                        right_touch = np.random.randn(1062).astype(np.float32)
                        if eval_config['tactile_enc_type'] == "image":
                            tactile_images = tactile_preprocess_fn({
                                "left_tactile": left_touch,
                                "right_tactile": right_touch,
                            })
                            for tac_name in tactile_names:
                                observation[f"observation.images.{tac_name}"] = torch.from_numpy(tactile_images[tac_name])
                        elif eval_config['tactile_enc_type'] == "state":
                            tactiles = tactile_preprocess_fn({
                                "left_tactile": left_touch,
                                "right_tactile": right_touch,
                            })
                            state_tactiles = torch.concatenate([
                                torch.from_numpy(tactiles.pop(key)) for key in sorted(tactiles.keys())
                            ], axis=-1)

                observation = keep_single_camera_observation(observation)
                observation = apply_rename_map_to_observation(observation, cfg.rename_map)

                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                if cfg.tactile_use and cfg.ee == "inspire1" and eval_config.get('tactile_enc_type') == "state" and "state_tactiles" in locals():
                    state_tensor = torch.concatenate([state_tensor, state_tactiles], axis=-1)
                observation["observation.state"] = state_tensor

                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()
                #logger_mp.info(f"Dummy step {idx}: Action shape {action_np.shape}, mean {action_np.mean():.4f}")

                # 3. Dummy action execution (just log)
                arm_action = action_np[:arm_dof]
                #logger_mp.info(f"Dummy: Arm action - mean {arm_action.mean():.4f}")

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    #logger_mp.info(f"Dummy: EE actions - left mean {left_ee_action.mean():.4f}, right mean {right_ee_action.mean():.4f}")

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Simulate frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
        #logger_mp.info("Dummy evaluation completed")
        return

    # Original code for real hardware
    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            #.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            #logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                        # Tactile data for inspire hand
                        if cfg.tactile_use and cfg.ee == "inspire1" and "touch" in ee_shared_mem:
                            full_touch = np.array(ee_shared_mem["touch"][:])
                            left_touch = full_touch[:1062]
                            right_touch = full_touch[1062:]
                            if eval_config['tactile_enc_type'] == "image":
                                tactile_images = tactile_preprocess_fn({
                                    "left_tactile": left_touch,
                                    "right_tactile": right_touch,
                                })
                                for tac_name in tactile_names:
                                    observation[f"observation.images.{tac_name}"] = torch.from_numpy(tactile_images[tac_name])
                            elif eval_config['tactile_enc_type'] == "state":
                                tactiles = tactile_preprocess_fn({
                                    "left_tactile": left_touch,
                                    "right_tactile": right_touch,
                                })
                                state_tactiles = torch.concatenate([
                                    torch.from_numpy(tactiles.pop(key)) for key in sorted(tactiles.keys())
                                ], axis=-1)
                observation = keep_single_camera_observation(observation)
                observation = apply_rename_map_to_observation(observation, cfg.rename_map)
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                if cfg.tactile_use and cfg.ee == "inspire1" and eval_config.get('tactile_enc_type') == "state" and "state_tactiles" in locals():
                    state_tensor = torch.concatenate([state_tensor, state_tactiles], axis=-1)
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()
                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()

    # When loading pretrained processors, keep their saved normalization stats.
    # Overriding with dataset stats can cause action-dimension mismatch (e.g., 6 vs 26).
    dataset_stats = rename_stats(dataset.meta.stats, cfg.rename_map)

    action_feature = policy.config.output_features.get("action")
    if action_feature is not None:
        dataset_stats = align_action_stats_to_policy(dataset_stats, int(action_feature.shape[0]))

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=None,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()

"""
python unitree_lerobot/eval_robot/eval_g1_inference_dummy.py \
    --policy.path=eunjuri/pi0_training_pick_and_place_full_fullpft \
    --repo_id=eunjuri/pick_and_place_full \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="inspire1" \
    --tactile_use=true \
    --dummy=true \
    --visualization=true
"""