
import torch
import sys
import os

# Add the workspace to the path so we can import lerobot
sys.path.append(os.getcwd())

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

def investigate_loss():
    # Configuration
    # User specified: task_index=0 -> [0, 1], task_index=1 -> [1, 0]
    # This implies dimension is 2.
    object_condition_dim = 2
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=10,
        n_action_steps=10,
        input_shapes={
            "observation.images.cam1": [3, 96, 96],
            "observation.state": [6],
        },
        output_shapes={
            "action": [6],
        },
        object_condition_feature=object_condition_dim,
        use_vae=True, 
    )

    # Dummy dataset stats for normalization
    dataset_stats = {
        "observation.images.cam1": {"mean": torch.zeros(3, 96, 96), "std": torch.ones(3, 96, 96)},
        "observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
        "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
    }

    # Initialize Policy
    policy = ACTPolicy(config, dataset_stats)
    policy.train() 

    # Create dummy batch
    batch_size = 2
    device = torch.device("cpu")
    
    batch = {
        "observation.images.cam1": torch.randn(batch_size, 3, 96, 96),
        "observation.state": torch.randn(batch_size, 6),
        "action": torch.randn(batch_size, 10, 6),
        "action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool),
    }

    # Case 1: With Task Index (Automatic Conversion)
    print("\n--- Case 1: With Task Index (Automatic Conversion) ---")
    # task_index = 0 -> Should become [0, 1]
    # task_index = 1 -> Should become [1, 0]
    
    task_indices = torch.tensor([0, 1], dtype=torch.long)
    
    batch_with_index = batch.copy()
    batch_with_index["task_index"] = task_indices
    # Ensure no explicit object condition is passed
    if "observation.object_condition" in batch_with_index:
        del batch_with_index["observation.object_condition"]
    
    print(f"Input Task Indices: {task_indices}")
    
    loss_1, loss_dict_1 = policy(batch_with_index)
    print(f"Total Loss: {loss_1.item():.6f}")
    print(f"L1 Loss: {loss_dict_1['l1_loss']:.6f}")

    # Case 2: Without Object Condition (Implicit Zeros)
    print("\n--- Case 2: Without Object Condition (Implicit Zeros) ---")
    batch_without_condition = batch.copy()
    if "observation.object_condition" in batch_without_condition:
        del batch_without_condition["observation.object_condition"]
        
    loss_2, loss_dict_2 = policy(batch_without_condition)
    print(f"Total Loss: {loss_2.item():.6f}")
    print(f"L1 Loss: {loss_dict_2['l1_loss']:.6f}")

    print("\n--- Comparison ---")
    diff = abs(loss_dict_1['l1_loss'] - loss_dict_2['l1_loss'])
    print(f"Difference in L1 Loss: {diff:.6f}")

    
    if diff > 1e-6:
        print("Result: Object condition AFFECTS the loss (as expected).")
    else:
        print("Result: Object condition DOES NOT affect the loss (unexpected if weights are initialized).")

if __name__ == "__main__":
    investigate_loss()
