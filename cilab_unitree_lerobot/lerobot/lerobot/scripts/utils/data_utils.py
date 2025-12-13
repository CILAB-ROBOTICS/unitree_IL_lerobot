import torch
import torch.utils.data


def data_split(dataset, val_ratio=0.2):
    episode_data_index = dataset.episode_data_index
    num_episodes = len(episode_data_index["from"])
    all_episodes = torch.randperm(num_episodes).tolist()
    
    val_size = int(num_episodes * val_ratio)
    train_size = num_episodes - val_size
    
    train_episodes = all_episodes[:train_size]
    val_episodes = all_episodes[train_size:]
    
    train_indices = []
    for ep_idx in train_episodes:
        start = episode_data_index["from"][ep_idx].item()
        end = episode_data_index["to"][ep_idx].item()
        train_indices.extend(range(start, end))
        
    val_indices = []
    for ep_idx in val_episodes:
        start = episode_data_index["from"][ep_idx].item()
        end = episode_data_index["to"][ep_idx].item()
        val_indices.extend(range(start, end))
        
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

def time_stacking(x):
    if x.ndim in (5, 3):
        return x.mean(dim=1)
    return x