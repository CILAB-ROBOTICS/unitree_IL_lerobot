#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser

import torchvision
from torch import nn
import torch
from typing import Tuple, Dict, Union, Callable, List
from torch.utils.data import DataLoader
import h5py
import os
import cv2
from torchvision.transforms import Normalize
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import get_norm_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# the projection head for CLIP. I'm using resnet's approach of an average pooling layer followed by a linear layer.
class ClipProjectionHead(nn.Module):
    def __init__(self, out_dim: int, conditioning_dim: int = 0, num_channels:int = 512, normailize: bool = True):
        """
        Create a projection head for CLIP. The projection head consists of an 
        average pooling layer followed by a linear layer.
        out_dim: The output dimension of the linear layer.
        conditioning_dim: The dimension of the conditioning vector. If 0, no conditioning is used.
        num_channels: The number of channels in the feature map.
        normailize: If true, the output of the linear layer is normalized. (default: True)
        """

        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(num_channels + conditioning_dim, out_dim)
        self.normalize = normailize
    
    def forward(self, feature_map, conditioning=None) -> torch.Tensor:
        x = self.pooling(feature_map)
        x = self.flatten(x)
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=-1)

        x = self.linear(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x

def modified_resnet18(weights=None, features_per_group=16) -> nn.Module:
    """
    Get a resnet18 model with all BatchNorm layers replaced with GroupNorm.
    weights: The weights to load into the model. If None, uses default pretraiend weights.
    features_per_group: The number of features per group in the GroupNorm layer.
    return: The modified resnet18 model."""
    # get a resnet18 model
    resnet18 = getattr(torchvision.models, 'resnet18')()

    # remove the final fully connected layer and average pooling
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

    # replace all BatchNorm with GroupNorm
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)
    return resnet18   

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 episode_ids: List[int], 
                 dataset_dir: str, 
                 camera_names: List[str], 
                 norm_stats: Dict[str, Union[float, np.ndarray]],
                 image_size: Tuple[int, int] = None, 
                 tactile_size: Tuple[int, int] = None,
                 min_distance = 5,
                 n_images = 10,
                 is_cluster=False):
        
        if is_cluster:
            # pre-load all the data into memory
            self.episodes = {}
            for episode_id in self.episode_ids:
                dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
                with h5py.File(dataset_path, 'r') as root:
                    all_cam_images = []
                    all_gelsight_images = []
                    all_positions = []
                    for timestep in range(self.episode_lengths[episode_id]):
                        # get camera images
                        timestep_cam_images = []

                        for cam_name in self.camera_names:
                            image = root[f'/observations/images/{cam_name}'][timestep]
                            
                            # convert to tensor
                            image = torch.tensor(image, dtype=torch.float32)/255.0
                            image = torch.einsum('h w c -> c h w', image)

                            # normalize image
                            image = self.image_normalize(image)
                            timestep_cam_images.append(image)

                        images = torch.stack(timestep_cam_images, axis=0)

                        # get gelsight data
                        gelsight_data = root['observations/gelsight/depth_strain_image'][timestep]

                        # resize gelsight data
                        if self.gelsight_size != gelsight_data.shape[:2]:
                            raise ValueError("Image size must be the same for all cameras")
                            gelsight_data = cv2.resize(gelsight_data, (self.gelsight_size[1], self.gelsight_size[0]))
                        
                        # convert to tensor
                        gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                        gelsight_data = torch.einsum('h w c -> c h w', gelsight_data) # change to c h w

                        # normalize gelsight data
                        gelsight_data = self.gelsight_normalize(gelsight_data)

                        # get qpos and normalize
                        position = root['observations/qpos'][timestep]
                        position = (position - self.position_mean) / self.position_std

                        # don't include the last element, which is the gripper
                        position = torch.tensor(position[:3], dtype=torch.float32)

                        all_cam_images.append(images)
                        all_gelsight_images.append(gelsight_data)
                        all_positions.append(position)
                    
                self.episodes[episode_id] = (torch.stack(all_cam_images, axis=0), torch.stack(all_gelsight_images, axis=0), torch.stack(all_positions, axis=0))

def clip_pretraining(train_dataset,
                     test_dataset,
                     device: torch.device,
                     save_dir: str,
                     save_freq: int = 100,
                     plot_freq: int = 50,
                     n_epochs: int = 10,
                     clip_dim: int = 512,
                     features_per_group: int = 16,
                     resnet_lr: float = 1e-5,
                     projection_lr: float = 1e-4):

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    # get the camera, gelsight, and state dimensions from the dataset
    features = train_dataset.meta.features
    camera_keys = [k for k in features if k.startswith("observation.images.") and "tactile" not in k and not k.endswith("carpet_0")]
    tactile_keys = [k for k in features if k.startswith("observation.images.") and "tactile" in k or k.endswith("carpet_0")]
    n_cameras = 1 #len(camera_keys)

    #camera_sizes = [dataset.image_size]*n_cameras
    #gelsight_size = dataset.gelsight_size
    state_size = 3
    
    # get resnet models for each camera
    vision_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    # create a projection head
    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    # get resnet models for each tactile
    tactile_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    # create a projection head
    tactile_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    optim_params = [{"params": tactile_encoder.parameters(), "lr": resnet_lr},
                    {"params": tactile_projection.parameters(), "lr": projection_lr},
                    {"params": vision_encoder.parameters(), "lr": resnet_lr},
                    {"params": vision_projection.parameters(), "lr": projection_lr},]

    print('optim_params:', optim_params)

    optimizer = torch.optim.Adam(optim_params)
    
    training_losses = np.empty([n_epochs, n_cameras])
    testing_losses = np.empty([n_epochs, n_cameras])

    for epoch in tqdm(range(n_epochs)):
    # train the model
        training_loss = np.zeros(n_cameras)

        tactile_encoder.train()
        tactile_projection.train()
        vision_encoder.train()
        vision_projection.train()

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        for batch_idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                #print('Batch key:', k, 'Value shape/type:', v.shape if isinstance(v, torch.Tensor) else type(v))
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            camera_keys = [k for k in camera_keys if "cam_left_high" not in k]
            cam_tensors = [batch[k] for k in camera_keys] # n_cameras, C, H, W
            images = torch.stack(cam_tensors, dim=1) # B, N, C, H, W
            print(images.shape)

            tactile_keys = [k for k in tactile_keys if "carpet_0" in k]
            tac_tensors = [batch[k] for k in tactile_keys]  
            tactiles = torch.stack(tac_tensors, dim=1) # 1, 1, 3, 32, 32
            print(tactiles.shaepe)

            images = images.to(device)
            tactiles = tactiles.to(device)

            B, n_cameras, C, H, W = images.shape
            images = images.view(-1, C, H, W)
            image_embeddings = vision_projection(vision_encoder(images))
            image_embeddings = image_embeddings.view(B, n_cameras, clip_dim) # 1, 1, 512

            B, n_tactile, C, H, W = tactiles.shape
            tactiles = tactiles.view(-1, C, H, W)
            tactile_embeddings = tactile_projection(vision_encoder(tactiles))
            tactile_embeddings = tactile_embeddings.view(B, n_tactile, clip_dim) # 1, 1, 512

            batch_size = images.shape[0]
            clip_N = 2

            # calculate target matrix
            target_matrix = torch.eye(clip_N).to(device)

            if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
                loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)

                try:
                    for cam_num, plot_map in enumerate(plot_maps):
                        plt.figure()
                        plt.imshow(plot_map)
                        plt.colorbar()
                        plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Train')
                        plt.close()
                except:
                    print('Error in train plots')
                    raise

            else:
                loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)

            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

def run_clip_pretraining():
    save_dir = "data/clip_models/"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset_id = "eunjuri/towel_strong_train_full"
    test_dataset_id = "eunjuri/towel_strong_test_full"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_train = 3
    batch_size_test = 3

    train_dataset = LeRobotDataset(repo_id=train_dataset_id)
    test_dataset  = LeRobotDataset(repo_id=test_dataset_id)

    # create directory to save models and plots
    # get all folders in the clip_models directory
    ns = [-1]
    for folder in os.listdir(save_dir):
        ns.append(int(folder))

    n = max(ns) + 1
    os.makedirs(f'{save_dir}/{n}')
    os.makedirs(f'{save_dir}/{n}/graphs')

    # save run stats:
    with open(f'{save_dir}/{n}/run_stats.txt', 'w') as f:
        f.write(f'train_dataset: {train_dataset}\n')
        f.write(f'test_dataset: {test_dataset}\n')

    clip_pretraining(train_dataset, test_dataset, device, save_dir=f'{save_dir}/{n}', clip_dim=512, features_per_group=16, n_epochs=1501)

'''
@parser.wrap()
def clip_pretraining(cfg: TrainPipelineConfig):

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    #logging.info(f"Dataset info:\n{pformat(dataset.meta)}")

    features = dataset.meta.features
    camera_keys = [k for k in features if k.startswith("observation.images.") and "tactile" not in k and not k.endswith("carpet_0")]
    tactile_keys = [k for k in features if k.startswith("observation.images.") and "tactile" in k or k.endswith("carpet_0")]

    clip_dim: int = 512
    features_per_group: int = 16
    resnet_lr: float = 1e-5
    projection_lr: float = 1e-4
    n_epochs: int = 1000
    n_cameras: int = len(camera_keys)
    plot_freq: int = 50

    # get resnet models for each camera
    vision_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    # create a projection head
    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    # get resnet models for each tactile
    tactile_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    # create a projection head
    tactile_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    optim_params = [{"params": tactile_encoder.parameters(), "lr": resnet_lr},
                    {"params": tactile_projection.parameters(), "lr": projection_lr},
                    {"params": vision_encoder.parameters(), "lr": resnet_lr},
                    {"params": vision_projection.parameters(), "lr": projection_lr},]

    print('optim_params:', optim_params)

    optimizer = torch.optim.Adam(optim_params)
    
    training_losses = np.empty([n_epochs, n_cameras])
    testing_losses = np.empty([n_epochs, n_cameras])

    shuffle = True
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    print(dataset.num_episodes)

    for epoch in tqdm(range(n_epochs)):
    # train the model
        training_loss = np.zeros(n_cameras)

        tactile_encoder.train()
        tactile_projection.train()
        vision_encoder.train()
        vision_projection.train(
        )

        for batch_idx, batch in enumerate(dataloader):
            print('Batch idx:', batch_idx)
            print('Batch keys:', batch.keys())

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            camera_keys = [k for k in camera_keys if "cam_left_high" not in k]
            cam_tensors = [batch[k] for k in camera_keys] # 1, 3, 480, 640
            images = torch.stack(cam_tensors, dim=1) # 1, 1, 3, 480, 640 (B, N, C, H, W)

            tactile_keys = [k for k in tactile_keys if "carpet_0" in k]
            tac_tensors = [batch[k] for k in tactile_keys]  
            tactiles = torch.stack(tac_tensors, dim=1) # 1, 1, 3, 32, 32

            images = images.to(device)
            tactiles = tactiles.to(device)

            B, n_cameras, C, H, W = images.shape
            images = images.view(-1, C, H, W)
            image_embeddings = vision_projection(vision_encoder(images))
            image_embeddings = image_embeddings.view(B, n_cameras, clip_dim) # 1, 1, 512

            B, n_tactile, C, H, W = tactiles.shape
            tactiles = tactiles.view(-1, C, H, W)
            tactile_embeddings = tactile_projection(vision_encoder(tactiles))
            tactile_embeddings = tactile_embeddings.view(B, n_tactile, clip_dim) # 1, 1, 512

            batch_size = images.shape[0]
            clip_N = 2

            # calculate target matrix
            target_matrix = torch.eye(clip_N).to(device)

            if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
                loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)

                try:
                    for cam_num, plot_map in enumerate(plot_maps):
                        plt.figure()
                        plt.imshow(plot_map)
                        plt.colorbar()
                        plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Train')
                        plt.close()
                except:
                    print('Error in train plots')
                    raise

            else:
                loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)

            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
'''

def clip_loss(image_embeddings:torch.Tensor, tactile_embeddings:torch.Tensor, target_matrix:torch.Tensor, logit_scale = 1.0, visualize = False):
    n_cameras = image_embeddings.shape[1]
    batch_size = image_embeddings.shape[0]

    visualizations = []
    image_embeddings = image_embeddings.squeeze(1)      # (B, D)
    tactile_embeddings = tactile_embeddings.squeeze(1)    # (B, D)

    image_logits   = logit_scale * (image_embeddings @ tactile_embeddings.T)  # (B, B)
    tactile_logits = logit_scale * (tactile_embeddings @ image_embeddings.T)  # (B, B)

    if visualize:
        visualizations = image_logits[0].clone().detach().cpu().numpy()/logit_scale
    
    device = image_embeddings.device
    B = image_embeddings.shape[0]
    targets = torch.arange(image_embeddings.shape[0], device=device)

    # flatten the batch and camera dimensions, then calculate the loss
    image_logits = image_logits.reshape(B, B)
    tactile_logits = tactile_logits.reshape(B, B)

    # need to make the target matrix B, N, N
    image_loss = F.cross_entropy(image_logits, targets)
    tactile_loss = F.cross_entropy(tactile_logits, targets)

    loss = ((image_loss + tactile_loss)/2.0).mean(dim=0)

    return loss, visualizations

    
if __name__ == "__main__":
    init_logging()
    run_clip_pretraining()
    logging.info("All done!")