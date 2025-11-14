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

@parser.wrap()
def pretrain(cfg: TrainPipelineConfig):

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
    #cam_features = [k.split(".")[-1] for k in camera_keys]
    #logging.info(f"Camera features: {cam_features}")
    #tac_features = [k.split(".")[-1] for k in tactile_keys]
    #logging.info(f"Tactile features: {tac_features}")

    clip_dim: int = 512
    features_per_group: int = 16
    resnet_lr: float = 1e-5
    projection_lr: float = 1e-4
    n_epochs: int = 1000
    n_cameras: int = len(camera_keys)
    plot_freq: int = 50

    vision_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    tactile_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
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

        dataloader = torch.utils.data.DataLoader(dataset)
        
        for batch_idx, batch in enumerate(dataloader):
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
                '''
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
                '''
            else:
                loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)

            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

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


def clip_loss(image_embeddings:torch.Tensor, tactile_embeddings:torch.Tensor, target_matrix:torch.Tensor, logit_scale = 1.0, visualize = False):
    # image_embeddings: batch, clip_N, camera, clip_dim
    # gelsight_embeddings: batch, clip_N, clip_dim
    # same as clip_loss, but vectorized
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

    '''
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
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
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)


        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
    '''
    
if __name__ == "__main__":
    init_logging()
    pretrain()
    logging.info("All done!")