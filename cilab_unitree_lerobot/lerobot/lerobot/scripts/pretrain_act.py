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
from pprint import pformat

import einops
import torch
import torch.nn as nn
from termcolor import colored

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


def extract_embeddings(policy, batch, device):
    policy.eval()

    tactile_backbone = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # downsample but keep spatial
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 512, kernel_size=1),  # project to model dim
                        # NO AdaptiveAvgPool2d((1,1)) - preserve spatial information!
                    )
    tactile_backbone = tactile_backbone.to(device)
    with torch.no_grad():
        batch = policy.normalize_inputs(batch)
        if policy.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in policy.config.image_features]

        model = policy.model
        batch_size = batch["observation.state"].shape[0]

        latent_sample = torch.zeros([batch_size, model.config.latent_dim], dtype=torch.float32).to(device)

        tokens_1d = [model.encoder_latent_input_proj(latent_sample)]
        if model.config.robot_state_feature:
            tokens_1d.append(model.encoder_robot_state_input_proj(batch["observation.state"]))
        if model.config.env_state_feature:
            tokens_1d.append(model.encoder_env_state_input_proj(batch["observation.environment_state"]))

        tokens_1d = torch.stack(tokens_1d, dim=0)  # (N1D, B, D)
        pos_embed_1d = model.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (N1D, 1, D)

        tokens_2d = None
        pos_embed_2d = None
        if model.config.image_features and "observation.images" in batch:
            all_2d_features = []
            all_2d_pos_embeds = []
            for img_key, img in zip(model.config.image_features, batch["observation.images"]):
                if "tactile" in img_key and tactile_backbone is not None:
                    tac_features = tactile_backbone(img)
                    tac_pos_embed = model.encoder_cam_feat_pos_embed(tac_features).to(dtype=tac_features.dtype)
                    
                    tac_features = einops.rearrange(tac_features, "b c h w -> (h w) b c")
                    tac_pos_embed = einops.rearrange(tac_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(tac_features)
                    all_2d_pos_embeds.append(tac_pos_embed)
                
                elif "tactile" not in img_key and hasattr(model, "backbone"):
                    cam_features = model.backbone(img)["feature_map"]
                    cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = model.encoder_img_feat_input_proj(cam_features)

                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(cam_features)
                    all_2d_pos_embeds.append(cam_pos_embed)

            if all_2d_features:
                tokens_2d = torch.cat(all_2d_features, dim=0)
                pos_embed_2d = torch.cat(all_2d_pos_embeds, dim=0)

        if tokens_2d is not None and tokens_2d.numel() > 0:
            encoder_tokens = torch.cat([tokens_1d, tokens_2d], dim=0)
            encoder_pos = torch.cat([pos_embed_1d, pos_embed_2d], dim=0)
        else:
            encoder_tokens = tokens_1d
            encoder_pos = pos_embed_1d

        return encoder_tokens, encoder_pos


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Creating policy (for backbone & normalization)")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy.to(device)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")

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

    logging.info("Start extracting embeddings...")

    for step in range(cfg.steps):
        batch = next(dl_iter)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        encoder_tokens, encoder_pos = extract_embeddings(policy, batch, device)

        if step % 10 == 0:
            logging.info(f"Step {step}: Embeddings Shape: {encoder_tokens.shape}, Pos Embeds Shape: {encoder_pos.shape}")

    logging.info("End of extraction")


if __name__ == "__main__":
    init_logging()
    train()