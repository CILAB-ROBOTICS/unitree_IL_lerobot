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

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import wandb
from tqdm import tqdm

import einops
import torch
import torch.nn as nn
from termcolor import colored

from lerobot.common.datasets.factory import make_dataset
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
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.cm as cm


def extract_embeddings(policy, batch, device):
    policy.eval()
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
    # policy.eval() # Removed: extract_embeddings is used in train loop, so we need gradients.
    # with torch.no_grad(): # Removed: extract_embeddings is used in train loop, so we need gradients.
    batch = policy.normalize_inputs(batch)
    if policy.config.image_features:
        # The original code had `batch["observation.images"] = [batch[key] for key in policy.config.image_features]`
        # This is not needed if we iterate directly over `policy.config.image_features` and access `batch[img_key]`.
        pass

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

    # tactile_backbone is defined locally here, so it needs to be passed or accessed.
    # For simplicity, let's assume it's part of the model or defined globally if needed.
    # The original code defined it here, so we keep it.
    tactile_backbone = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, policy.config.dim_model, kernel_size=1),
    ).to(device)

    tokens_2d = None
    pos_embed_2d = None
    carpet_embeddings = None
    third_embeddings = None # Initialize third_embeddings

    if model.config.image_features:
        all_2d_features = []
        all_2d_pos_embeds = []
        carpet_features = []
        third_features = []

        for img_key in model.config.image_features:
            if img_key not in batch:
                continue

            img = batch[img_key]

            # Check for 5D (B, T, C, H, W)
            is_temporal = img.ndim == 5
            B = img.shape[0]
            T = img.shape[1] if is_temporal else 1

            if is_temporal:
                img = einops.rearrange(img, "b t c h w -> (b t) c h w")

            features = None
            pos_embed = None

            if "tactile" in img_key and tactile_backbone is not None:
                features = tactile_backbone(img)
                pos_embed = model.encoder_cam_feat_pos_embed(features).to(dtype=features.dtype)
            elif "cam_left_high" in img_key and hasattr(model, "backbone"):
                features = model.backbone(img)["feature_map"]
                pos_embed = model.encoder_cam_feat_pos_embed(features).to(dtype=features.dtype)
                features = model.encoder_img_feat_input_proj(features)
            elif "cam_third" in img_key and hasattr(model, "backbone"):
                features = model.backbone(img)["feature_map"]
                features = model.encoder_img_feat_input_proj(features)
                # Original code did not have pos_embed for cam_third
            elif "carpet" in img_key and hasattr(model, "backbone"):
                features = model.backbone(img)["feature_map"]
                features = model.encoder_img_feat_input_proj(features)
                # Original code did not have pos_embed for carpet

            # Post-process features
            if features is not None:
                features = einops.rearrange(features, "b c h w -> (h w) b c") # (L, B*T, D)

                if is_temporal:
                    # Reshape to (L, B, T, D) and mean pool over time
                    features = einops.rearrange(features, "l (b t) d -> l b t d", b=B, t=T)
                    features = features.mean(dim=2) # (L, B, D)

                if pos_embed is not None:
                    pos_embed = einops.rearrange(pos_embed, "b c h w -> (h w) b c") # (L, B*T, D)
                    if is_temporal:
                        pos_embed = einops.rearrange(pos_embed, "l (b t) d -> l b t d", b=B, t=T)
                        pos_embed = pos_embed.mean(dim=2) # (L, B, D)

                # Append to lists based on img_key
                if "cam_left_high" in img_key or "tactile" in img_key:
                    all_2d_features.append(features)
                    if pos_embed is not None:
                        all_2d_pos_embeds.append(pos_embed)
                elif "cam_third" in img_key:
                    third_features.append(features)
                elif "carpet" in img_key:
                    carpet_features.append(features)

        if all_2d_features:
            tokens_2d = torch.cat(all_2d_features, dim=0)
            pos_embed_2d = torch.cat(all_2d_pos_embeds, dim=0)

        if third_features:
            third_embeddings = torch.cat(third_features, dim=0)

        if carpet_features:
            carpet_embeddings = torch.cat(carpet_features, dim=0)

    if tokens_2d is not None and tokens_2d.numel() > 0:
        ego_embedding = torch.cat([tokens_1d, tokens_2d], dim=0)
        encoder_pos = torch.cat([pos_embed_1d, pos_embed_2d], dim=0)
    else:
        ego_embedding = tokens_1d
        encoder_pos = pos_embed_1d

    return ego_embedding, third_embeddings, carpet_embeddings


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def contrastive_loss(emb1, emb2, temperature=0.07):
    """
    Computes InfoNCE loss (CLIP-style) between two batches of embeddings.
    emb1, emb2: (Batch_Size, Dim)
    """
    # Normalize embeddings
    emb1 = torch.nn.functional.normalize(emb1, dim=-1)
    emb2 = torch.nn.functional.normalize(emb2, dim=-1)

    # Similarity matrix
    logits = torch.matmul(emb1, emb2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    batch_size = emb1.shape[0]
    labels = torch.arange(batch_size, device=emb1.device)
    
    # Symmetric loss
    loss_1 = torch.nn.functional.cross_entropy(logits, labels)
    loss_2 = torch.nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_1 + loss_2) / 2


def tsne_plot(emb1, emb2, step, output_dir, tag="default"):
    """
    Saves a t-SNE plot of the embeddings.
    Returns the path to the saved plot.
    """
    B = emb1.size(0)
    
    # Combine embeddings
    combined = torch.cat([emb1, emb2], dim=0).detach().cpu().numpy()
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, B-1))
    reduced = tsne.fit_transform(combined)
    
    # Split back
    reduced_1 = reduced[:B]
    reduced_2 = reduced[B:]
    
    fig = plt.figure(figsize=(10, 8))
    
    # Create colors for each pair
    colors = cm.rainbow(np.linspace(0, 1, B))
    
    # Plot
    for i in range(B):
        plt.scatter(reduced_1[i, 0], reduced_1[i, 1], color=colors[i], marker='o', s=50, label='Encoder' if i == 0 else "")
        plt.scatter(reduced_2[i, 0], reduced_2[i, 1], color=colors[i], marker='^', s=50, label=f'{tag.capitalize()}' if i == 0 else "")
        # Draw line between pair
        plt.plot([reduced_1[i, 0], reduced_2[i, 0]], [reduced_1[i, 1], reduced_2[i, 1]], color=colors[i], alpha=0.3)
        
    plt.title(f"t-SNE Visualization ({tag}, Step {step})")
    plt.legend()
    
    # Save to disk
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"tsne_{tag}_step_{step:06d}.png")
    plt.savefig(path)
    
    plt.close(fig)
    return path


def accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, topk=(1, 5, 10)):
    """
    Computes top-k accuracy for both directions (1->2 and 2->1).
    """
    device = embedding1.device
    B = embedding1.size(0)
    targets = torch.arange(B, device=device).view(-1, 1)

    # Cosine similarity
    emb1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
    emb2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
    sim_1to2 = emb1_norm @ emb2_norm.T

    metrics = {}
    max_k = max(topk)
    k_eff = min(max_k, B)

    _, topk_indices_1to2 = sim_1to2.topk(k_eff, dim=1, largest=True, sorted=True)
    _, topk_indices_2to1 = sim_1to2.t().topk(k_eff, dim=1, largest=True, sorted=True)

    for k in topk:
        if B < k:
            continue
        
        current_k = min(k, B)
        indices_1to2_k = topk_indices_1to2[:, :current_k]
        indices_2to1_k = topk_indices_2to1[:, :current_k]

        correct_1to2 = (indices_1to2_k == targets).any(dim=1)
        correct_2to1 = (indices_2to1_k == targets).any(dim=1)

        metrics[f"top{k}_1to2"] = correct_1to2.float().mean().item() * 100
        metrics[f"top{k}_2to1"] = correct_2to1.float().mean().item() * 100

    return metrics


def similarity_matrix(emb1, emb2):
    """
    Computes the cosine similarity matrix between two batches of embeddings.
    Returns: (Batch_Size, Batch_Size) matrix where [i, j] is sim(emb1[i], emb2[j])
    """
    emb1_norm = torch.nn.functional.normalize(emb1, dim=-1)
    emb2_norm = torch.nn.functional.normalize(emb2, dim=-1)
    return emb1_norm @ emb2_norm.T


def similarity_heatmap(sim_matrix, step, tag="default"):
    """
    Saves a heatmap of the similarity matrix.
    Returns the path to the saved plot.
    """ 
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix_np, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Similarity Matrix ({tag}, Step {step})")
    plt.xlabel(f"{tag.capitalize()} Embeddings")
    plt.ylabel("Encoder Embeddings")
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def validate(policy, val_loader, device, head_encoder, head_third=None, head_carpet=None, step=0, mode="both"):
    policy.eval()
    head_encoder.eval()
    if head_third: head_third.eval()
    if head_carpet: head_carpet.eval()
    
    total_loss = 0
    total_samples = 0
    metrics_accum = {}
    
    sim_matrix_third = None
    sim_matrix_carpet = None
    
    # Run validation on a few batches (e.g., 10 batches or full val set)
    # For speed, let's limit to 50 batches max
    max_val_batches = 50
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_val_batches:
                break
                
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            ego_embedding, third_embeddings, carpet_embeddings = extract_embeddings(policy, batch, device)
            
            # Check if we have enough data for the selected mode
            valid_step = True
            if mode in ["third", "both"] and third_embeddings is None:
                valid_step = False
            if mode in ["carpet", "both"] and carpet_embeddings is None:
                valid_step = False

            if valid_step:
                feat_encoder = ego_embedding.mean(dim=0) 
                
                feat_third = None
                if mode in ["third", "both"] and third_embeddings is not None:
                    feat_third = third_embeddings.mean(dim=0)
                    
                feat_carpet = None
                if mode in ["carpet", "both"] and carpet_embeddings is not None:
                    feat_carpet = carpet_embeddings.mean(dim=0)

                proj_encoder = head_encoder(feat_encoder)
                
                proj_third = None
                if mode in ["third", "both"] and third_embeddings is not None and head_third is not None:
                    proj_third = head_third(feat_third)
                    
                proj_carpet = None
                if mode in ["carpet", "both"] and carpet_embeddings is not None and head_carpet is not None:
                    proj_carpet = head_carpet(feat_carpet)

                loss = torch.tensor(0.0, device=device)
                
                if proj_third is not None:
                    loss_third = contrastive_loss(proj_encoder, proj_third)
                    loss += loss_third
                
                if proj_carpet is not None:
                    loss_carpet = contrastive_loss(proj_encoder, proj_carpet)
                    loss += loss_carpet
                
                batch_size = feat_encoder.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accumulate accuracy metrics
                if proj_third is not None:
                    acc_metrics = accuracy(proj_encoder, proj_third, topk=(1, 5, 10))
                    for k, v in acc_metrics.items():
                        key_name = f"third_{k}"
                        if key_name not in metrics_accum:
                            metrics_accum[key_name] = 0.0
                        metrics_accum[key_name] += v * batch_size
                        
                if proj_carpet is not None:
                    acc_metrics = accuracy(proj_encoder, proj_carpet, topk=(1, 5, 10))
                    for k, v in acc_metrics.items():
                        key_name = f"carpet_{k}"
                        if key_name not in metrics_accum:
                            metrics_accum[key_name] = 0.0
                        metrics_accum[key_name] += v * batch_size

                # Save heatmap for the first batch only
                if i == 0:
                     if proj_third is not None:
                        sim_matrix_third = similarity_matrix(proj_encoder, proj_third)
                     
                     if proj_carpet is not None:
                        sim_matrix_carpet = similarity_matrix(proj_encoder, proj_carpet)

    val_metrics = {}
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        log_msg = f"Validation Step {step}: Loss = {avg_loss:.4f}"
        
        val_metrics["val/loss"] = avg_loss
        
        for k in metrics_accum:
            avg_acc = metrics_accum[k] / total_samples
            log_msg += f", {k} = {avg_acc:.1f}%"
            val_metrics[f"val/{k}"] = avg_acc
            
        logging.info(colored(log_msg, "green"))
        
    policy.train()
    head_encoder.train()
    if head_third: head_third.train()
    if head_carpet: head_carpet.train()
    
    return sim_matrix_third, sim_matrix_carpet


def split_dataset_by_episodes(dataset, val_ratio=0.1):
    """
    Splits the dataset into train and validation sets by episodes.
    """
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


def save_checkpoint(step, output_dir, head_encoder, head_third, head_carpet, optimizer):
    """
    Saves model checkpoint.
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pth")
    
    torch.save({
        'step': step,
        'head_encoder_state_dict': head_encoder.state_dict(),
        'head_third_state_dict': head_third.state_dict(),
        'head_carpet_state_dict': head_carpet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logging.info(f"Saved checkpoint to {checkpoint_path}")


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check available device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize WandB
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg.to_dict(),
            job_type="pretrain_act",
            notes=cfg.wandb.notes,
        )

    logging.info("Creating dataset")
    # Load dataset
    dataset = make_dataset(cfg)
    
    # Split dataset by episodes
    val_ratio = 0.2
    train_dataset, val_dataset = split_dataset_by_episodes(dataset, val_ratio)
    logging.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy.to(device)
    
    if hasattr(cfg.policy, "n_obs_steps"):
        logging.info(f"Policy n_obs_steps: {cfg.policy.n_obs_steps}")
    else:
        logging.info("Policy n_obs_steps not found in config (defaulting to 1?)")

    # Initialize Projection Heads
    embed_dim = policy.config.dim_model
    proj_dim = 256 
    
    # Loss mode configuration: 'carpet', 'third_cam', 'both'
    mode = os.getenv("MODE", getattr(cfg, "mode", "both"))
    logging.info(f"Mode: {mode}")

    head_encoder = ProjectionHead(embed_dim, proj_dim).to(device)
    if mode in ["carpet", "both"]:
        head_carpet = ProjectionHead(embed_dim, proj_dim).to(device)
    if mode in ["third", "both"]:
        head_third = ProjectionHead(embed_dim, proj_dim).to(device)
    
    optimizer = torch.optim.Adam(
        list(head_encoder.parameters()) + list(head_third.parameters()) + list(head_carpet.parameters()), 
        lr=1e-4
    )

    run_name = os.path.basename(cfg.output_dir) # outputs/train/2025-04-05/06-30-11_act
    output_dir = os.path.join("/workspace/pretrain/model", run_name)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    dl_iter = cycle(train_loader)
    logging.info("Start extracting embeddings...")

    for step in tqdm(range(cfg.steps)):
        batch = next(dl_iter)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        ego_embedding, third_embeddings, carpet_embeddings = extract_embeddings(policy, batch, device)

        # Check if we have enough data for the selected mode
        valid_step = True
        if mode in ["third", "both"] and third_embeddings is None:
            valid_step = False
        if mode in ["carpet", "both"] and carpet_embeddings is None:
            valid_step = False

        if valid_step:
            feat_encoder = ego_embedding.mean(dim=0) 
            feat_third = None
            if mode in ["third", "both"] and third_embeddings is not None:
                feat_third = third_embeddings.mean(dim=0)
            feat_carpet = None
            if mode in ["carpet", "both"] and carpet_embeddings is not None:
                feat_carpet = carpet_embeddings.mean(dim=0)

            proj_encoder = head_encoder(feat_encoder)
            proj_third = None
            if mode in ["third", "both"] and third_embeddings is not None:
                proj_third = head_third(feat_third)
            proj_carpet = None
            if mode in ["carpet", "both"] and carpet_embeddings is not None:
                proj_carpet = head_carpet(feat_carpet)

            loss = torch.tensor(0.0, device=device)
            loss_third_val = 0.0
            loss_carpet_val = 0.0

            if mode in ["third", "both"] and third_embeddings is not None:
                loss_third = contrastive_loss(proj_encoder, proj_third)
                loss += loss_third
                loss_third_val = loss_third.item()
            if mode in ["carpet", "both"] and carpet_embeddings is not None:
                loss_carpet = contrastive_loss(proj_encoder, proj_carpet)
                loss += loss_carpet
                loss_carpet_val = loss_carpet.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb_metrics = {}

            if step % 10 == 0:
                # Metrics for Third Cam
                if mode in ["third", "both"] and third_embeddings is not None:
                    acc_metrics_third = accuracy(proj_encoder, proj_third, topk=(1, 5, 10))
                    sim_matrix_third = similarity_matrix(proj_encoder, proj_third)
                
                # Metrics for Carpet
                if mode in ["carpet", "both"] and carpet_embeddings is not None:
                    acc_metrics_carpet = accuracy(proj_encoder, proj_carpet, topk=(1, 5, 10))
                    sim_matrix_carpet = similarity_matrix(proj_encoder, proj_carpet)

                if cfg.wandb.enable:
                    wandb_metrics.update({
                        "train/loss": loss.item(),
                    })
                    if mode in ["third", "both"]:
                        wandb_metrics["train/loss_third"] = loss_third_val
                    if mode in ["carpet", "both"]:
                        wandb_metrics["train/loss_carpet"] = loss_carpet_val

                    if mode in ["third", "both"] and third_embeddings is not None:
                        for k, v in acc_metrics_third.items():
                            wandb_metrics[f"train/third_{k}"] = v
                        wandb_metrics["train/similarity_matrix_third"] = similarity_heatmap(sim_matrix_third, step, tag="third")

                    if mode in ["carpet", "both"] and carpet_embeddings is not None:
                        for k, v in acc_metrics_carpet.items():
                            wandb_metrics[f"train/carpet_{k}"] = v
                        wandb_metrics["train/similarity_matrix_carpet"] = similarity_heatmap(sim_matrix_carpet, step, tag="carpet")
        else:
            logging.info("Not enough data for the selected mode")

        # Validation
        if step > 0 and step % 1000 == 0:
            logging.info("Validation...")

            sim_matrix_third, sim_matrix_carpet = validate(policy, val_loader, device, head_encoder, head_third, head_carpet, step, mode)
            
            if cfg.wandb.enable:
                if mode in ["third", "both"]:
                    wandb_metrics["val/similarity_matrix_third"] = similarity_heatmap(sim_matrix_third, step, tag="third")
                if mode in ["carpet", "both"]:
                    wandb_metrics["val/similarity_matrix_carpet"] = similarity_heatmap(sim_matrix_carpet, step, tag="carpet")
            
            # Save t-SNE plot
            if 'proj_encoder' in locals():
                if mode in ["third", "both"] and 'proj_third' in locals() and proj_third is not None:
                    tsne_path = tsne_plot(proj_encoder, proj_third, step, output_dir, tag="third")
                    if cfg.wandb.enable:
                        wandb_metrics["val/tsne_plot_third"] = tsne_path # 주의: tsne_plot이 wandb.Image를 반환하는지 확인 필요
                
                if mode in ["carpet", "both"] and 'proj_carpet' in locals() and proj_carpet is not None:
                    tsne_path = tsne_plot(proj_encoder, proj_carpet, step, output_dir, tag="carpet")
                    if cfg.wandb.enable:
                        wandb_metrics["val/tsne_plot_carpet"] = tsne_path
            
            # Save Checkpoint
            save_checkpoint(step, output_dir, head_encoder, head_third, head_carpet, optimizer)

        # Log to WandB
        if cfg.wandb.enable and wandb_metrics:
            wandb.log(wandb_metrics, step=step)

    logging.info("End")
    # Save final checkpoint
    save_checkpoint(cfg.steps, output_dir, head_encoder, head_third, head_carpet, optimizer)


if __name__ == "__main__":
    init_logging()
    train()