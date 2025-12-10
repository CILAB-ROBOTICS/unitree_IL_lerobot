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
            batch["observation.images"] = [batch[key].mean(dim=1) if batch[key].ndim == 5 else batch[key] for key in policy.config.image_features]

        if policy.config.env_state_feature:
            env = batch["observation.environment_state"]
            if env.ndim == 3:
                env = env.mean(dim=1)
            batch["observation.environment_state"] = env 

        if policy.config.robot_state_feature:
            rob = batch["observation.state"]
            if rob.ndim == 3:
                rob = rob.mean(dim=1)
            batch["observation.state"] = rob 

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
        third_embeddings = None

        if model.config.image_features and "observation.images" in batch:
            all_2d_features = []
            all_2d_pos_embeds = []
            carpet_features = []
            third_features = []
            
            for img_key, img in zip(model.config.image_features, batch["observation.images"]):
                if "tactile" in img_key and tactile_backbone is not None:
                    tac_features = tactile_backbone(img)
                    tac_pos_embed = model.encoder_cam_feat_pos_embed(tac_features).to(dtype=tac_features.dtype)

                    tac_features = einops.rearrange(tac_features, "b c h w -> (h w) b c")
                    tac_pos_embed = einops.rearrange(tac_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(tac_features)
                    all_2d_pos_embeds.append(tac_pos_embed)
                elif "cam_left_high" in img_key and hasattr(model, "backbone"):
                    cam_features = model.backbone(img)["feature_map"]
                    cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = model.encoder_img_feat_input_proj(cam_features)

                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(cam_features)
                    all_2d_pos_embeds.append(cam_pos_embed)
                elif "cam_third" in img_key and hasattr(model, "backbone"):
                    cam_features = model.backbone(img)["feature_map"]
                    cam_features = model.encoder_img_feat_input_proj(cam_features)

                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    third_features.append(cam_features)
                elif "carpet" in img_key and hasattr(model, "backbone"):
                    car_features = model.backbone(img)["feature_map"]
                    car_features = model.encoder_img_feat_input_proj(car_features)

                    car_features = einops.rearrange(car_features, "b c h w -> (h w) b c")
                    carpet_features.append(car_features)

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


def tsne_plot(emb1, emb2, step, output_dir, tag="default", task_indices=None, time_indices=None):
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
    
    fig = plt.figure(figsize=(6, 4))
    
    # Create colors
    if task_indices is not None:
        # Ensure task_indices is a numpy array
        if isinstance(task_indices, torch.Tensor):
            task_indices = task_indices.cpu().numpy()
        if time_indices is not None and isinstance(time_indices, torch.Tensor):
            time_indices = time_indices.cpu().numpy()
        
        unique_tasks = np.unique(task_indices)
        num_tasks = len(unique_tasks)
        
        # Define a list of sequential colormaps for different tasks
        # These are single-hue or multi-hue sequential maps
        task_cmaps = [
            plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples, 
            plt.cm.Greys, plt.cm.YlOrBr, plt.cm.PuRd, plt.cm.GnBu, plt.cm.PuBuGn
        ]
        
        # Plot with legend for tasks
        for i, task in enumerate(unique_tasks):
            mask = (task_indices == task)
            task_subset_time = time_indices[mask] if time_indices is not None else None
            
            # Select colormap (cycle if more tasks than maps)
            cmap = task_cmaps[i % len(task_cmaps)]
            
            if task_subset_time is not None:
                # Normalize time for this task to 0.2-1.0 range (avoid too light colors)
                t_min, t_max = task_subset_time.min(), task_subset_time.max()
                if t_max > t_min:
                    norm_time = 0.3 + 0.7 * (task_subset_time - t_min) / (t_max - t_min)
                else:
                    norm_time = np.ones_like(task_subset_time) * 0.7
                
                colors = cmap(norm_time)
            else:
                # Solid color if no time info
                colors = cmap(0.7)
            
            # Plot ego
            plt.scatter(reduced_1[mask, 0], reduced_1[mask, 1], c=colors, marker='o', s=8, label=f'Task {task} (ego)')
            # Plot counter
            plt.scatter(reduced_2[mask, 0], reduced_2[mask, 1], c=colors, marker="*", s=8, label=f'Task {task} ({tag})')
            
    else:
        colors = cm.rainbow(np.linspace(0, 1, B))
        # Plot
        plt.scatter(reduced_1[:, 0], reduced_1[:, 1], c=colors, marker='o', s=8, label='ego')
        plt.scatter(reduced_2[:, 0], reduced_2[:, 1], c=colors, marker="*", s=8, label=f'{tag}')
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save to disk
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"tsne_{tag}_step_{step:06d}.png")
    plt.savefig(path)
    
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, topk=(1, )):
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

        metrics[f"top{k}_1to2"] = correct_1to2.float().mean().item()
        metrics[f"top{k}_2to1"] = correct_2to1.float().mean().item()

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
    
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(sim_matrix_np, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.xlabel(f"{tag}")
    plt.ylabel("ego")
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def validate(policy, val_loader, encoder1, encoder2, step, m, cfg):
    policy.eval()
    encoder1.eval()
    encoder2.eval()
    
    total_loss = 0
    total_samples = 0
    metrics_accum = {}
    
    sim = None
    max_val_batches = 50
    
    all_ego_embs = []
    all_counter_embs = []
    all_task_indices = []
    all_time_indices = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_val_batches:
                break
                
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(cfg.policy.device, non_blocking=True)

            ego_emb, third_emb, carpet_emb = extract_embeddings(policy, batch, cfg.policy.device)
            
            emb = {"encoder": ego_emb.mean(dim=0),}

            if m == "third" and third_emb is not None:
                emb["counter"] = third_emb.mean(dim=0)
            elif m == "carpet" and carpet_emb is not None:
                emb["counter"] = carpet_emb.mean(dim=0)
            else:
                logging.info(f"Skip batch: missing modality {m}")
                continue

            proj_ego = encoder1(emb["encoder"])
            proj_counter = encoder2(emb["counter"])

            all_ego_embs.append(proj_ego.cpu())
            all_counter_embs.append(proj_counter.cpu())
            
            if "task_index" in batch:
                all_task_indices.append(batch["task_index"].cpu())
            
            if "index" in batch:
                all_time_indices.append(batch["index"].cpu())

            loss = contrastive_loss(proj_ego, proj_counter)

            batch_size = proj_ego.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            acc_metrics = accuracy(proj_ego, proj_counter, topk=(1, ))
            for k, v in acc_metrics.items():
                key_name = f"{m}_{k}"
                metrics_accum[key_name] = metrics_accum.get(key_name, 0.0) + v * batch_size

            if i == 0:
                sim_matrix = similarity_matrix(proj_ego, proj_counter)

    val_metrics = {}
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        val_metrics["val/loss"] = avg_loss
        
        log_msg = f"Validation Step {step}: Loss = {avg_loss:.4f}"
        
        for k in metrics_accum:
            avg_acc = metrics_accum[k] / total_samples
            val_metrics[f"val/accuracy (ego-{m})"] = avg_acc

    policy.train()
    encoder1.train()
    encoder2.train()
    
    if all_ego_embs:
        all_ego_embs = torch.cat(all_ego_embs, dim=0)
        all_counter_embs = torch.cat(all_counter_embs, dim=0)
    else:
        all_ego_embs = torch.tensor([])
        all_counter_embs = torch.tensor([])
        
    if all_task_indices:
        all_task_indices = torch.cat(all_task_indices, dim=0)
    else:
        all_task_indices = None
        
    if all_time_indices:
        all_time_indices = torch.cat(all_time_indices, dim=0)
    else:
        all_time_indices = None

    return sim_matrix, val_metrics, all_ego_embs, all_counter_embs, all_task_indices, all_time_indices


def split_dataset_by_episodes(dataset, val_ratio=0.2):
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

    run_name = f"{cfg.wandb.run_id}_seed{cfg.seed}"

    # Initialize WandB
    if cfg.wandb.enable:
        wandb.init(
            project='pretrain_act',
            entity='cilab-robot',
            config=cfg.to_dict(),
            job_type='pretrain_act',
            name=run_name,
            notes=cfg.wandb.notes,
        )

    logging.info("Creating dataset")
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

    # Initialize Projection Heads
    embed_dim = policy.config.dim_model
    proj_dim = 256 
    
    # calculate mode
    mode = os.getenv("MODE", getattr(cfg, "mode", "both"))

    enabled = []
    if mode in ["third", "both"]:
        enabled.append("third")
    if mode in ["carpet", "both"]:
        enabled.append("carpet")
    logging.info(f"\033[32mEnabled modalities: {enabled}\033[0m")

    head_encoder = ProjectionHead(embed_dim, proj_dim).to(device)
    encoder = {"encoder": head_encoder}

    for m in enabled:
        encoder[m] = ProjectionHead(embed_dim, proj_dim).to(device)

    params = list(encoder["encoder"].parameters())
    for m in enabled:
        params += list(encoder[m].parameters())

    optimizer = torch.optim.Adam(params, lr=1e-4)

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
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    dl_iter = cycle(train_loader)
    logging.info("Start extracting embeddings...")

    for step in tqdm(range(cfg.steps)):
        batch = next(dl_iter)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device, non_blocking=True)

        ego_emb, third_emb, carpet_emb = extract_embeddings(policy, batch, device)

        emb = {"encoder": ego_emb.mean(dim=0),}

        if "third" in enabled and third_emb is not None:
            emb["third"] = third_emb.mean(dim=0)
        if "carpet" in enabled and carpet_emb is not None:
            emb["carpet"] = carpet_emb.mean(dim=0)

        if any(m not in emb for m in enabled):
            logging.info("Not enough data for required modality")
            continue

        proj = {}
        for m in emb:
            proj[m] = encoder[m](emb[m])

        loss = 0
        loss_details = {}
        for m in enabled:
            l = contrastive_loss(proj["encoder"], proj[m])
            loss += l
            loss_details[m] = l.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb_metrics = {}
        if cfg.wandb.enable:
            wandb_metrics["train/loss"] = loss.item()

            direction_label = {
                "encoder": "ego",
                "third": "third",
                "carpet": "carpet",
            }

            for m in enabled:
                wandb_metrics[f"train/loss (ego-{m})"] = loss_details[m]

                acc = accuracy(proj["encoder"], proj[m], topk=(1, ))

                ego_name = direction_label["encoder"]
                other_name = direction_label[m]

                # forward direction: ego → m
                for k, v in acc.items():
                    topk_name = k.split("_")[0]
                    wandb_key = f"train/accuracy ({ego_name}-{other_name})"
                    wandb_metrics[wandb_key] = v

                # reverse direction: m → ego
                for k, v in acc.items():
                    topk_name = k.split("_")[0]
                    wandb_key = f"train/accuracy ({other_name}-{ego_name})"
                    wandb_metrics[wandb_key] = v

                sim = similarity_matrix(proj["encoder"], proj[m])
                wandb_metrics[f"train/similarity (ego-{m})"] = similarity_heatmap(sim, step, tag=m)

        if step > 0 and step % 10 == 0:
            logging.info("Validation...")
            for m in enabled:
                sim, val_metrics, val_ego, val_counter, val_tasks, val_times = validate(policy, val_loader, encoder["encoder"], encoder[m], step, m, cfg)
                wandb_metrics.update(val_metrics)
                wandb_metrics[f"val/similarity (ego-{m})"] = similarity_heatmap(sim, step, tag=m)

                if val_ego.numel() > 0 and val_counter.numel() > 0:
                    tsne = tsne_plot(val_ego, val_counter, step, output_dir, tag=m, task_indices=val_tasks, time_indices=val_times)
                    wandb_metrics[f"val/t-SNE (ego-{m})"] = tsne

            os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoints/checkpoint_step_{step:05d}.pth")
            torch.save({"step": step, "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),}, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")    

        if wandb_metrics:
            wandb.log(wandb_metrics, step=step)

if __name__ == "__main__":
    init_logging()
    train()