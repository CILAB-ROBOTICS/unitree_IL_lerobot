import argparse
import io
import os
import math
import logging
import time
from typing import Callable
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

from lerobot.common.utils.utils import init_logging
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.configs import parser

def accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, topk=(1, 5, 10)):
    """
    Computes top-k accuracy for both directions (1->2 and 2->1).
    Optimized to compute similarity matrix and top-k indices only once.
    Handles cases where batch size < k gracefully.
    """
    device = embedding1.device
    B = embedding1.size(0)
    targets = torch.arange(B, device=device).view(-1, 1)

    sim_1to2 = embedding1 @ embedding2.T

    metrics = {}
    max_k = max(topk)
    k_eff = min(max_k, B)

    _, topk_indices_1to2 = sim_1to2.topk(k_eff, dim=1, largest=True, sorted=True)
    _, topk_indices_2to1 = sim_1to2.t().topk(k_eff, dim=1, largest=True, sorted=True)

    for k in topk:
        if B < k:
            logging.warning(f"Requested top-{k} but batch size is {B}. Using top-{B} instead.")

        current_k = min(k, B)

        indices_1to2_k = topk_indices_1to2[:, :current_k]
        indices_2to1_k = topk_indices_2to1[:, :current_k]

        correct_1to2 = (indices_1to2_k == targets).any(dim=1)
        correct_2to1 = (indices_2to1_k == targets).any(dim=1)

        metrics[f"top{k}_1to2"] = correct_1to2.float().mean().item()
        metrics[f"top{k}_2to1"] = correct_2to1.float().mean().item()

    return metrics

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

def modified_cnn(out_dim=512) -> nn.Module:
    """
    CNN for tactile encoding.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # downsample but keep spatial
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, out_dim, kernel_size=1),  # project to model dim
    )

def clip_loss(image_embeddings:torch.Tensor, tactile_embeddings:torch.Tensor, carpet_embeddings:torch.Tensor=None, logit_scale = 1.0, 
              visualize = False, camera_names=None, tactile_names=None, carpet_names=None):

    device = image_embeddings.device
    B = image_embeddings.shape[0]
    targets = torch.arange(B, device=device)
    
    total_loss = 0.0
    num_pairs = 0
    visualizations = None
    loss_dict = {}
    
    def compute_pair_loss(emb1, emb2):
        B = emb1.shape[0]  # emb1 batch size 기준
        targets = torch.arange(B, device=emb1.device)
        logits_12 = logit_scale * (emb1 @ emb2.T)
        logits_21 = logit_scale * (emb2 @ emb1.T)
        loss = (F.cross_entropy(logits_12, targets) +
                F.cross_entropy(logits_21, targets)) / 2
        return loss, logits_12

    n_cam = image_embeddings.shape[1]
    n_tac = tactile_embeddings.shape[1]
    n_carpet = carpet_embeddings.shape[1]

    for i in range(n_cam):
        cam_name = camera_names[i] if camera_names else f"cam_{i}"
        for j in range(n_tac):
            tac_name = tactile_names[j] if tactile_names else f"tac_{j}"
            loss, logits = compute_pair_loss(image_embeddings[:, i], tactile_embeddings[:, j])
            total_loss += loss
            num_pairs += 1
            loss_dict[f"loss/{cam_name}_vs_{tac_name}"] = loss.item()
            
            # Calculate Top-K Accuracy
            acc_metrics = accuracy(image_embeddings[:, i], tactile_embeddings[:, j], topk=(1, 5, 10))
            for metric_k, metric_v in acc_metrics.items():
                loss_dict[f"acc/{cam_name}_vs_{tac_name}/{metric_k}"] = metric_v

            if visualize and i == 0 and j == 0:
                scale = float(logit_scale) if isinstance(logit_scale, torch.Tensor) else logit_scale
                visualizations = logits.detach().cpu().numpy() / scale
    '''
    if carpet_embeddings is not None:
        for i in range(n_cam):
            cam_name = camera_names[i] if camera_names else f"cam_{i}"
            for k in range(n_carpet):
                carpet_name = carpet_names[k] if carpet_names else f"carpet_{k}"
                loss, _ = compute_pair_loss(image_embeddings[:, i], carpet_embeddings[:, k])
                total_loss += loss
                num_pairs += 1
                loss_dict[f"loss/{cam_name}_vs_{carpet_name}"] = loss.item()
                
                # Calculate Top-K Accuracy
                acc_metrics = accuracy(image_embeddings[:, i], carpet_embeddings[:, k], topk=(1, 5, 10))
                for metric_k, metric_v in acc_metrics.items():
                    loss_dict[f"acc/{cam_name}_vs_{carpet_name}/{metric_k}"] = metric_v
    '''
    return total_loss / max(num_pairs, 1), visualizations, loss_dict

def clip_pretraining(train_dataset, test_dataset, train_features, save_dir: str, args):
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    run_name = os.path.basename(save_dir)

    features = train_features
    camera_keys = [k for k in features if k.startswith("observation.images.") and "tactile" not in k and not k.endswith("carpet_0")]
    tactile_keys = [k for k in features if k.startswith("observation.images.") and "tactile" in k and not k.endswith("carpet_0")]
    carpet_keys = [k for k in features if k.endswith("carpet_0")]

    camera_keys  = [k.replace(".", "_") for k in camera_keys]
    tactile_keys = [k.replace(".", "_") for k in tactile_keys]
    carpet_keys  = [k.replace(".", "_") for k in carpet_keys]

    n_cameras = len(camera_keys)
   
    # Camera Encoders
    vision_encoders = nn.ModuleDict()
    vision_projections = nn.ModuleDict()
    for key in camera_keys:
        vision_encoders[key] = modified_resnet18(weights=None, features_per_group=args.features_per_group).to(args.device)
        vision_projections[key] = ClipProjectionHead(out_dim=args.clip_dim).to(args.device)

    # Tactile Encoder
    tactile_encoders = nn.ModuleDict()
    tactile_projections = nn.ModuleDict()
    for key in tactile_keys:
        tactile_encoders[key] = modified_cnn(out_dim=512).to(args.device)
        tactile_projections[key] = ClipProjectionHead(out_dim=args.clip_dim).to(args.device)

    # Carpet Encoder
    carpet_encoder = modified_resnet18(weights=None, features_per_group=args.features_per_group).to(args.device)
    carpet_projection = ClipProjectionHead(out_dim=args.clip_dim).to(args.device)

    # State Encoder
    if "observation.state" in features:
        state_dim = features["observation.state"]["shape"][0]
        state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.clip_dim),
            nn.LayerNorm(args.clip_dim) # Optional: normalize like CLIP tokens
        ).to(args.device)
    else:
        state_encoder = None
        logging.warning("observation.state not found in features. QPos integration skipped.")

    optim_params = [{"params": carpet_encoder.parameters(), "lr": args.resnet_lr},
                    {"params": carpet_projection.parameters(), "lr": args.projection_lr},]
    if state_encoder:
        optim_params.append({"params": state_encoder.parameters(), "lr": args.projection_lr}) # Use projection LR for MLP
    for encoder in vision_encoders.values():
        optim_params.append({"params": encoder.parameters(), "lr": args.resnet_lr})
    for projection in vision_projections.values():
        optim_params.append({"params": projection.parameters(), "lr": args.projection_lr})
    for encoder in tactile_encoders.values():
        optim_params.append({"params": encoder.parameters(), "lr": args.resnet_lr})
    for projection in tactile_projections.values():
        optim_params.append({"params": projection.parameters(), "lr": args.projection_lr})
    optimizer = torch.optim.Adam(optim_params)
    
    training_losses = np.empty([args.n_epochs, n_cameras])
    testing_losses = np.empty([args.n_epochs, n_cameras])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    for epoch in tqdm(range(args.n_epochs)): # train the model
        training_loss = np.zeros(n_cameras)
        train_loss_sums = defaultdict(float) # Accumulate individual losses and accuracies

        carpet_encoder.train()
        carpet_projection.train()
        if state_encoder:
            state_encoder.train()
        for encoder in vision_encoders.values():
            encoder.train()
        for projection in vision_projections.values():
            projection.train()
        for encoder in tactile_encoders.values():
            encoder.train()
        for projection in tactile_projections.values():
            projection.train()

        for batch_idx, batch in enumerate(train_loader):
            train_topk_sums = defaultdict(float)
            total_samples = 0
            sample_counter = 0

            keys = list(batch.keys())
            for k in keys:
                v = batch[k]
                key = k.replace(".", "_")
                batch[key] = v.to(args.device) if isinstance(v, torch.Tensor) else v
                if key != k:
                    del batch[k]

            image_embedding_list = []
            for key in camera_keys:
                cam_tensor = batch[key]  # B, C, H, W
                out = vision_encoders[key](cam_tensor)
                out = vision_projections[key](out)
                image_embedding_list.append(out)
            image_embeddings = torch.stack(image_embedding_list, dim=1)  # B, N_cameras, D
        
            tactile_feat_list = []
            for key in tactile_keys:
                tac_tensor = batch[key]                     # (B, C, H, W)
                tac_features = tactile_encoders[key](tac_tensor) # (B, D, Hf, Wf)
                global_feat = tactile_projections[key](tac_features) # (B, out_dim)
                tactile_feat_list.append(global_feat)
            tactile_embeddings = torch.stack(tactile_feat_list, dim=1) # (B, N_tactile, D)

            current_tactile_keys = list(tactile_keys)
            if state_encoder is not None and "observation.state" in batch:
                qpos = batch["observation.state"] # (B, state_dim)
                qpos_emb = state_encoder(qpos)    # (B, D)
                tactile_embeddings = torch.cat([tactile_embeddings, qpos_emb.unsqueeze(1)], dim=1) # (B, N_tactile+1, D)
                current_tactile_keys.append("qpos")

            car_tensors = [batch[k] for k in carpet_keys]   # list of B, C, H, W 
            carpets = torch.stack(car_tensors, dim=1)       # B, N_carpet, C, H, W
            B, Ncarpet, C, H, W = carpets.shape
            carpets = carpets.view(-1, C, H, W)
            carpet_emb = carpet_projection(carpet_encoder(carpets))
            carpet_embeddings = carpet_emb.view(B, Ncarpet, args.clip_dim)

            # calculate loss
            if batch_idx == 0 and epoch%args.plot_freq == 0: # visualize the first batch in each epoch
                loss, plot_maps, batch_loss_dict = clip_loss(image_embeddings, tactile_embeddings, carpet_embeddings, visualize=True,
                                                             camera_names=camera_keys, tactile_names=current_tactile_keys, carpet_names=carpet_keys)
                try:
                    if args.wandb_enable:
                        plt.imshow(plot_maps)
                        plt.colorbar()
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        plt.close()
                        wandb.log({
                            "train/similarity_matrix": wandb.Image(Image.open(buf)),
                            "epoch": epoch
                        })
                except:
                    raise
            else:
                loss, _, batch_loss_dict = clip_loss(image_embeddings, tactile_embeddings, carpet_embeddings, visualize=False,
                                                     camera_names=camera_keys, tactile_names=current_tactile_keys, carpet_names=carpet_keys)
            training_loss += loss.item()
            for k, v in batch_loss_dict.items():
                train_loss_sums[k] += v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_losses[epoch] = training_loss/len(train_loader)

        # test the model
        carpet_encoder.eval()
        carpet_projection.eval()
        if state_encoder:
            state_encoder.eval()
        for encoder in vision_encoders.values():
            encoder.eval()
        for projection in vision_projections.values():
            projection.eval()
        for encoder in tactile_encoders.values():
            encoder.eval()
        for projection in tactile_projections.values():
            projection.eval()

        test_loss = np.zeros(n_cameras)
        test_loss_sums = defaultdict(float)

        all_embeddings = []
        all_time_ids = []
        all_key_labels = []
        time_key_to_id = {}
        next_time_id = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                keys = list(batch.keys())
                for k in keys:
                    v = batch[k]
                    key = k.replace(".", "_")
                    batch[key] = v.to(args.device) if isinstance(v, torch.Tensor) else v
                    if key != k:
                        del batch[k]
                
                # Extract indices for t-SNE labeling
                B = batch[camera_keys[0]].shape[0]
                if "dataset_index" in batch:
                    dataset_indices = batch["dataset_index"].detach().cpu().numpy()
                else:
                    dataset_indices = np.zeros(B, dtype=np.int32)
                if "frame_index" in batch:
                    frame_indices = batch["frame_index"].detach().view(-1).cpu().numpy().astype(np.int64)
                else:
                    frame_indices = np.arange(sample_counter, sample_counter + B, dtype=np.int64)
                if "episode_index" in batch:
                    episode_indices = batch["episode_index"].detach().view(-1).cpu().numpy().astype(np.int64)
                else:
                    episode_indices = np.zeros(B, dtype=np.int64)
                
                time_ids = np.empty(B, dtype=np.int64)
                for idx_item, (ep_idx, frame_idx) in enumerate(zip(episode_indices, frame_indices)):
                    key = (int(ep_idx), int(frame_idx))
                    if key not in time_key_to_id:
                        time_key_to_id[key] = next_time_id
                        time_id_to_pair[next_time_id] = key
                        next_time_id += 1
                    time_ids[idx_item] = time_key_to_id[key]
                sample_counter += B

                image_embedding_list = []
                for key in camera_keys:
                    cam_tensor = batch[key]  # B, C, H, W
                    out = vision_encoders[key](cam_tensor)
                    out = vision_projections[key](out)
                    image_embedding_list.append(out)
                image_embeddings = torch.stack(image_embedding_list, dim=1)  # B, N_cameras, D
            
                tactile_feat_list = []
                for key in tactile_keys:
                    tac_tensor = batch[key]                     # (B, C, H, W)
                    tac_features = tactile_encoders[key](tac_tensor) # (B, D, Hf, Wf)
                    global_feat = tactile_projections[key](tac_features) # (B, out_dim)
                    tactile_feat_list.append(global_feat)
                tactile_embeddings = torch.stack(tactile_feat_list, dim=1) # (B, N_tactile, D)

                current_tactile_keys = list(tactile_keys)
                if state_encoder is not None and "observation.state" in batch:
                    qpos = batch["observation.state"] # (B, state_dim)
                    qpos_emb = state_encoder(qpos)    # (B, D)
                    tactile_embeddings = torch.cat([tactile_embeddings, qpos_emb.unsqueeze(1)], dim=1) # (B, N_tactile+1, D)
                    current_tactile_keys.append("qpos")

                car_tensors = [batch[k] for k in carpet_keys]   # list of B, C, H, W 
                carpets = torch.stack(car_tensors, dim=1)       # B, N_carpet, C, H, W
                B, Ncarpet, C, H, W = carpets.shape
                carpets = carpets.view(-1, C, H, W)
                carpet_emb = carpet_projection(carpet_encoder(carpets))
                carpet_embeddings = carpet_emb.view(B, Ncarpet, args.clip_dim)

                if batch_idx == 0 and epoch%args.plot_freq == 0:
                    loss, plot_maps, batch_loss_dict = clip_loss(image_embeddings, tactile_embeddings, carpet_embeddings, visualize=True,
                                                             camera_names=camera_keys, tactile_names=current_tactile_keys, carpet_names=carpet_keys)
                    try:
                        if args.wandb_enable:
                            plt.imshow(plot_maps)
                            plt.colorbar()
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            plt.close()

                            wandb.log({
                                "test/similarity_matrix": wandb.Image(Image.open(buf)),
                                "epoch": epoch
                            })
                    except:
                        raise
                else:
                    loss, _, batch_loss_dict = clip_loss(image_embeddings, tactile_embeddings, carpet_embeddings, visualize=False,
                                                             camera_names=camera_keys, tactile_names=current_tactile_keys, carpet_names=carpet_keys)
                test_loss += loss.item()
                for k, v in batch_loss_dict.items():
                    test_loss_sums[k] += v
                
                # Store embeddings for t-SNE visualization (grouped by time step)
                n_cameras = image_embeddings.shape[1]
                n_tactile = tactile_embeddings.shape[1]
                n_carpet = carpet_embeddings.shape[1]

                image_embeddings_np = image_embeddings.detach().cpu().numpy().reshape(B * n_cameras, args.clip_dim)
                tactile_embeddings_np = tactile_embeddings.detach().cpu().numpy().reshape(B * n_tactile, args.clip_dim)
                carpet_embeddings_np = carpet_embeddings.detach().cpu().numpy().reshape(B * n_carpet, args.clip_dim)
                
                img_time_ids = np.repeat(time_ids, n_cameras)
                tac_time_ids = np.repeat(time_ids, n_tactile)
                car_time_ids = np.repeat(time_ids, n_carpet)

                all_embeddings.append(image_embeddings_np)
                all_time_ids.append(img_time_ids)
                all_key_labels.append(np.tile(camera_keys, B))

                all_embeddings.append(tactile_embeddings_np)
                all_time_ids.append(tac_time_ids)
                all_key_labels.append(np.tile(current_tactile_keys, B))

                all_embeddings.append(carpet_embeddings_np)
                all_time_ids.append(car_time_ids)
                all_key_labels.append(np.tile(carpet_keys, B))
        testing_losses[epoch] = test_loss/len(test_loader)

        if epoch >= 0:
            try:
                if len(all_embeddings) == 0:
                    logging.warning("No embeddings for t-SNE, skipping.")
                else:
                    emb = np.concatenate(all_embeddings, axis=0)
                    time_labels = np.concatenate(all_time_ids, axis=0)
                    key_labels = np.concatenate(all_key_labels, axis=0)

                    unique_time_ids = np.unique(time_labels)
                    if unique_time_ids.size == 0:
                        logging.warning("No time labels for t-SNE, skipping.")
                    else:
                        rng = np.random.default_rng(args.tsne_seed)
                        num_time_samples = min(args.tsne_time_samples, unique_time_ids.size)
                        sampled_time_ids = rng.choice(unique_time_ids, size=num_time_samples, replace=False)

                        mask = np.isin(time_labels, sampled_time_ids)
                        emb = emb[mask]
                        time_labels = time_labels[mask]
                        key_labels = key_labels[mask]

                        if emb.shape[0] < 2:
                            logging.warning("Not enough sampled embeddings for t-SNE, skipping.")
                        else:
                            tsne = TSNE(
                                n_components=2,
                                init="random",
                                perplexity=30,
                                learning_rate="auto",
                            )
                            emb_2d = tsne.fit_transform(emb)

                            plt.figure(figsize=(12, 8))
                            color_palette = plt.cm.tab20(np.linspace(0, 1, len(sampled_time_ids)))
                            time_handles = []

                            def get_marker(key):
                                if "carpet" in key: return "s" # square
                                if "tactile" in key: return "x" # x
                                if "qpos" in key: return "*" # star
                                return "o" # circle (camera)

                            for idx_time, time_id in enumerate(sampled_time_ids):
                                color = color_palette[idx_time % len(color_palette)]
                                time_mask = time_labels == time_id
                                if not np.any(time_mask):
                                    continue
                                ep_idx, frame_idx = time_id_to_pair.get(int(time_id), (None, None))
                                time_label = (
                                    f"ep{ep_idx}_frame{frame_idx}" if ep_idx is not None else f"time_{time_id}"
                                )

                                # Plot each key type with specific marker
                                current_keys = key_labels[time_mask]
                                current_emb = emb_2d[time_mask]
                                
                                unique_keys_in_batch = np.unique(current_keys)
                                for key in unique_keys_in_batch:
                                    key_mask = current_keys == key
                                    marker = get_marker(key)
                                    plt.scatter(
                                        current_emb[key_mask, 0],
                                        current_emb[key_mask, 1],
                                        s=30 if marker == "*" else 15,
                                        label=None,
                                        alpha=0.8,
                                        color=color,
                                        marker=marker,
                                    )

                                time_handles.append(Line2D([], [], marker="o", linestyle="", color=color, label=time_label))

                            modality_handles = [
                                Line2D([], [], marker="o", linestyle="", color="black", label="Camera"),
                                Line2D([], [], marker="x", linestyle="", color="black", label="Tactile"),
                                Line2D([], [], marker="s", linestyle="", color="black", label="Carpet"),
                                Line2D([], [], marker="*", linestyle="", color="black", label="QPos"),
                            ]
                            
                            # Combine handles: Modality types first, then Time colors
                            plt.legend(handles=modality_handles + time_handles, loc='upper right', fontsize=8, ncol=2)
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", bbox_inches="tight")
                            buf.seek(0)
                            plt.close()
                            if args.wandb_enable:
                                tsne_img = Image.open(buf)
                                wandb.log({
                                    "tsne/test_embeddings": wandb.Image(tsne_img),
                                    "epoch": epoch,
                                })
            except Exception as e:
                logging.warning(f"t-SNE plotting failed at epoch {epoch+1}: {e}")

        # Log metrics to W&B
        if args.wandb_enable:
            log_dict = {}
            log_dict["train/loss"] = training_losses[epoch].mean()
            log_dict["test/loss"] = testing_losses[epoch].mean()
            log_dict["epoch"] = epoch
            for k, v in train_loss_sums.items():
                log_dict[f"train/{k}"] = v / len(train_loader)
            for k, v in test_loss_sums.items():
                log_dict[f"test/{k}"] = v / len(test_loader)
            wandb.log(log_dict)

        # save the models
        if (epoch+1) % args.save_freq == 0:
            checkpoint_prefix = f'{run_name}_epoch_{epoch}'
            for key in vision_encoders.keys():
                torch.save(vision_encoders[key].state_dict(), f'{save_dir}/{checkpoint_prefix}_vision_encoder_{key}.pth')
            for key in vision_projections.keys():
                torch.save(vision_projections[key].state_dict(), f'{save_dir}/{checkpoint_prefix}_vision_projection_{key}.pth')
            for key in tactile_encoders.keys():
                torch.save(tactile_encoders[key].state_dict(), f'{save_dir}/{checkpoint_prefix}_tactile_encoder_{key}.pth')
            for key in tactile_projections.keys():
                torch.save(tactile_projections[key].state_dict(), f'{save_dir}/{checkpoint_prefix}_tactile_projection_{key}.pth')
            torch.save(carpet_encoder.state_dict(), f'{save_dir}/{checkpoint_prefix}_carpet_encoder.pth')
            torch.save(carpet_projection.state_dict(), f'{save_dir}/{checkpoint_prefix}_carpet_projection.pth')
            if state_encoder:
                torch.save(state_encoder.state_dict(), f'{save_dir}/{checkpoint_prefix}_state_encoder.pth')
            
            # save the optimizer
            torch.save(optimizer.state_dict(), f'{save_dir}/{checkpoint_prefix}_optimizer.pth')
    
def run_clip_pretraining(args):
    # argparse with nargs='+' already returns a list
    train_dataset_ids = args.train_dataset_id
    test_dataset_ids = args.test_dataset_id
    
    if args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "train_datasets": train_dataset_ids,
                "test_datasets": test_dataset_ids,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "clip_dim": args.clip_dim,
                "features_per_group": args.features_per_group,
                "resnet_lr": args.resnet_lr,
                "projection_lr": args.projection_lr,
                "plot_freq": args.plot_freq,
                "save_freq": args.save_freq,
                "device": args.device,
            }
        )
        run_name = wandb_run.name
    else:
        wandb_run = None
        run_name = args.wandb_name or time.strftime("%Y%m%d_%H%M%S")

    start_time = time.time()
    os.makedirs(args.save_dir, exist_ok=True)


    if len(train_dataset_ids) == 1:
        train_dataset = LeRobotDataset(repo_id=train_dataset_ids[0])
        train_features = train_dataset.features
    else:
        dataset_list = [LeRobotDataset(repo_id=d) for d in train_dataset_ids]
        train_dataset = ConcatDataset(dataset_list)
        train_features = dataset_list[0].features

    if len(test_dataset_ids) == 1:
        test_dataset = LeRobotDataset(repo_id=test_dataset_ids[0])
        test_features = test_dataset.features
    else:
        dataset_list = [LeRobotDataset(repo_id=d) for d in test_dataset_ids]
        test_dataset = ConcatDataset(dataset_list)

    desired_dir = os.path.join(args.save_dir, run_name)
    save_run_dir = desired_dir
    suffix = 1
    while os.path.exists(save_run_dir):
        save_run_dir = f"{desired_dir}_{suffix}"
        suffix += 1
    os.makedirs(save_run_dir, exist_ok=True)

    with open(f'{save_run_dir}/run_stats.txt', 'w') as f:
        f.write(f'train_datasets: {train_dataset_ids}\n')
        f.write(f'train_dataset: {train_dataset}\n')
        f.write(f'test_datasets: {test_dataset_ids}\n')
        f.write(f'test_dataset: {test_dataset}\n')
        f.write(f'trian_features: {train_features}\n')
        f.write(f'test_features: {test_features}\n')

    clip_pretraining(train_dataset, test_dataset, train_features, save_dir=save_run_dir, args=args)
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    logging.warning(f"Total training time: {minutes}m {seconds}s")

    if args.wandb_enable:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_id', type=str, nargs='+', default=['eunjuri/towel_strong_train_full', 'eunjuri/towel_weak_train_full'], help='HuggingFace train dataset repository IDs')
    parser.add_argument('--test_dataset_id', type=str, nargs='+', default=['eunjuri/towel_strong_test_full', 'eunjuri/towel_weak_test_full'], help='HuggingFace test dataset repository IDs')

    # model parameters
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--clip_dim', type=int, default=256, help='Dimension of CLIP projection head')
    parser.add_argument('--features_per_group', type=int, default=16, help='Number of features per group in projection head')
    parser.add_argument('--resnet_lr', type=float, default=1e-5, help='Learning rate for the ResNet backbone')
    parser.add_argument('--projection_lr', type=float, default=1e-4, help='Learning rate for the projection head')
    parser.add_argument('--plot_freq', type=int, default=1, help='Frequency (in epochs) of similarity plot logging')
    parser.add_argument('--save_freq', type=int, default=100, help='Frequency (in epochs) of saving checkpoints')
    parser.add_argument('--tsne_time_samples', type=int, default=200, help='Number of unique time steps to visualize in t-SNE')
    parser.add_argument('--tsne_seed', type=int, default=42, help='Random seed used for t-SNE time sampling')

    parser.add_argument('--save_dir', type=str, default='/workspace/clip_pretrain/clip_models/', help='Directory to save trained CLIP models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run training on')

    parser.add_argument('--wandb_enable', type=bool, default=False, help='Enable or disable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='clip-pretraining', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='cilab-robot', help='WandB entity name')
    parser.add_argument('--wandb_name', type=str, default='test', help='WandB run name')
    args = parser.parse_args()

    init_logging()
    run_clip_pretraining(args)

'''
python pretrain.py \
  --train_dataset_id dataset1 dataset2 dataset3 \
  --test_dataset_id test1 test2 test3 \
  --n_epochs 10 \
  --wandb_enable=true \
  --wandb_name=clip-pretraining \
'''