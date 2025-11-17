import argparse
import io
import os
import logging
import time
from typing import Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

from lerobot.common.utils.utils import init_logging
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.configs import parser

def accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, topk=(1, 5, 10)):
    device = embedding1.device
    B = embedding1.size(0)
    targets = torch.arange(B, device=device)  

    sim_1to2 = embedding1 @ embedding2.T
    sim_2to1 = embedding2 @ embedding1.T

    metrics = {}

    for k in topk:
        k_eff = min(k, B)

        if k_eff < k:
            logging.warning(f"Requested top-{k} but batch size is {B}. ")
            continue

        topk_1to2 = sim_1to2.topk(k_eff, dim=-1).indices  # (B, k_eff)
        correct_1to2 = (topk_1to2 == targets.unsqueeze(1)).any(dim=-1)  # (B,)
        metrics[f"top{k}_1to2"] = correct_1to2.float().mean().item()

        topk_2to1 = sim_2to1.topk(k_eff, dim=-1).indices
        correct_2to1 = (topk_2to1 == targets.unsqueeze(1)).any(dim=-1)
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

def clip_loss(image_embeddings:torch.Tensor, tactile_embeddings:torch.Tensor, target_matrix:torch.Tensor, logit_scale = 1.0, visualize = False):

    visualizations = []
    image_embeddings = image_embeddings.squeeze(1)      # (B, D)
    tactile_embeddings = tactile_embeddings.squeeze(1)    # (B, D)

    image_logits   = logit_scale * (image_embeddings @ tactile_embeddings.T)  # (B, B)
    tactile_logits = logit_scale * (tactile_embeddings @ image_embeddings.T)  # (B, B)

    if visualize:
        visualizations = (image_embeddings @ tactile_embeddings.T).detach().cpu().numpy()
        if isinstance(logit_scale, torch.Tensor):
            scale = float(logit_scale)
        else:
            scale = logit_scale
        visualizations = visualizations / scale  # (B, B)

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

def clip_pretraining(train_dataset, test_dataset, save_dir: str, args):
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    run_name = os.path.basename(save_dir)

    features = train_dataset.meta.features
    camera_keys = [k for k in features if k.startswith("observation.images.") and "tactile" not in k and not k.endswith("carpet_0")]
    tactile_keys = [k for k in features if k.startswith("observation.images.") and "tactile" in k or k.endswith("carpet_0")]
    
    camera_keys = [k for k in camera_keys if "cam_left_high" not in k]
    tactile_keys = [k for k in tactile_keys if "carpet_0" in k]

    n_cameras = len(camera_keys)
    

    vision_encoder = modified_resnet18(weights=None, features_per_group=args.features_per_group).to(args.device)
    vision_projection = ClipProjectionHead(out_dim=args.clip_dim).to(args.device)

    tactile_encoder = modified_resnet18(weights=None, features_per_group=args.features_per_group).to(args.device)
    tactile_projection = ClipProjectionHead(out_dim=args.clip_dim).to(args.device)

    optim_params = [{"params": tactile_encoder.parameters(), "lr": args.resnet_lr},
                    {"params": tactile_projection.parameters(), "lr": args.projection_lr},
                    {"params": vision_encoder.parameters(), "lr": args.resnet_lr},
                    {"params": vision_projection.parameters(), "lr": args.projection_lr},]
    optimizer = torch.optim.Adam(optim_params)
    
    training_losses = np.empty([args.n_epochs, n_cameras])
    testing_losses = np.empty([args.n_epochs, n_cameras])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    for epoch in tqdm(range(args.n_epochs)): # train the model
        training_loss = np.zeros(n_cameras)

        tactile_encoder.train()
        tactile_projection.train()
        vision_encoder.train()
        vision_projection.train()

        for batch_idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device, non_blocking=True)

            cam_tensors = [batch[k] for k in camera_keys] # n_cameras, C, H, W 
            images = torch.stack(cam_tensors, dim=1) # B, N, C, H, W

            tac_tensors = [batch[k] for k in tactile_keys]  
            tactiles = torch.stack(tac_tensors, dim=1) # 1, 1, 3, 32, 32

            images = images.to(args.device)
            tactiles = tactiles.to(args.device)

            B, n_cameras, C, H, W = images.shape
            images = images.view(-1, C, H, W)
            image_embeddings = vision_projection(vision_encoder(images))
            image_embeddings = image_embeddings.view(B, n_cameras, args.clip_dim) # 1, 1, 512

            B, n_tactile, C, H, W = tactiles.shape
            tactiles = tactiles.view(-1, C, H, W)
            tactile_embeddings = tactile_projection(tactile_encoder(tactiles))
            tactile_embeddings = tactile_embeddings.view(B, n_tactile, args.clip_dim) # 1, 1, 512

            # calculate target matrix
            clip_N = n_cameras + n_tactile
            target_matrix = torch.eye(clip_N).to(args.device)

            if batch_idx == 0 and epoch%args.plot_freq == 0: # visualize the first batch in each epoch
                loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)
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
                loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)

            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        training_losses[epoch] = training_loss/len(train_loader)

        # test the model
        tactile_encoder.eval()
        tactile_projection.eval()
        vision_encoder.eval()
        vision_projection.eval()

        test_loss = np.zeros(n_cameras)

        topk_sums = defaultdict(float)
        total_samples = 0
        all_img_embeds = []
        all_tac_embeds = []
        all_img_dataset_indices = []
        all_tac_dataset_indices = []

        # Get dataset names for labeling
        if isinstance(test_dataset, MultiLeRobotDataset):
            dataset_names = test_dataset.repo_ids
        else:
            dataset_names = [test_dataset.repo_id]

        with torch.no_grad():
            test_features = test_dataset.meta.features
            test_camera_keys = [k for k in test_features if k.startswith("observation.images.") and "tactile" not in k and not k.endswith("carpet_0")]
            test_tactile_keys = [k for k in test_features if k.startswith("observation.images.") and "tactile" in k or k.endswith("carpet_0")]
            test_camera_keys = [k for k in test_camera_keys if "cam_left_high" not in k]
            test_tactile_keys = [k for k in test_tactile_keys if "carpet_0" in k]
            
            for batch_idx, batch in enumerate(test_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(args.device, non_blocking=True)

                cam_tensors = [batch[k] for k in test_camera_keys] # n_cameras, C, H, W
                images = torch.stack(cam_tensors, dim=1) # B, N, C, H, W

                tac_tensors = [batch[k] for k in test_tactile_keys]  
                tactiles = torch.stack(tac_tensors, dim=1) # 1, 1, 3, 32, 32

                images = images.to(args.device)
                tactiles = tactiles.to(args.device)

                B, n_cameras, C, H, W = images.shape
                
                # Extract dataset_index if available (for MultiLeRobotDataset)
                if "dataset_index" in batch:
                    dataset_indices = batch["dataset_index"].cpu().numpy()
                else:
                    dataset_indices = np.zeros(B, dtype=np.int32)
                images = images.view(-1, C, H, W)
                image_embeddings = vision_projection(vision_encoder(images))
                image_embeddings = image_embeddings.view(B, n_cameras, args.clip_dim) # 1, 1, 512

                B, n_tactile, C, H, W = tactiles.shape
                tactiles = tactiles.view(-1, C, H, W)
                tactile_embeddings = tactile_projection(tactile_encoder(tactiles))
                tactile_embeddings = tactile_embeddings.view(B, n_tactile, args.clip_dim) # 1, 1, 512

                batch_topk = accuracy(image_embeddings.squeeze(1), tactile_embeddings.squeeze(1), topk=(1, 5, 10))
                for k, v in batch_topk.items():
                    topk_sums[k] += v * B
                total_samples += B

                all_img_embeds.append(image_embeddings.squeeze(1).detach().cpu().numpy())
                all_tac_embeds.append(tactile_embeddings.squeeze(1).detach().cpu().numpy())
                all_img_dataset_indices.append(dataset_indices)
                all_tac_dataset_indices.append(dataset_indices)

                clip_N = n_cameras + n_tactile
                target_matrix = torch.eye(clip_N).to(args.device)

                if batch_idx == 0 and epoch%args.plot_freq == 0:
                    loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)
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
                    loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)
                test_loss += loss.clone().detach().cpu().numpy()
        
        testing_losses[epoch] = test_loss/len(test_loader)
        
        epoch_topk = {k: (v / total_samples) for k, v in topk_sums.items()}

        #if (epoch + 1) >= 10 and ((epoch + 1) % 10 == 0):
        if epoch >= 0:
            try:
                if len(all_img_embeds) == 0 or len(all_tac_embeds) == 0:
                    logging.warning("No embeddings for t-SNE, skipping.")
                else:
                    img_arr = np.concatenate(all_img_embeds, axis=0)
                    tac_arr = np.concatenate(all_tac_embeds, axis=0)
                    img_dataset_indices = np.concatenate(all_img_dataset_indices, axis=0)
                    tac_dataset_indices = np.concatenate(all_tac_dataset_indices, axis=0)

                    emb = np.concatenate([img_arr, tac_arr], axis=0)
                    modality_labels = np.array([0] * len(img_arr) + [1] * len(tac_arr))  # 0=image, 1=tactile
                    dataset_labels = np.concatenate([img_dataset_indices, tac_dataset_indices], axis=0)

                    tsne = TSNE(
                        n_components=2,
                        init="random",
                        perplexity=30,
                        learning_rate="auto",
                    )
                    emb_2d = tsne.fit_transform(emb)
                    
                    # Create color map for datasets
                    num_datasets = len(dataset_names)
                    colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
                    
                    plt.figure(figsize=(8, 8))
                    
                    # Plot each combination of modality and dataset
                    for dataset_idx in range(num_datasets):
                        for modality_idx, modality_name in enumerate(["image", "tactile"]):
                            mask = (dataset_labels == dataset_idx) & (modality_labels == modality_idx)
                            if np.any(mask):
                                dataset_name = dataset_names[dataset_idx].split('/')[-1]  # Get short name
                                label = f"{dataset_name}_{modality_name}"
                                color = colors[dataset_idx]
                                # Make tactile slightly darker
                                if modality_idx == 1:
                                    color = color * 0.7
                                plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=5, label=label, alpha=0.6, color=color)
                    
                    plt.legend(loc='upper right', fontsize=8)
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
            log_dict = {
                "train/loss": training_losses[epoch].mean(),
                "test/loss": testing_losses[epoch].mean(),
                "epoch": epoch,
            }
            for k, v in epoch_topk.items():
                direction = "embedding1 to embedding2" if "1to2" in k else "embedding2 to embedding1"
                metric_name = k.split('_')[0]
                log_dict[f"test/{metric_name}: {direction}"] = v
            wandb.log(log_dict)

        # save the models
        if (epoch+1) % args.save_freq == 0:
            checkpoint_prefix = f'{run_name}_epoch_{epoch}'
            torch.save(vision_encoder.state_dict(), f'{save_dir}/{checkpoint_prefix}_vision_encoder.pth')
            torch.save(vision_projection.state_dict(), f'{save_dir}/{checkpoint_prefix}_vision_projection.pth')
            torch.save(tactile_encoder.state_dict(), f'{save_dir}/{checkpoint_prefix}_tactile_encoder.pth')
            torch.save(tactile_projection.state_dict(), f'{save_dir}/{checkpoint_prefix}_tactile_projection.pth')

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
    else:
        logging.warning(f"Combining {len(train_dataset_ids)} train datasets: {train_dataset_ids}")
        train_dataset = MultiLeRobotDataset(repo_ids=train_dataset_ids)

    if len(test_dataset_ids) == 1:
        test_dataset = LeRobotDataset(repo_id=test_dataset_ids[0])
    else:
        logging.warning(f"Combining {len(test_dataset_ids)} test datasets: {test_dataset_ids}")
        test_dataset = MultiLeRobotDataset(repo_ids=test_dataset_ids)

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

    clip_pretraining(train_dataset, test_dataset, save_dir=save_run_dir, args=args)
    
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
    parser.add_argument('--clip_dim', type=int, default=512, help='Dimension of CLIP projection head')
    parser.add_argument('--features_per_group', type=int, default=16, help='Number of features per group in projection head')
    parser.add_argument('--resnet_lr', type=float, default=1e-5, help='Learning rate for the ResNet backbone')
    parser.add_argument('--projection_lr', type=float, default=1e-4, help='Learning rate for the projection head')
    parser.add_argument('--plot_freq', type=int, default=1, help='Frequency (in epochs) of similarity plot logging')
    parser.add_argument('--save_freq', type=int, default=100, help='Frequency (in epochs) of saving checkpoints')

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