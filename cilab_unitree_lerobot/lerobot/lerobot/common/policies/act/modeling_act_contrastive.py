#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import logging
import math
from collections import deque
from itertools import chain
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # Original structure: backbone group is kept as is, tactile_backbone group is added as a separate group.
        other_params = [
            p
            for n, p in self.named_parameters()
            if not (n.startswith("model.backbone") or n.startswith("model.tactile_backbone"))
            and p.requires_grad
        ]

        backbone_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("model.backbone") and p.requires_grad
        ]

        tactile_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("model.tactile_backbone") and p.requires_grad
        ]

        param_groups = [{"params": other_params}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.config.optimizer_lr_backbone})
        if tactile_params:
            param_groups.append({"params": tactile_params, "lr": self.config.optimizer_lr_backbone})
        return param_groups

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat), contrastive_loss = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        lambda_contrast = getattr(self.config, "lambda_contrast", 1.0)
        if contrastive_loss is not None:
            enc_contrastive_loss = lambda_contrast * contrastive_loss
            loss_dict["contrastive_loss"] = float(contrastive_loss.detach())
            loss_dict["lambda_contrast"] = float(lambda_contrast)

            loss_dict["enc_contrastive_loss_tensor"] = enc_contrastive_loss
            loss_dict["has_contrastive"] = True
        else:
            loss_dict["contrastive_loss"] = 0.0
            loss_dict["lambda_contrast"] = float(lambda_contrast)
            loss_dict["has_contrastive"] = False

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

            # Add lightweight CNN for tactile (preserves 2D spatial information)
            if any("tactile" in key for key in self.config.image_features):
                self.tactile_backbone = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # downsample but keep spatial
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, config.dim_model, kernel_size=1),  # project to model dim
                    # NO AdaptiveAvgPool2d((1,1)) - preserve spatial information!
                )
            else:
                self.tactile_backbone = None

        # camera view-specific backbones and projection heads for contrastive learning @eunjuyummy
        self.cam_backbones = nn.ModuleDict()
        self.cam_proj_heads = nn.ModuleDict()

        # parameters for constrative learning @eunjuyummy
        self.clip_dim = getattr(self.config, 'clip_dim', 512)

        self.cam_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tac_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj_cam = nn.Linear(self.config.dim_model, self.clip_dim)
        self.proj_tac = nn.Linear(self.config.dim_model, self.clip_dim)

        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        self.proj_third  = nn.Linear(self.config.dim_model,  self.clip_dim)
        self.proj_carpet = nn.Linear(self.config.dim_model, self.clip_dim)
        self.proj_vision = nn.Linear(self.config.dim_model, self.clip_dim) 

        self.lambda_third_carpet = 1.0
        self.lambda_carpet_vision = 1.0

        import copy

        def _make_cam_backbone():
            return copy.deepcopy(self.backbone)

        def _make_cam_proj_head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
            )
        
        self.config.cam_views = [view.split('.')[-1] for view in self.config.image_features if "tactile" not in view]

        for view in self.config.cam_views:
            self.cam_backbones[view] = _make_cam_backbone()                     
            self.cam_proj_heads[view] = _make_cam_proj_head(self.config.dim_model, self.clip_dim)

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        # Note: tactile is processed as 2D spatial tokens, not 1D
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

        # Optionally swap the encoder with weights from a separately trained encoder
        # (e.g., a head-camera encoder trained on tactile carpet data).
        if getattr(self.config, "pretrained_encoder_checkpoint", None):
            self.replace_encoder_from_checkpoint(
                self.config.pretrained_encoder_checkpoint,
                getattr(self.config, "pretrained_encoder_key", None),
            )

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def replace_encoder_from_checkpoint(self, checkpoint_path: str | Path, encoder_prefix: str | None = None):
        """Load an external encoder's weights and overwrite the current ACT encoder.

        Args:
            checkpoint_path: Path to a checkpoint that contains the pretrained encoder weights.
            encoder_prefix: Optional prefix indicating where the encoder weights live inside the
                checkpoint's state dict (e.g., "model.encoder"). If not provided, common prefixes
                are tried in order.
        """

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        prefixes = []
        if encoder_prefix:
            prefixes.append(encoder_prefix.rstrip("."))
        prefixes.extend(["encoder", "model.encoder"])

        encoder_state_dict = None
        selected_prefix = None
        for prefix in prefixes:
            full_prefix = prefix if prefix.endswith(".") else f"{prefix}."
            subset = {k[len(full_prefix) :]: v for k, v in state_dict.items() if k.startswith(full_prefix)}
            if subset:
                encoder_state_dict = subset
                selected_prefix = prefix
                break

        if encoder_state_dict is None:
            raise ValueError(
                "Could not find encoder weights in checkpoint. Tried prefixes: "
                + ", ".join(prefixes)
            )

        incompatible = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logging.warning(
                "Loaded encoder from '%s' (prefix='%s') with missing keys %s and unexpected keys %s.",
                checkpoint_file,
                selected_prefix,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        else:
            logging.info("Successfully replaced ACT encoder using checkpoint '%s'.", checkpoint_file)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None], Tensor | None]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim] # (B, L)
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :] # (B, L)

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu) # latent vector: z (B, L)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # Prepare transformer encoder inputs.
        # 1) Build 1D tokens (latent, robot_state, env_state)
        tokens_1d = [self.encoder_latent_input_proj(latent_sample)]
        if self.config.robot_state_feature:
            tokens_1d.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        if self.config.env_state_feature:
            tokens_1d.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        tokens_1d = torch.stack(tokens_1d, dim=0)  # (N1D, B, D)
        pos_embed_1d = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (N1D, 1, D)

        # 2) Build 2D spatial tokens (camera + tactile) - preserve spatial information
        tokens_2d = None
        pos_embed_2d = None
        if self.config.image_features and "observation.images" in batch:
            all_2d_features = []
            all_2d_pos_embeds = []
            
            # contrastive learning global list @eunjuyummy
            global_cam_list = []
            global_tac_list = []

            for img_key, img in zip(self.config.image_features, batch["observation.images"]):
                if "tactile" in img_key and self.tactile_backbone is not None:
                    # Process tactile as 2D spatial tokens (preserve spatial information)
                    tac_features = self.tactile_backbone(img)  # (B, D, H', W')
                    tac_pos_embed = self.encoder_cam_feat_pos_embed(tac_features).to(dtype=tac_features.dtype)
                    
                    # embedding for contrastive learning @eunjuyummy
                    tac_pooled = self.tac_pool(tac_features).flatten(1)  # (B, D)
                    tac_glb = F.normalize(self.proj_tac(tac_pooled), dim=-1)  # (B, clip_dim)
                    global_tac_list.append(tac_glb)

                    # Rearrange to sequence format
                    tac_features = einops.rearrange(tac_features, "b c h w -> (h w) b c")
                    tac_pos_embed = einops.rearrange(tac_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(tac_features)
                    all_2d_pos_embeds.append(tac_pos_embed)

                elif "tactile" not in img_key and hasattr(self, "backbone"):
                    img_key = img_key.split('.')[-1] # e.g., cam_left_high, cam_third, carpet_0
                    img_view = (img_key in self.cam_backbones and img_key in self.cam_proj_heads) # True or False

                    if img_view:
                        # Process camera as 2D spatial tokens (original path)
                        cam_features = self.cam_backbones[img_key](img)["feature_map"]
                    else:
                        cam_features = self.backbone(img)["feature_map"]   

                    cam_features_tok = self.encoder_img_feat_input_proj(cam_features)
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features_tok).to(dtype=cam_features_tok.dtype)

                    cam_features  = einops.rearrange(cam_features_tok, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    all_2d_features.append(cam_features)
                    all_2d_pos_embeds.append(cam_pos_embed)

                    cam_pooled = self.cam_pool(cam_features_tok).flatten(1)

                    if img_view:
                        cam_glb = F.normalize(self.cam_proj_heads[img_key](cam_pooled), dim=-1)
                    else:
                        cam_glb    = F.normalize(self.proj_cam(cam_pooled), dim=-1)
                    
                    global_cam_list.append(cam_glb)

            if all_2d_features:
                tokens_2d = torch.cat(all_2d_features, dim=0)  # (NPIX_total, B, D)
                pos_embed_2d = torch.cat(all_2d_pos_embeds, dim=0)  # (NPIX_total, B, D)

        # 3) Concatenate 1D and 2D tokens/positional embeddings along sequence dimension
        if tokens_2d is not None and tokens_2d.numel() > 0:
            encoder_tokens = torch.cat([tokens_1d, tokens_2d], dim=0)
            encoder_pos = torch.cat([pos_embed_1d, pos_embed_2d], dim=0)
        else:
            encoder_tokens = tokens_1d
            encoder_pos = pos_embed_1d

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_tokens, pos_embed=encoder_pos)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        contrastive_loss = None
        metrics = {}
        pair_losses, pair_weights = [], []

        has_cam = isinstance(global_cam_list, list) and len(global_cam_list) > 0
        has_tac = isinstance(global_tac_list, list) and len(global_tac_list) > 0

        if has_cam and has_tac:
            cam_views = getattr(self.config, "cam_views", None)
            tac_views = getattr(self.config, "tac_views", None)
            if not cam_views:
                cam_views = [k.split(".")[-1] for k in self.config.image_features if "tactile" not in k]
            if not tac_views:
                tac_views = [k.split(".")[-1] for k in self.config.image_features if "tactile" in k]

            cam_idx = {name: i for i, name in enumerate(cam_views)}
            tac_idx = {name: i for i, name in enumerate(tac_views)}

            need_third  = "cam_third" in cam_idx
            need_carpet = "carpet_0" in cam_idx

            third = None
            if need_third:
                third_raw = global_cam_list[cam_idx["cam_third"]]
                third = F.normalize(self.proj_third(third_raw), dim=-1)

            carpet = None
            if need_carpet:
                carpet_raw = global_cam_list[cam_idx["carpet_0"]]
                carpet = F.normalize(self.proj_carpet(carpet_raw), dim=-1)

            vision = None
            vision_views = getattr(self.config, "vision_views_for_contrast", None)
            if vision_views is None:
                vision_views = [v for v in cam_views if v != "cam_third"]

            vision_feats = []
            for v in vision_views:
                if v in cam_idx:
                    vision_feats.append(global_cam_list[cam_idx[v]]) 

            if len(vision_feats) > 0:
                vision_raw = torch.stack(vision_feats, dim=0).mean(0)       # (B, D_view)
                vision = F.normalize(self.proj_vision(vision_raw), dim=-1)  # (B, clip_dim)

            def clip_nce(a, b, logit_scale):
                # a, b: (B, clip_dim)
                logits = logit_scale.exp() * (a @ b.t())     # (B, B)
                labels = torch.arange(logits.size(0), device=logits.device)
                loss_c2t = F.cross_entropy(logits,     labels)
                loss_t2c = F.cross_entropy(logits.t(), labels)
                return logits, 0.5 * (loss_c2t + loss_t2c)

            if (third is not None) and (carpet is not None) and (self.lambda_third_carpet > 0):
                logits, loss_pair = clip_nce(third, carpet, self.logit_scale)
                pair_losses.append(loss_pair)
                pair_weights.append(self.lambda_third_carpet)

                with torch.no_grad():
                    ks = [int(k) for k in getattr(self, "topk", (1, 5)) if int(k) > 0]
                    if ks:
                        B = logits.size(0)
                        arangeB = torch.arange(B, device=logits.device).unsqueeze(1)
                        for k in ks:
                            kk = min(k, B)
                            topk_c2t = torch.topk(logits, k=kk, dim=1).indices
                            topk_t2c = torch.topk(logits.t(), k=kk, dim=1).indices
                            acc_c2t = topk_c2t.eq(arangeB).any(dim=1).float().mean()
                            acc_t2c = topk_t2c.eq(arangeB).any(dim=1).float().mean()
                            metrics[f"third<->carpet:acc@{k} (cam2tac)"] = float(acc_c2t)
                            metrics[f"third<->carpet:acc@{k} (tac2cam)"] = float(acc_t2c)

            if (carpet is not None) and (vision is not None) and (self.lambda_carpet_vision > 0):
                logits, loss_pair = clip_nce(carpet, vision, self.logit_scale)
                pair_losses.append(loss_pair)
                pair_weights.append(self.lambda_carpet_vision)

                with torch.no_grad():
                    ks = [int(k) for k in getattr(self, "topk", (1, 5)) if int(k) > 0]
                    if ks:
                        B = logits.size(0)
                        arangeB = torch.arange(B, device=logits.device).unsqueeze(1)
                        for k in ks:
                            kk = min(k, B)
                            topk_c2t = torch.topk(logits, k=kk, dim=1).indices
                            topk_t2c = torch.topk(logits.t(), k=kk, dim=1).indices
                            acc_c2t = topk_c2t.eq(arangeB).any(dim=1).float().mean()
                            acc_t2c = topk_t2c.eq(arangeB).any(dim=1).float().mean()
                            metrics[f"carpet<->vision:acc@{k} (tac2vis)"] = float(acc_c2t)
                            metrics[f"carpet<->vision:acc@{k} (vis2tac)"] = float(acc_t2c)

            if pair_losses:
                w = torch.tensor(pair_weights, device=pair_losses[0].device, dtype=pair_losses[0].dtype)
                loss_stack = torch.stack(pair_losses)  # (num_pairs,)
                contrastive_loss = (loss_stack * w).sum() / (w.sum() + 1e-8)
                metrics["logit_scale_exp"] = float(self.logit_scale.detach())
                self.last_contrastive_stats = metrics
            else:
                contrastive_loss = None
                self.last_contrastive_stats = None
        else:
            contrastive_loss = None
            self.last_contrastive_stats = None

        return actions, (mu, log_sigma_x2), contrastive_loss

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
