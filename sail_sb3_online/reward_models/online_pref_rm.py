"""
OnlinePrefRewardModel — trainable preference reward model for online RM training.

Architecture: MLP mapping concat(obs, act) -> scalar reward per timestep.
  Linear(obs_dim + act_dim, 256) -> Tanh
  Linear(256, 256) -> Tanh
  Linear(256, 1)

Matches discriminator architecture (2 hidden layers, 256 units, Tanh) for
consistency. No gradient penalty in v1.

Interface:
    reward(obs_np, act_np) -> np.ndarray [T]
        Drop-in replacement for PrefRewardModel.reward() used by the offline RM.
        Returns per-step reward scalars.

Training:
    update(batch) -> float
        One gradient step of Bradley-Terry BCE loss on a pair batch from SegmentStore.
        batch keys: obs1, act1, mask1, obs2, act2, mask2, label  (from SegmentStore.sample_pair_batch)
        Loss: BCE(sigmoid(R2 - R1), label)  where R = sum(r_t * mask_t)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlinePrefRewardModel(nn.Module):
    """
    Trainable MLP preference reward model.

    Owned and trained exclusively by OnlineRMManager.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_size: int = 256,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 device: torch.device = None):
        """
        Args:
            obs_dim:      observation dimension
            act_dim:      action dimension
            hidden_size:  hidden layer width (default 256, matches discriminator)
            lr:           AdamW learning rate
            weight_decay: AdamW weight decay
            device:       torch device
        """
        super().__init__()
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size
        self._device     = device if device is not None else torch.device("cpu")

        input_dim = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.to(self._device)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay)

        # Training statistics (updated by update())
        self._last_loss: float = float('nan')
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Per-step reward predictions.

        Args:
            obs: [..., obs_dim]
            act: [..., act_dim]

        Returns:
            rewards: [..., 1]  scalar reward per timestep
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

    # ------------------------------------------------------------------
    # Inference interface — matches PrefRewardModel.reward()
    # ------------------------------------------------------------------

    def reward(self, obs_np: np.ndarray, act_np: np.ndarray) -> np.ndarray:
        """
        Score a trajectory segment.  Drop-in replacement for the offline RM interface.

        Args:
            obs_np: [T, obs_dim]  float32 numpy array
            act_np: [T, act_dim]  float32 numpy array

        Returns:
            rewards: [T]  float32 numpy array of per-step reward values
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self._device)
        act_t = torch.tensor(act_np, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            r = self.forward(obs_t, act_t)  # [T, 1]
        return r.squeeze(-1).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> float:
        """
        One gradient step of Bradley-Terry preference loss.

        Loss: BCE(sigmoid(R2 - R1), label)
              where R = sum(r_t * mask_t) over segment.

        Args:
            batch: dict from SegmentStore.sample_pair_batch() with keys:
                obs1, act1, mask1  — segment 1  [B, segment_len, dim]
                obs2, act2, mask2  — segment 2  [B, segment_len, dim]
                label              — [B, 1] float32  (1 if seg2 preferred)

        Returns:
            loss value (float)
        """
        obs1  = batch['obs1'].to(self._device)   # [B, L, obs_dim]
        act1  = batch['act1'].to(self._device)
        mask1 = batch['mask1'].to(self._device)  # [B, L]
        obs2  = batch['obs2'].to(self._device)
        act2  = batch['act2'].to(self._device)
        mask2 = batch['mask2'].to(self._device)
        label = batch['label'].to(self._device)  # [B, 1]

        # Per-step rewards: [B, L, 1]
        r1 = self.forward(obs1, act1)
        r2 = self.forward(obs2, act2)

        # Segment returns: sum(r_t * mask_t) -> [B, 1]
        R1 = (r1.squeeze(-1) * mask1).sum(dim=1, keepdim=True)
        R2 = (r2.squeeze(-1) * mask2).sum(dim=1, keepdim=True)

        # Bradley-Terry BCE: P(seg2 preferred) = sigmoid(R2 - R1)
        loss = F.binary_cross_entropy(torch.sigmoid(R2 - R1), label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self._last_loss = loss_val
        self._update_count += 1
        return loss_val

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def evaluate_pairs(self, batch: dict) -> float:
        """
        Compute preference prediction accuracy on a batch (no gradient).

        Returns fraction of pairs where predicted preference matches label.
        (label=1 means seg2 preferred; correct when sigmoid(R2-R1) > 0.5)
        """
        obs1  = batch['obs1'].to(self._device)
        act1  = batch['act1'].to(self._device)
        mask1 = batch['mask1'].to(self._device)
        obs2  = batch['obs2'].to(self._device)
        act2  = batch['act2'].to(self._device)
        mask2 = batch['mask2'].to(self._device)
        label = batch['label'].to(self._device)

        with torch.no_grad():
            r1 = self.forward(obs1, act1)
            r2 = self.forward(obs2, act2)
            R1 = (r1.squeeze(-1) * mask1).sum(dim=1, keepdim=True)
            R2 = (r2.squeeze(-1) * mask2).sum(dim=1, keepdim=True)
            pred = (torch.sigmoid(R2 - R1) > 0.5).float()
            acc  = (pred == label).float().mean().item()
        return acc
