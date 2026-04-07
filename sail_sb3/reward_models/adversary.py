import torch
import torch.nn as nn
import torch.nn.functional as F


class Adversary(nn.Module):
    """
    Discriminator network for GAIL/SAIL.

    Architecture matches the original TF codebase exactly:
      - Two hidden layers with tanh activation (original used tf.nn.tanh)
      - Gradient Penalty regularization to prevent saturaton
      - Entropy loss to prevent overconfident predictions

    Reward formula confirmed from original TF code (adversary.py line 853):
        reward_op = -log(1 - sigmoid(logits) + 1e-8)
                  = softplus(logits) - log(1 + 1e-8)  [numerically close to softplus]
    The 1e-8 is critical: prevents reward from exactly hitting 0 at saturation.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_size: int = 256,
                 entcoeff: float = 0.01,
                 gradcoeff: float = 10.0,
                 dropout_prob: float = 0.0,
                 normalize: bool = True):
        super(Adversary, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entcoeff = entcoeff
        self.gradcoeff = gradcoeff
        self.normalize = normalize

        # Matches TF: Use internal RunningMeanStd for observation normalization
        self.obs_rms = None
        if self.normalize:
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            self.obs_rms = RunningMeanStd(shape=(state_dim,))

        input_dim = state_dim + action_dim

        # Matches original TF: 2x dense(256, tanh), then dense(1, identity)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply internal obs_rms normalization if enabled."""
        if self.obs_rms is None:
            return obs
        
        # Pull stats from RunningMeanStd (which stores them locally in numpy)
        # Move to correct device and cast to tensor type
        mean = torch.tensor(self.obs_rms.mean, device=obs.device, dtype=obs.dtype)
        var = torch.tensor(self.obs_rms.var, device=obs.device, dtype=obs.dtype)
        return (obs - mean) / torch.sqrt(var + 1e-8)

    def update_obs_rms(self, obs: torch.Tensor) -> None:
        """Helper to update internal RunningMeanStd stats from latest batch."""
        if self.obs_rms is not None:
            self.obs_rms.update(obs.detach().cpu().numpy())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (pre-sigmoid).
        Input shape: (batch, state_dim), (batch, action_dim) → output (batch, 1)
        """
        if self.normalize:
            state = self._normalize_obs(state)
        x = torch.cat([state.float(), action.float()], dim=-1)
        return self.net(x)

    def _logit_bernoulli_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy of a Bernoulli with the given logits. Matches original TF."""
        # ent = (1 - sigmoid(x)) * x - log(sigmoid(x))
        probs = torch.sigmoid(logits)
        return (1.0 - probs) * logits - F.logsigmoid(logits)

    def _gradient_penalty(self, state_expert, action_expert,
                          state_policy, action_policy) -> torch.Tensor:
        """
        WGAN-GP gradient penalty on interpolated inputs.
        Matches the original TF implementation (adversary.py line 789–813).
        """
        batch_size = min(state_expert.shape[0], state_policy.shape[0])
        alpha = torch.rand(batch_size, 1, device=state_expert.device)

        # TF Parity: Interpolate in the concatenated (state, action) space.
        # We MUST normalize the states before interpolation to match the TF graph structure.
        state_expert_norm = self._normalize_obs(state_expert[:batch_size])
        state_policy_norm = self._normalize_obs(state_policy[:batch_size])

        x_expert = torch.cat([state_expert_norm.float(),
                               action_expert[:batch_size].float()], dim=-1)
        x_policy = torch.cat([state_policy_norm.float(),
                               action_policy[:batch_size].float()], dim=-1)
        interpolated = (alpha * x_expert + (1 - alpha) * x_policy).requires_grad_(True)

        # Split back to use forward() which handles normalization (redundant but safe)
        # or just call self.net() directly now that inputs are normalized.
        logits = self.net(interpolated)
        grad = torch.autograd.grad(
            outputs=logits,
            inputs=interpolated,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = torch.sqrt((grad ** 2).sum(dim=1) + 1e-8)
        penalty = ((grad_norm - 1.0) ** 2).mean()
        return penalty

    def compute_loss(self,
                     expert_state: torch.Tensor, expert_action: torch.Tensor,
                     policy_state: torch.Tensor, policy_action: torch.Tensor
                     ) -> tuple:
        """
        Computes total discriminator loss:
          L = BCE_expert + BCE_policy - entcoeff * entropy + gradcoeff * GP
        Returns (total_loss, expert_bce, policy_bce, entropy, grad_penalty).
        """
        expert_logits = self.forward(expert_state, expert_action)
        policy_logits = self.forward(policy_state, policy_action)

        # BCE losses
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits))

        # Entropy regularization (prevents overconfident discriminator)
        all_logits = torch.cat([expert_logits, policy_logits], dim=0)
        entropy = self._logit_bernoulli_entropy(all_logits).mean()
        entropy_loss = -self.entcoeff * entropy

        # Gradient penalty (prevents discriminator from saturating)
        gp = self._gradient_penalty(expert_state, expert_action,
                                    policy_state, policy_action)
        gp_loss = self.gradcoeff * gp

        total_loss = expert_loss + policy_loss + entropy_loss + gp_loss
        return total_loss, expert_loss, policy_loss, entropy, gp

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Surrogate reward for TD3 critic.
        Confirmed from original TF code (adversary.py line 853):
            reward = -log(1 - sigmoid(logits) + 1e-8)

        The +1e-8 is essential: prevents reward from collapsing to exactly 0
        when the discriminator is saturated on the policy side.

        Note: -log(1 - sigmoid(x) + eps) ≈ softplus(x) for small eps, but
        the eps keeps it away from zero even at saturation.
        """
        with torch.no_grad():
            logits = self.forward(state, action)
        prob = torch.sigmoid(logits)
        return -torch.log(1.0 - prob + 1e-8)

    def compute_pref_loss(self, pos_obs: torch.Tensor, pos_acs: torch.Tensor, pos_mask: torch.Tensor,
                          neg_obs: torch.Tensor, neg_acs: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the Bradley-Terry preference ranking loss over two padded batches of trajectories.
        pos_obs, neg_obs: (batch_size, max_len, state_dim)
        pos_mask, neg_mask: (batch_size, max_len)
        """
        def compute_J(obs_bt, acs_bt, mask_bt):
            B, T = obs_bt.shape[0], obs_bt.shape[1]
            obs_flat = obs_bt.reshape(B * T, -1)
            acs_flat = acs_bt.reshape(B * T, -1)
            
            # Use same exact formulation as working vanilla SAIL reward
            logits = self.forward(obs_flat, acs_flat)
            prob = torch.sigmoid(logits)
            rewards = -torch.log(1.0 - prob + 1e-8)  # shape (B*T, 1)
            
            rewards = rewards.view(B, T)
            J = torch.sum(rewards * mask_bt, dim=1)  # shape (B,)
            return J
            
        J_pos = compute_J(pos_obs, pos_acs, pos_mask)
        J_neg = compute_J(neg_obs, neg_acs, neg_mask)
        
        # Bradley-Terry loss: mean(softplus(-(J_pos - J_neg)))
        loss = F.softplus(-(J_pos - J_neg)).mean()
        return loss


if __name__ == "__main__":
    print("--- Testing Adversary Model (matching original TF architecture) ---")
    state_dim, action_dim = 17, 6  # HalfCheetah (after trim)
    batch_size = 32

    model = Adversary(state_dim=state_dim, action_dim=action_dim)

    expert_s = torch.randn((batch_size, state_dim))
    expert_a = torch.randn((batch_size, action_dim))
    policy_s = torch.randn((batch_size, state_dim))
    policy_a = torch.randn((batch_size, action_dim))

    logits = model(expert_s, expert_a)
    print(f"Logits shape:      {logits.shape}  mean={logits.mean():.3f}  std={logits.std():.3f}")
    print(f"Expert prob mean:  {torch.sigmoid(logits).mean():.3f}")

    total, el, pl, ent, gp = model.compute_loss(expert_s, expert_a, policy_s, policy_a)
    print(f"Total loss:        {total.item():.4f}")
    print(f"  expert_bce:      {el.item():.4f}")
    print(f"  policy_bce:      {pl.item():.4f}")
    print(f"  entropy:         {ent.item():.4f}")
    print(f"  grad_penalty:    {gp.item():.4f}")

    reward = model.get_reward(policy_s, policy_a)
    print(f"Surrogate reward:  min={reward.min():.4f}  max={reward.max():.4f}  mean={reward.mean():.4f}")
