import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Discriminator reward helper — ablation over reward assignment functions.
#
# Let ℓ = logit (raw discriminator output before sigmoid).
# All four formulas are derived from different divergence objectives:
#   gail_js          softplus(ℓ)         Jensen-Shannon (GAIL default)
#   airl_backward_kl ℓ                   Backward KL (AIRL)
#   fairl_forward_kl -ℓ·exp(ℓ)           Forward KL (FAIRL)
#   gail_heuristic   -softplus(-ℓ)       Alternative GAIL heuristic
#
# Controlled via --disc_reward_type CLI flag.  Only get_reward() is affected;
# discriminator loss, architecture, and preference/TAC logic are unchanged.
# ---------------------------------------------------------------------------

VALID_DISC_REWARD_TYPES = (
    "gail_js",
    "airl_backward_kl",
    "fairl_forward_kl",
    "gail_heuristic",
)
_FAIRL_LOGIT_CLAMP = 10.0  # guard: exp(10) ≈ 22026, safe in float32; exp(89) overflows


def compute_disc_reward_from_logits(logits: torch.Tensor, reward_type: str) -> torch.Tensor:
    """
    Convert raw discriminator logits to a scalar surrogate reward for policy training.

    Args:
        logits: Raw discriminator output (pre-sigmoid), shape (..., 1).
        reward_type: One of VALID_DISC_REWARD_TYPES.

    Returns:
        Reward tensor, same shape as logits.
    """
    if reward_type == "gail_js":
        # softplus(ℓ) = log(1 + exp(ℓ)) — numerically stable GAIL reward.
        # Equivalent to -log(1 - sigmoid(ℓ)) without the saturation epsilon.
        return F.softplus(logits)
    elif reward_type == "airl_backward_kl":
        # Backward KL formulation: reward = ℓ (the raw logit).
        return logits
    elif reward_type == "fairl_forward_kl":
        # Forward KL formulation: reward = -ℓ · exp(ℓ).
        # NUMERICAL STABILITY: clamp logits to max=_FAIRL_LOGIT_CLAMP before exp
        # to prevent float32 overflow (exp(89) ≈ 5e38 = float32 max).
        clamped = logits.clamp(max=_FAIRL_LOGIT_CLAMP)
        return -clamped * torch.exp(clamped)
    elif reward_type == "gail_heuristic":
        # Alternative GAIL heuristic: reward = -softplus(-ℓ) = -log(1 + exp(-ℓ)).
        return -F.softplus(-logits)
    else:
        raise ValueError(
            f"Unknown disc_reward_type: {reward_type!r}. "
            f"Choose from {VALID_DISC_REWARD_TYPES}"
        )


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
                 normalize: bool = True,
                 use_expert_weights: bool = False):
        super(Adversary, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entcoeff = entcoeff
        self.gradcoeff = gradcoeff
        self.normalize = normalize
        # When True, expert BCE loss is a weighted mean: sum(w_i*L_i)/(sum(w_i)+1e-8).
        # Coupled to pref_reweight_teacher — set True only when that flag is on.
        # TF parity: DiscriminatorCalssifier.use_expert_weights (adversary.py:533).
        self.use_expert_weights = bool(use_expert_weights)

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
                     policy_state: torch.Tensor, policy_action: torch.Tensor,
                     expert_w: torch.Tensor = None
                     ) -> tuple:
        """
        Computes total discriminator loss:
          L = BCE_expert + BCE_policy - entcoeff * entropy + gradcoeff * GP

        Args:
            expert_w: Optional (N, 1) per-sample weight tensor for the expert BCE loss.
                      When self.use_expert_weights=True, computes weighted mean:
                          expert_loss = sum(w_i * L_i) / (sum(w_i) + 1e-8)
                      TF parity: DiscriminatorCalssifier loss block, adversary.py:749-756.
                      When None or use_expert_weights=False, falls back to unweighted mean.

        Returns (total_loss, expert_bce, policy_bce, entropy, grad_penalty).
        """
        expert_logits = self.forward(expert_state, expert_action)
        policy_logits = self.forward(policy_state, policy_action)

        # Expert BCE: weighted mean when use_expert_weights=True, else plain mean.
        # TF parity: sample_expert_loss reshaped to (-1,1); weighted sum / sum(w).
        sample_expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits), reduction='none')  # (N, 1)
        if self.use_expert_weights and expert_w is not None:
            expert_loss = (sample_expert_loss * expert_w).sum() / (expert_w.sum() + 1e-8)
        else:
            expert_loss = sample_expert_loss.mean()

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

    def get_reward(self, state: torch.Tensor, action: torch.Tensor,
                   reward_type: str = "gail_js") -> torch.Tensor:
        """
        Surrogate reward for TD3 critic.

        Delegates to compute_disc_reward_from_logits() for the actual formula.
        Default reward_type="gail_js" (softplus(logits)) preserves the original
        behavior: equivalent to -log(1 - sigmoid(logits) + 1e-8) to float32 precision.

        Args:
            state:       Observation tensor, shape (B, obs_dim).
            action:      Action tensor, shape (B, act_dim).
            reward_type: One of VALID_DISC_REWARD_TYPES. Controlled via --disc_reward_type.
        """
        with torch.no_grad():
            logits = self.forward(state, action)
        return compute_disc_reward_from_logits(logits, reward_type)

    def compute_soft_tac_loss(self,
                              pos_obs: torch.Tensor, pos_acs: torch.Tensor, pos_mask: torch.Tensor,
                              neg_obs: torch.Tensor, neg_acs: torch.Tensor, neg_mask: torch.Tensor,
                              y_labels: torch.Tensor,
                              temp: float = 1.0) -> tuple:
        """
        Tanh Soft-TAC discriminator alignment loss.

        Asks: does the discriminator's own J_disc ranking agree with the RM's ranking?
        Uses the same disc reward formula as get_reward() so the loss is consistent with
        what the policy optimises against.

        TF reference: adversary.py lines 685–720 (_compute_soft_tac_loss_unweighted).

        Formula:
            r_t   = -log(1 - sigmoid(disc(s_t, a_t)) + 1e-8)   [per-step disc reward]
            J_pos = sum_t  r_pos_t * pos_mask_t                  [B] disc episode return
            J_neg = sum_t  r_neg_t * neg_mask_t                  [B]
            delta  = J_pos - J_neg                                [B]
            tac_term      = y_labels * tanh(delta / T)           [B]  label-gated alignment
            tac_alignment = mean(tac_term)                        scalar ∈ [-1, 1]
            soft_tac_loss = 1.0 - tac_alignment                  scalar ∈ [0, 2]; min at perfect alignment

        Args:
            pos_obs, pos_acs, pos_mask: padded positive episode batch  [B, T_max, dim]
            neg_obs, neg_acs, neg_mask: padded negative episode batch  [B, T_max, dim]
            y_labels: discrete preference labels {-1, 0, +1} for each pair [B]
                      y=+1: RM prefers pos; y=-1: RM prefers neg; y=0: tied (zero gradient)
            temp: temperature T — controls tanh steepness. Lower T = harder alignment.

        Returns:
            (soft_tac_unweighted, tac_alignment) — both scalar Tensors.
            Caller multiplies by weight: total_disc_loss += w * soft_tac_unweighted.
        """
        def _disc_J(obs_bt: torch.Tensor, acs_bt: torch.Tensor,
                    mask_bt: torch.Tensor) -> torch.Tensor:
            """Compute per-episode discriminator returns via forward pass. [B]"""
            B, T = obs_bt.shape[0], obs_bt.shape[1]
            logits = self.forward(obs_bt.reshape(B * T, -1),
                                  acs_bt.reshape(B * T, -1))   # [B*T, 1]
            prob = torch.sigmoid(logits)
            r = -torch.log(1.0 - prob + 1e-8)                  # [B*T, 1] — same as get_reward()
            r = r.reshape(B, T)                                 # [B, T]
            return (r * mask_bt).sum(dim=1)                     # [B]

        J_pos = _disc_J(pos_obs, pos_acs, pos_mask)
        J_neg = _disc_J(neg_obs, neg_acs, neg_mask)

        delta_J   = J_pos - J_neg                                     # [B]
        alpha     = 1.0 / max(float(temp), 1e-6)
        tac_term  = y_labels * torch.tanh(alpha * delta_J)            # [B]
        tac_alignment     = tac_term.mean()                           # scalar
        soft_tac_unweighted = 1.0 - tac_alignment                     # scalar
        return soft_tac_unweighted, tac_alignment

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
