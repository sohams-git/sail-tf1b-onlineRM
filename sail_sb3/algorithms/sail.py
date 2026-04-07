import torch
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update
import numpy as np


class SAIL(TD3):
    """
    SAIL (Smooth Adversarial Imitation Learning) — SB3 TD3 subclass.

    Key design decisions matching the original TF codebase:
    - Discriminator is updated every `disc_train_freq` env steps, NOT every gradient step.
      Original TF used train_discriminator_freq=500. Default here is 200.
    - Discriminator batch size is independent from critic batch size.
    - Reward formula: -log(1 - D(s,a) + 1e-8). Confirmed from original adversary.py line 853.
    - Surrogate rewards fully replace environment rewards in the Bellman target.
    """

    def __init__(self, policy, env, discriminator, teacher_buffer,
                 disc_lr: float = 3e-4,
                 disc_batch_size: int = 128,
                 disc_train_freq: int = 500,
                 disc_gradient_steps: int = 10,
                 pref_rank_disc: bool = False,
                 pref_rank_weight: float = 0.0,
                 pref_rank_batch_size: int = 32,
                 adaptive: bool = False,
                 expert_scores: list = None,
                 lfd_mixing: bool = False,
                 debug: bool = False,
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.discriminator = discriminator
        self.teacher_buffer = teacher_buffer
        self.disc_batch_size = disc_batch_size
        self.disc_train_freq = disc_train_freq
        self.disc_gradient_steps = disc_gradient_steps
        self.pref_rank_disc = pref_rank_disc
        self.pref_rank_weight = pref_rank_weight
        self.pref_rank_batch_size = pref_rank_batch_size
        self.adaptive = adaptive
        self.lfd_mixing = lfd_mixing
        self.debug = debug
        self._last_disc_update_step = 0

        # Adaptive SAIL/PAIL: expert score threshold tracking
        # Passed to SAILAdaptiveCallback for dynamic updates
        if expert_scores is None:
            self.expert_scores = []
        else:
            self.expert_scores = list(expert_scores)  # Insertion order (TF parity: [0] = first-loaded episode)

        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr)

        # Accumulate disc losses between train() calls (disc now fires in _store_transition)
        self._pending_disc_losses: list = []
        self._pending_pref_losses: list = []

    # ------------------------------------------------------------------
    # TF parity: discriminator fires independently every disc_train_freq
    # env steps, NOT only inside train().  TF sail.py line 1672 fires
    # disc at step % 500 == 0 independently of policy training at 1000.
    # ------------------------------------------------------------------
    def _update_discriminator(self) -> None:
        """
        Run disc_gradient_steps discriminator updates.
        Called from _store_transition every disc_train_freq env steps.
        Accumulates losses into _pending_disc_losses / _pending_pref_losses
        which train() drains and logs once per policy update.
        """
        if self.replay_buffer.size() < self.disc_batch_size:
            return

        self.discriminator.train()

        for _ in range(self.disc_gradient_steps):
            replay_data = self.replay_buffer.sample(
                self.disc_batch_size, env=self._vec_normalize_env)
            expert_batch = self.teacher_buffer.sample_batch(self.disc_batch_size)

            expert_states  = expert_batch['states'].to(self.device)
            expert_actions = expert_batch['actions'].to(self.device)

            if self.discriminator.normalize:
                all_obs = torch.cat([replay_data.observations, expert_states], dim=0)
                self.discriminator.update_obs_rms(all_obs)

            total_loss, e_bce, p_bce, entropy, gp = self.discriminator.compute_loss(
                expert_states, expert_actions,
                replay_data.observations, replay_data.actions
            )

            pref_loss_val = 0.0
            if self.pref_rank_disc and self.pref_rank_weight > 0.0:
                try:
                    pair = self.teacher_buffer.sample_pref_pairs(self.pref_rank_batch_size)
                    p_obs, p_acs, p_mask, n_obs, n_acs, n_mask = [
                        t.to(self.device) for t in pair]
                    pref_loss = self.discriminator.compute_pref_loss(
                        p_obs, p_acs, p_mask, n_obs, n_acs, n_mask)
                    total_loss = total_loss + self.pref_rank_weight * pref_loss
                    pref_loss_val = pref_loss.item()
                    self._pending_pref_losses.append(pref_loss_val)
                except Exception as e:
                    if self.debug:
                        print(f"[SAIL] Pref RM sampling/loss error: {e}")

            self.disc_optimizer.zero_grad()
            total_loss.backward()
            self.disc_optimizer.step()
            self._pending_disc_losses.append(total_loss.item())

            if self.debug:
                with torch.no_grad():
                    e_logits = self.discriminator(expert_states, expert_actions)
                    p_logits = self.discriminator(
                        replay_data.observations, replay_data.actions)
                pref_str = f" pref_loss={pref_loss_val:.4f} " if self.pref_rank_disc else " "
                print(
                    f"[DISC] expert_logits  mean={e_logits.mean():.3f} std={e_logits.std():.3f} "
                    f"prob={torch.sigmoid(e_logits).mean():.3f}"
                )
                print(
                    f"[DISC] policy_logits  mean={p_logits.mean():.3f} std={p_logits.std():.3f} "
                    f"prob={torch.sigmoid(p_logits).mean():.3f}"
                )
                print(
                    f"[DISC] losses: expert_bce={e_bce.item():.4f} "
                    f"policy_bce={p_bce.item():.4f} entropy={entropy.item():.4f} "
                    f"gp={gp.item():.4f}{pref_str}total={total_loss.item():.4f}"
                )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        """
        Override SB3's _store_transition to fire the discriminator every
        disc_train_freq env steps independently — matching TF sail.py line 1672
        which fires disc at `step % train_discriminator_freq == 0` in the main
        loop, separate from the policy training loop at step % 1000 == 0.

        With train_freq=1000 and disc_train_freq=500 this gives 2 disc fires
        per policy update (at env steps N+500 and N+1000), vs the old behavior
        of 1 disc fire (only inside train() at N+1000).
        """
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        if (self.num_timesteps >= self.learning_starts and
                self.num_timesteps - self._last_disc_update_step >= self.disc_train_freq):
            if self.debug:
                print(f"[DEBUG] Disc update triggered by _store_transition at step {self.num_timesteps}")
            self._update_discriminator()
            self._last_disc_update_step = self.num_timesteps

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.debug:
            print(f"[DEBUG] Entering train() at step {self.num_timesteps}")
        """
        Overrides TD3.train() to:
        1. Replace env rewards with discriminator surrogate rewards
        2. Run standard TD3 actor/critic updates
        (Discriminator updates now happen in _store_transition, not here.)
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        sr_list = []

        # ------- TD3 Critic + Actor updates -------
        # LfD mixing (TF parity): before first promotion, mix 50% expert transitions
        # into every critic/actor batch.  Mirrors TF generate_train_data() lines 453-458:
        #   expert_batch_size = self_batch_size = batch_size // 2
        #   if lfd and mix and 'dynamic' in mode: concat(expert, policy)
        # After first promotion teacher_buffer grows → initial_size check fails →
        # mixing disabled permanently (matches TF: self.mix = False at line 1569).
        # TF parity: LfD mixing is active until the first student promotion.
        # Old check (num_transitions == initial_size) breaks with ring buffer because
        # num_transitions stays capped at max_size after the first add_episode().
        # Use _has_promotions flag instead — set True by TeacherBuffer.add_episode().
        lfd_active = (self.lfd_mixing and not self.teacher_buffer._has_promotions)

        self.discriminator.eval()
        for gradient_step in range(gradient_steps):
            self._n_updates += 1

            if lfd_active:
                # --- Mixed batch: 50% expert + 50% policy ---
                n_expert = batch_size // 2
                n_policy = batch_size - n_expert

                policy_data = self.replay_buffer.sample(
                    n_policy, env=self._vec_normalize_env)
                expert_batch = self.teacher_buffer.sample_batch(n_expert)

                exp_obs   = expert_batch['states'].to(self.device)       # (n_expert, obs_dim)
                exp_acts  = expert_batch['actions'].to(self.device)      # (n_expert, act_dim)
                exp_nobs  = expert_batch['next_states'].to(self.device)  # (n_expert, obs_dim)
                exp_dones = expert_batch['dones'].to(self.device)        # (n_expert, 1)

                mixed_obs   = torch.cat([policy_data.observations,      exp_obs],   dim=0)
                mixed_acts  = torch.cat([policy_data.actions,           exp_acts],  dim=0)
                mixed_nobs  = torch.cat([policy_data.next_observations, exp_nobs],  dim=0)
                mixed_dones = torch.cat([policy_data.dones,             exp_dones], dim=0)
            else:
                # --- Pure policy batch (after promotion or when lfd_mixing=False) ---
                policy_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env)
                mixed_obs   = policy_data.observations
                mixed_acts  = policy_data.actions
                mixed_nobs  = policy_data.next_observations
                mixed_dones = policy_data.dones

            # Compute surrogate reward for ALL transitions (expert + policy).
            # TF: rewards fully replaced by get_imitate_reward() for every batch element.
            with torch.no_grad():
                surrogate_rewards = self.discriminator.get_reward(
                    mixed_obs, mixed_acts)  # shape (batch_size, 1)

            sr_list.append(surrogate_rewards.mean().item())

            if self.debug and gradient_step == 0:
                print(
                    f"[REWARD] surrogate   mean={surrogate_rewards.mean():.4f} "
                    f"min={surrogate_rewards.min():.4f} max={surrogate_rewards.max():.4f} "
                    f"std={surrogate_rewards.std():.4f}  lfd_active={lfd_active}"
                )
                done_frac = mixed_dones.float().mean().item()
                print(f"[BUFFER] done_frac={done_frac:.3f}  batch_size={mixed_obs.shape[0]}")

            # --- Critic Update ---
            with torch.no_grad():
                noise = torch.zeros_like(mixed_acts).normal_(
                    0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(mixed_nobs) + noise
                ).clamp(-1, 1)

                target_q1, target_q2 = self.critic_target(mixed_nobs, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = surrogate_rewards + (1 - mixed_dones) * self.gamma * target_q

                if self.debug and gradient_step == 0:
                    print(
                        f"[CRITIC] target_q  mean={target_q.mean():.4f} "
                        f"std={target_q.std():.4f} min={target_q.min():.4f}"
                    )

            current_q1, current_q2 = self.critic(mixed_obs, mixed_acts)

            if self.debug and gradient_step == 0:
                print(
                    f"[CRITIC] current_q mean={current_q1.mean():.4f} "
                    f"std={current_q1.std():.4f}"
                )

            critic_loss = (F.mse_loss(current_q1, target_q) +
                           F.mse_loss(current_q2, target_q))
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor Update (TD3 delayed policy, full mixed batch) ---
            # TF: actor trained on batch_obs which is the full mixed 256-sample batch.
            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(mixed_obs)
                actor_loss = -self.critic.q1_forward(
                    mixed_obs, actor_actions).mean()
                actor_losses.append(actor_loss.item())

                if self.debug and gradient_step == 0:
                    print(
                        f"[ACTOR] action mean={actor_actions.mean():.3f} "
                        f"std={actor_actions.std():.3f}  "
                        f"actor_loss={actor_loss.item():.4f}"
                    )

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(),
                              self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(),
                              self.actor_target.parameters(), self.tau)

        # ------- Logging -------
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # Drain disc losses accumulated across the two _store_transition disc fires
        if self._pending_disc_losses:
            self.logger.record("train/disc_loss", np.mean(self._pending_disc_losses))
            self._pending_disc_losses = []
        if self._pending_pref_losses:
            self.logger.record("train/pref_loss", np.mean(self._pending_pref_losses))
            self._pending_pref_losses = []
        self.logger.record("train/surrogate_reward_mean", np.mean(sr_list))
        self.logger.record("train/surrogate_reward_std",  np.std(sr_list))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
