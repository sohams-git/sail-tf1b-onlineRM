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
                 disc_train_freq: int = 200,
                 disc_gradient_steps: int = 10,
                 pref_rank_disc: bool = False,
                 pref_rank_weight: float = 0.0,
                 pref_rank_batch_size: int = 32,
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
        self.debug = debug
        self._last_disc_update_step = 0

        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.debug:
            print(f"[DEBUG] Entering train() at step {self.num_timesteps}")
        """
        Overrides TD3.train() to:
        1. Conditionally update the discriminator (controlled by disc_train_freq)
        2. Replace env rewards with discriminator surrogate rewards
        3. Run standard TD3 actor/critic updates
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        disc_losses_this_call = []
        sr_list = []

        # ------- Discriminator update (frequency-controlled) -------
        # Count how many env steps occurred since last disc update.
        # We update the disc every disc_train_freq steps so that it
        # does NOT overpower the policy before it has a chance to adapt.
        if self.num_timesteps - self._last_disc_update_step >= self.disc_train_freq:
            if self.debug:
                print(f"[DEBUG] Triggering discriminator update at step {self.num_timesteps}")
            self._last_disc_update_step = self.num_timesteps
            self.discriminator.train()

            for _ in range(self.disc_gradient_steps):
                if self.replay_buffer.size() < self.disc_batch_size:
                    break

                replay_data = self.replay_buffer.sample(
                    self.disc_batch_size, env=self._vec_normalize_env)
                expert_batch = self.teacher_buffer.sample_batch(self.disc_batch_size)

                # Move expert data to correct device
                expert_states = expert_batch['states'].to(self.device)
                expert_actions = expert_batch['actions'].to(self.device)

                total_loss, e_bce, p_bce, entropy, gp = self.discriminator.compute_loss(
                    expert_states, expert_actions,
                    replay_data.observations, replay_data.actions
                )
                
                # --- NEW: Preference Ranking Loss ---
                pref_loss_val = 0.0
                if self.pref_rank_disc and self.pref_rank_weight > 0.0:
                    try:
                        pair = self.teacher_buffer.sample_pref_pairs(self.pref_rank_batch_size)
                        p_obs, p_acs, p_mask, n_obs, n_acs, n_mask = [t.to(self.device) for t in pair]
                        
                        pref_loss = self.discriminator.compute_pref_loss(
                            p_obs, p_acs, p_mask, n_obs, n_acs, n_mask
                        )
                        total_loss = total_loss + self.pref_rank_weight * pref_loss
                        pref_loss_val = pref_loss.item()
                    except Exception as e:
                        if self.debug:
                            print(f"[SAIL] Pref RM sampling/loss error: {e}")
                # ------------------------------------
                
                self.disc_optimizer.zero_grad()
                total_loss.backward()
                self.disc_optimizer.step()
                disc_losses_this_call.append(total_loss.item())

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

        # ------- TD3 Critic + Actor updates -------
        self.discriminator.eval()
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            # Compute surrogate reward (discriminator reward)
            with torch.no_grad():
                surrogate_rewards = self.discriminator.get_reward(
                    replay_data.observations, replay_data.actions)  # shape (batch, 1)

            sr_list.append(surrogate_rewards.mean().item())

            if self.debug and gradient_step == 0:
                print(
                    f"[REWARD] env_reward mean={replay_data.rewards.mean():.3f} "
                    f"std={replay_data.rewards.std():.3f}"
                )
                print(
                    f"[REWARD] surrogate   mean={surrogate_rewards.mean():.4f} "
                    f"min={surrogate_rewards.min():.4f} max={surrogate_rewards.max():.4f} "
                    f"std={surrogate_rewards.std():.4f}"
                )
                done_frac = replay_data.dones.float().mean().item()
                print(f"[BUFFER] done_frac={done_frac:.3f}  batch_size={batch_size}")

            # --- Critic Update ---
            with torch.no_grad():
                # Sample fresh noise (NOT from stored actions)
                noise = torch.zeros_like(replay_data.actions).normal_(
                    0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                target_q1, target_q2 = self.critic_target(
                    replay_data.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2)
                # Use surrogate_rewards in place of env rewards (confirmed from original)
                target_q = surrogate_rewards + (1 - replay_data.dones) * self.gamma * target_q

                if self.debug and gradient_step == 0:
                    print(
                        f"[CRITIC] target_q  mean={target_q.mean():.4f} "
                        f"std={target_q.std():.4f} min={target_q.min():.4f}"
                    )

            current_q1, current_q2 = self.critic(
                replay_data.observations, replay_data.actions)

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

            # --- Actor Update (TD3 delayed policy) ---
            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, actor_actions).mean()
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

        self.discriminator.train()  # restore for next disc update

        # ------- Logging -------
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if disc_losses_this_call:
            self.logger.record("train/disc_loss", np.mean(disc_losses_this_call))
        self.logger.record("train/surrogate_reward_mean", np.mean(sr_list))
        self.logger.record("train/surrogate_reward_std",  np.std(sr_list))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
