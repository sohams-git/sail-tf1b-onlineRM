import torch
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update, safe_mean
import numpy as np
from typing import Optional
from collections import deque


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
                 pref_reweight_teacher: bool = False,
                 adaptive: bool = False,
                 expert_scores: list = None,
                 lfd_mixing: bool = False,
                 debug: bool = False,
                 # Online RM manager (None = disabled)
                 online_rm_manager=None,
                 # QPREF: Q-preference ranking loss on the TD3 critic
                 qpref: bool = False,
                 qpref_weight: float = 0.0,
                 qpref_temp: float = 1.0,
                 qpref_batch_size: int = 4,
                 qpref_start_step: int = 0,
                 qpref_source: str = 'teacher',
                 qpref_mean_trajectory_q: bool = True,
                 qpref_grad_interval: int = 10,
                 qpref_guard_threshold: float = -0.3,
                 qpref_guard_confirm: int = 3,
                 qpref_guard_window: int = 10,
                 qpref_guard_positive_threshold: float = 1.0,
                 # Soft-TAC: tanh discriminator alignment with RM-derived preference labels
                 soft_tac: bool = False,
                 soft_tac_weight: float = 0.0,
                 soft_tac_temp: float = 1.0,
                 tac_tie_eps: float = 0.0,
                 # Normalized score logging
                 expert_return: Optional[float] = None,
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
        self.pref_reweight_teacher = pref_reweight_teacher
        self.adaptive = adaptive
        self.lfd_mixing = lfd_mixing
        self.debug = debug
        self._last_disc_update_step = 0
        # QPREF
        self.qpref = qpref
        self.qpref_weight = qpref_weight
        self.qpref_temp = max(float(qpref_temp), 1e-6)
        self.qpref_batch_size = qpref_batch_size
        self.qpref_start_step = qpref_start_step
        self.qpref_source = qpref_source
        self.qpref_mean_trajectory_q = qpref_mean_trajectory_q
        self.qpref_grad_interval = max(1, int(qpref_grad_interval))
        self.qpref_guard_threshold          = float(qpref_guard_threshold)
        self.qpref_guard_confirm            = max(1, int(qpref_guard_confirm))
        self.qpref_guard_window             = max(1, int(qpref_guard_window))
        self.qpref_guard_positive_threshold = float(qpref_guard_positive_threshold)
        self._qpref_active          = True
        self._qpref_guard_triggered = False
        self._qpref_guard_count     = 0
        self._qpref_skipped_updates = 0
        self._qpref_delta_deque     = deque(maxlen=50)
        self._qpref_peak_guard_mean = -float('inf')
        # Soft-TAC
        self.soft_tac = soft_tac
        self.soft_tac_weight = soft_tac_weight
        self.soft_tac_temp = max(float(soft_tac_temp), 1e-6)
        self.tac_tie_eps = float(tac_tie_eps)
        # Normalized score: policy_return / expert_return (None = disabled)
        self.expert_return = float(expert_return) if expert_return is not None else None

        # Adaptive SAIL/PAIL: expert score threshold tracking
        # Passed to SAILAdaptiveCallback for dynamic updates
        if expert_scores is None:
            self.expert_scores = []
        else:
            self.expert_scores = list(expert_scores)  # Insertion order (TF parity: [0] = first-loaded episode)

        # Online RM manager (Phase 3: passive wiring; Phase 4: active reweighting)
        self.online_rm_manager = online_rm_manager

        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr)

        # Accumulate disc losses between train() calls (disc now fires in _store_transition)
        self._pending_disc_losses: list = []
        self._pending_pref_losses: list = []
        # pref_reweight_teacher: track last-seen pool size and weight spread for logging
        self._pending_reweight_pool_size: list = []
        self._pending_reweight_w_max: list = []
        # QPREF logging accumulators (drained once per train() call)
        self._pending_qpref_losses:   list = []
        self._pending_qpref_q_pos:    list = []
        self._pending_qpref_q_neg:    list = []
        self._pending_qpref_delta:    list = []
        self._pending_qpref_pairs_ok: list = []   # bool: whether pairs were available
        # Soft-TAC logging accumulators
        self._pending_soft_tac_losses:  list = []
        self._pending_tac_alignments:   list = []
        self._pending_tac_label_pos:    list = []
        self._pending_tac_label_neg:    list = []
        self._pending_tac_label_zero:   list = []
        # pref_rank_disc J-diff stats (logged per train() call)
        self._pending_pref_j_diffs:     list = []
        # Discriminator internals logging accumulators
        self._pending_disc_e_bce:        list = []
        self._pending_disc_p_bce:        list = []
        self._pending_disc_entropy:      list = []
        self._pending_disc_gp:           list = []
        self._pending_disc_logits_exp:   list = []
        self._pending_disc_logits_pol:   list = []
        self._pending_disc_prob_exp:     list = []
        self._pending_disc_prob_pol:     list = []
        self._pending_disc_acc_exp:      list = []
        self._pending_disc_acc_pol:      list = []
        self._pending_disc_reward_exp:   list = []
        self._pending_disc_reward_pol:   list = []
        # TD3 critic internals logging accumulators
        self._pending_qf1_losses:        list = []
        self._pending_qf2_losses:        list = []
        self._pending_q1_mean:           list = []
        self._pending_q2_mean:           list = []
        self._pending_q_target_mean:     list = []
        self._pending_q_target_std:      list = []
        # Pref reweight extended stats
        self._pending_reweight_w_min:      list = []
        self._pending_reweight_w_mean:     list = []
        self._pending_reweight_w_std:      list = []
        self._pending_reweight_w_entropy:  list = []
        self._pending_reweight_j_mean:     list = []
        self._pending_reweight_j_std:      list = []
        self._pending_reweight_j_min:      list = []
        self._pending_reweight_j_max:      list = []

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

            # Expert batch: weighted if pref_reweight_teacher is on and pool is ready.
            # TF parity: _sample_pref_weighted_expert() returns (ob_expert, ac_expert, expert_w).
            # When flag is off, expert_w = ones → compute_loss ignores them (use_expert_weights=False).
            # Online RM path: additionally requires online_rm_manager.is_active so that
            # uniform weights are used during warmup (no offline RM fallback).
            _pref_pool_ready = len(self.teacher_buffer.pref_episodes) >= 2
            _online_rm_active = (self.online_rm_manager is not None
                                  and self.online_rm_manager.is_active)
            use_weighted = (self.pref_reweight_teacher
                            and _pref_pool_ready
                            and (self.teacher_buffer.pref_rm is not None
                                 or _online_rm_active))
            if use_weighted:
                expert_states, expert_actions, expert_w = \
                    self.teacher_buffer.sample_batch_weighted(self.disc_batch_size)
                expert_states  = expert_states.to(self.device)
                expert_actions = expert_actions.to(self.device)
                expert_w       = expert_w.to(self.device)
            else:
                expert_batch   = self.teacher_buffer.sample_batch(self.disc_batch_size)
                expert_states  = expert_batch['states'].to(self.device)
                expert_actions = expert_batch['actions'].to(self.device)
                expert_w       = torch.ones(self.disc_batch_size, 1,
                                            dtype=torch.float32, device=self.device)

            if self.discriminator.normalize:
                all_obs = torch.cat([replay_data.observations, expert_states], dim=0)
                self.discriminator.update_obs_rms(all_obs)

            total_loss, e_bce, p_bce, entropy, gp = self.discriminator.compute_loss(
                expert_states, expert_actions,
                replay_data.observations, replay_data.actions,
                expert_w=expert_w
            )

            # Accumulate discriminator component losses for logging
            self._pending_disc_e_bce.append(e_bce.item())
            self._pending_disc_p_bce.append(p_bce.item())
            self._pending_disc_entropy.append(entropy.item())
            self._pending_disc_gp.append(gp.item())

            pref_loss_val = 0.0
            # PrefRank (hard BT): uses pref_episodes (expert + promoted students only)
            if self.pref_rank_disc and self.pref_rank_weight > 0.0 and len(self.teacher_buffer.pref_episodes) >= 2:
                try:
                    pair_result = self.teacher_buffer.sample_pref_pairs(
                        self.pref_rank_batch_size, return_J=True)
                    p_obs, p_acs, p_mask, n_obs, n_acs, n_mask, pos_J, neg_J = pair_result
                    p_obs, p_acs, p_mask = [t.to(self.device) for t in (p_obs, p_acs, p_mask)]
                    n_obs, n_acs, n_mask = [t.to(self.device) for t in (n_obs, n_acs, n_mask)]
                    pos_J = pos_J.to(self.device)
                    neg_J = neg_J.to(self.device)
                    pref_loss = self.discriminator.compute_pref_loss(
                        p_obs, p_acs, p_mask, n_obs, n_acs, n_mask)
                    total_loss = total_loss + self.pref_rank_weight * pref_loss
                    pref_loss_val = pref_loss.item()
                    self._pending_pref_losses.append(pref_loss_val)
                    j_diffs = (pos_J - neg_J).detach().cpu().numpy()
                    self._pending_pref_j_diffs.extend(j_diffs.tolist())
                except Exception as e:
                    if self.debug:
                        print(f"[SAIL] PrefRank pair sampling/loss error: {e}")

            # Soft-TAC: uses soft_tac_pool (expert + ALL students, unfiltered by promotion).
            # Gated behind online RM activation and at least one full pool rescore so
            # that all J labels are in RM space — GT placeholders are never used.
            if (self.soft_tac
                    and self.soft_tac_weight > 0.0
                    and len(self.teacher_buffer.soft_tac_pool) >= 2
                    and self.online_rm_manager is not None
                    and self.online_rm_manager.is_active
                    and self.teacher_buffer._soft_tac_pool_rescored):
                try:
                    tac_result = self.teacher_buffer.sample_soft_tac_pairs(self.pref_rank_batch_size)
                    tp_obs, tp_acs, tp_mask, tn_obs, tn_acs, tn_mask, tpos_J, tneg_J = tac_result
                    tp_obs, tp_acs, tp_mask = [t.to(self.device) for t in (tp_obs, tp_acs, tp_mask)]
                    tn_obs, tn_acs, tn_mask = [t.to(self.device) for t in (tn_obs, tn_acs, tn_mask)]
                    tpos_J = tpos_J.to(self.device)
                    tneg_J = tneg_J.to(self.device)
                    diff = (tpos_J - tneg_J).detach().cpu().numpy()
                    y = np.zeros_like(diff, dtype=np.float32)
                    y[diff >  self.tac_tie_eps] =  1.0
                    y[diff < -self.tac_tie_eps] = -1.0
                    y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
                    tac_unweighted, tac_alignment = self.discriminator.compute_soft_tac_loss(
                        tp_obs, tp_acs, tp_mask, tn_obs, tn_acs, tn_mask,
                        y_labels=y_tensor, temp=self.soft_tac_temp)
                    total_loss = total_loss + self.soft_tac_weight * tac_unweighted
                    self._pending_soft_tac_losses.append(
                        (self.soft_tac_weight * tac_unweighted).item())
                    self._pending_tac_alignments.append(tac_alignment.item())
                    self._pending_tac_label_pos.append(int(np.sum(y > 0)))
                    self._pending_tac_label_neg.append(int(np.sum(y < 0)))
                    self._pending_tac_label_zero.append(int(np.sum(y == 0)))
                except Exception as e:
                    if self.debug:
                        print(f"[SAIL] Soft-TAC pair sampling/loss error: {e}")

            self.disc_optimizer.zero_grad()
            total_loss.backward()
            self.disc_optimizer.step()
            self._pending_disc_losses.append(total_loss.item())

            # Compute logits/probs/rewards for logging (single forward pass, always on)
            with torch.no_grad():
                _e_logits = self.discriminator(expert_states, expert_actions)
                _p_logits = self.discriminator(replay_data.observations, replay_data.actions)
                _e_prob = torch.sigmoid(_e_logits)
                _p_prob = torch.sigmoid(_p_logits)
                _e_rew = -torch.log(1.0 - _e_prob + 1e-8)
                _p_rew = -torch.log(1.0 - _p_prob + 1e-8)
            self._pending_disc_logits_exp.append(_e_logits.mean().item())
            self._pending_disc_logits_pol.append(_p_logits.mean().item())
            self._pending_disc_prob_exp.append(_e_prob.mean().item())
            self._pending_disc_prob_pol.append(_p_prob.mean().item())
            self._pending_disc_acc_exp.append((_e_prob > 0.5).float().mean().item())
            self._pending_disc_acc_pol.append((_p_prob < 0.5).float().mean().item())
            self._pending_disc_reward_exp.append(_e_rew.mean().item())
            self._pending_disc_reward_pol.append(_p_rew.mean().item())

            # Track reweighting stats once per gradient step (first iteration)
            if use_weighted and self.teacher_buffer.pref_teacher_weights is not None:
                self._pending_reweight_pool_size.append(
                    len(self.teacher_buffer.pref_episodes))
                self._pending_reweight_w_max.append(
                    float(self.teacher_buffer.pref_teacher_weights.max()))
                w = self.teacher_buffer.pref_teacher_weights
                self._pending_reweight_w_min.append(float(w.min()))
                self._pending_reweight_w_mean.append(float(w.mean()))
                self._pending_reweight_w_std.append(float(w.std()))
                w_safe = np.clip(w, 1e-9, None)
                self._pending_reweight_w_entropy.append(float(-np.sum(w_safe * np.log(w_safe))))
                j_scores = [ep['J'] for ep in self.teacher_buffer.pref_episodes]
                if j_scores:
                    self._pending_reweight_j_mean.append(float(np.mean(j_scores)))
                    self._pending_reweight_j_std.append(float(np.std(j_scores)))
                    self._pending_reweight_j_min.append(float(np.min(j_scores)))
                    self._pending_reweight_j_max.append(float(np.max(j_scores)))

            if self.debug:
                pref_str = f" pref_loss={pref_loss_val:.4f} " if self.pref_rank_disc else " "
                print(
                    f"[DISC] expert_logits  mean={_e_logits.mean():.3f} std={_e_logits.std():.3f} "
                    f"prob={torch.sigmoid(_e_logits).mean():.3f}"
                )
                print(
                    f"[DISC] policy_logits  mean={_p_logits.mean():.3f} std={_p_logits.std():.3f} "
                    f"prob={torch.sigmoid(_p_logits).mean():.3f}"
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

        # Online RM: train on schedule (passive in phase 3; drives reweighting in phase 4)
        if self.online_rm_manager is not None and self.num_timesteps >= self.learning_starts:
            self.online_rm_manager.maybe_train(self.num_timesteps)

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

            qf1_loss = F.mse_loss(current_q1, target_q)
            qf2_loss = F.mse_loss(current_q2, target_q)
            critic_loss = qf1_loss + qf2_loss
            self._pending_qf1_losses.append(qf1_loss.item())
            self._pending_qf2_losses.append(qf2_loss.item())
            self._pending_q1_mean.append(current_q1.detach().mean().item())
            self._pending_q2_mean.append(current_q2.detach().mean().item())
            self._pending_q_target_mean.append(target_q.mean().item())
            self._pending_q_target_std.append(target_q.std().item())

            # ---- QPREF: trajectory-mean-Q preference ranking loss ----
            # Added to the TD critic loss; backprops through the same critic weights.
            # Uses min(q1, q2) at every valid timestep, masked mean per trajectory,
            # then Bradley-Terry softplus ranking loss (TF-style but mean-Q instead of
            # random single-step Q).
            qpref_would_run = (
                self.qpref
                and self.qpref_weight > 0.0
                and self.qpref_mean_trajectory_q
                and self.num_timesteps >= self.qpref_start_step
                and gradient_step % self.qpref_grad_interval == 0
            )
            if qpref_would_run and not self._qpref_active:
                self._qpref_skipped_updates += 1
            qpref_active = qpref_would_run and self._qpref_active
            if qpref_active:
                pair = self.teacher_buffer.sample_qpref_pairs_aggregate(
                    self.qpref_batch_size, source=self.qpref_source)
                if pair is not None:
                    p_obs, p_acs, p_mask, n_obs, n_acs, n_mask = pair
                    # p_obs: [B, T_max, obs_dim], flatten for batch critic eval
                    B, T_max, obs_dim = p_obs.shape
                    act_dim = p_acs.shape[-1]

                    q1_p, q2_p = self.critic(
                        p_obs.reshape(B * T_max, obs_dim),
                        p_acs.reshape(B * T_max, act_dim))
                    q1_n, q2_n = self.critic(
                        n_obs.reshape(B * T_max, obs_dim),
                        n_acs.reshape(B * T_max, act_dim))

                    # min(q1, q2) per timestep  →  [B, T_max]
                    q_pos_flat = torch.minimum(q1_p, q2_p).reshape(B, T_max)
                    q_neg_flat = torch.minimum(q1_n, q2_n).reshape(B, T_max)

                    # Masked mean over valid timesteps  →  [B]
                    q_pos_mean = (q_pos_flat * p_mask).sum(dim=1) / \
                                 (p_mask.sum(dim=1) + 1e-8)
                    q_neg_mean = (q_neg_flat * n_mask).sum(dim=1) / \
                                 (n_mask.sum(dim=1) + 1e-8)

                    delta = (q_pos_mean - q_neg_mean) / self.qpref_temp  # [B]
                    qpref_loss = F.softplus(-delta).mean()                # scalar ≥ 0

                    critic_loss = critic_loss + self.qpref_weight * qpref_loss

                    # Accumulate for logging
                    self._pending_qpref_losses.append(qpref_loss.item())
                    self._pending_qpref_q_pos.append(q_pos_mean.mean().item())
                    self._pending_qpref_q_neg.append(q_neg_mean.mean().item())
                    self._pending_qpref_delta.append(delta.mean().item())
                    self._pending_qpref_pairs_ok.append(1)
                else:
                    self._pending_qpref_pairs_ok.append(0)
            # ---- end QPREF ----

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

        # ------- Online RM: rescore pref_episodes if pending -------
        if (self.online_rm_manager is not None
                and self.online_rm_manager.pending_rescore
                and self.online_rm_manager.is_active):
            n = self.online_rm_manager.rescore_pref_episodes(
                self.teacher_buffer.pref_episodes)
            if n > 0:
                # Recompute Boltzmann weights with updated J scores
                self.teacher_buffer._recompute_pref_weights()
                if self.teacher_buffer.pref_episodes:
                    j_scores = [ep['J'] for ep in self.teacher_buffer.pref_episodes]
                    self.logger.record("pref_reweight/j_score_std",
                                       float(np.std(j_scores)))
                    self.logger.record("pref_reweight/rescore_count",
                                       self.online_rm_manager._rescore_count)

            # Rescore soft_tac_pool on the same schedule (same pending_rescore trigger).
            # Expert and student J values are all moved to RM space together.
            # This also sets _soft_tac_pool_rescored=True on first call, unlocking the gate.
            if self.soft_tac:
                n_tac = self.teacher_buffer.rescore_soft_tac_pool(self.online_rm_manager)
                if n_tac > 0:
                    self.logger.record("soft_tac/rescore_count",
                                       self.teacher_buffer._soft_tac_rescore_count)
                    j_vals = [ep['J'] for ep in self.teacher_buffer.soft_tac_pool]
                    if j_vals:
                        self.logger.record("soft_tac/j_std_after_rescore",
                                           float(np.std(j_vals)))
                        self.logger.record("soft_tac/j_mean_after_rescore",
                                           float(np.mean(j_vals)))

        # ------- Logging -------
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # Drain disc losses accumulated across the two _store_transition disc fires
        if self._pending_disc_losses:
            self.logger.record("train/disc_loss", np.mean(self._pending_disc_losses))
            self._pending_disc_losses = []
        if self._pending_pref_losses:
            self.logger.record("train/pref_loss", np.mean(self._pending_pref_losses))
            self._pending_pref_losses = []

        # ---- Discriminator internals ----
        if self._pending_disc_e_bce:
            self.logger.record("train/disc_loss_exp",        float(np.mean(self._pending_disc_e_bce)))
            self.logger.record("train/disc_loss_gen",        float(np.mean(self._pending_disc_p_bce)))
            self.logger.record("train/disc_entropy",         float(np.mean(self._pending_disc_entropy)))
            self.logger.record("train/disc_grad_penalty",    float(np.mean(self._pending_disc_gp)))
        self._pending_disc_e_bce = []; self._pending_disc_p_bce = []
        self._pending_disc_entropy = []; self._pending_disc_gp = []
        if self._pending_disc_logits_exp:
            self.logger.record("train/disc_logits_exp_mean", float(np.mean(self._pending_disc_logits_exp)))
            self.logger.record("train/disc_logits_pol_mean", float(np.mean(self._pending_disc_logits_pol)))
            self.logger.record("train/disc_prob_exp_mean",   float(np.mean(self._pending_disc_prob_exp)))
            self.logger.record("train/disc_prob_pol_mean",   float(np.mean(self._pending_disc_prob_pol)))
            self.logger.record("train/disc_acc_exp",         float(np.mean(self._pending_disc_acc_exp)))
            self.logger.record("train/disc_acc_pol",         float(np.mean(self._pending_disc_acc_pol)))
            self.logger.record("train/disc_reward_exp_mean", float(np.mean(self._pending_disc_reward_exp)))
            self.logger.record("train/disc_reward_pol_mean", float(np.mean(self._pending_disc_reward_pol)))
        self._pending_disc_logits_exp = []; self._pending_disc_logits_pol = []
        self._pending_disc_prob_exp = []; self._pending_disc_prob_pol = []
        self._pending_disc_acc_exp = []; self._pending_disc_acc_pol = []
        self._pending_disc_reward_exp = []; self._pending_disc_reward_pol = []

        if self._pending_reweight_pool_size:
            self.logger.record("pref_reweight/pool_size",
                               int(np.mean(self._pending_reweight_pool_size)))
            self.logger.record("pref_reweight/weight_max",
                               float(np.mean(self._pending_reweight_w_max)))
            if self._pending_reweight_w_min:
                self.logger.record("pref_reweight/weight_min",  float(np.mean(self._pending_reweight_w_min)))
                self.logger.record("pref_reweight/weight_mean", float(np.mean(self._pending_reweight_w_mean)))
                self.logger.record("pref_reweight/weight_std",  float(np.mean(self._pending_reweight_w_std)))
                self.logger.record("pref_reweight/entropy",     float(np.mean(self._pending_reweight_w_entropy)))
            if self._pending_reweight_j_mean:
                self.logger.record("pref_reweight/j_mean", float(np.mean(self._pending_reweight_j_mean)))
                self.logger.record("pref_reweight/j_std",  float(np.mean(self._pending_reweight_j_std)))
                self.logger.record("pref_reweight/j_min",  float(np.mean(self._pending_reweight_j_min)))
                self.logger.record("pref_reweight/j_max",  float(np.mean(self._pending_reweight_j_max)))
            self._pending_reweight_pool_size = []
            self._pending_reweight_w_max = []
            self._pending_reweight_w_min = []; self._pending_reweight_w_mean = []
            self._pending_reweight_w_std = []; self._pending_reweight_w_entropy = []
            self._pending_reweight_j_mean = []; self._pending_reweight_j_std = []
            self._pending_reweight_j_min = []; self._pending_reweight_j_max = []
        self.logger.record("train/surrogate_reward_mean", np.mean(sr_list))
        self.logger.record("train/surrogate_reward_std",  np.std(sr_list))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        # ---- Critic internals ----
        if self._pending_qf1_losses:
            self.logger.record("train/qf1_loss",      float(np.mean(self._pending_qf1_losses)))
            self.logger.record("train/qf2_loss",      float(np.mean(self._pending_qf2_losses)))
            self.logger.record("train/q1_mean",       float(np.mean(self._pending_q1_mean)))
            self.logger.record("train/q2_mean",       float(np.mean(self._pending_q2_mean)))
            self.logger.record("train/q_target_mean", float(np.mean(self._pending_q_target_mean)))
            self.logger.record("train/q_target_std",  float(np.mean(self._pending_q_target_std)))
        self._pending_qf1_losses = []; self._pending_qf2_losses = []
        self._pending_q1_mean = []; self._pending_q2_mean = []
        self._pending_q_target_mean = []; self._pending_q_target_std = []

        # ---- LfD mixing ----
        self.logger.record("train/lfd_active", int(lfd_active))

        # ---- QPREF logging + guard check ----
        if self.qpref:
            n_teacher = len(self.teacher_buffer.pref_episodes)
            n_student = len(self.teacher_buffer.pref_student_episodes)
            self.logger.record("train/qpref_source_teacher_pool_size", n_teacher)
            self.logger.record("train/qpref_source_student_pool_size", n_student)
            pairs_avail = int(np.sum(self._pending_qpref_pairs_ok)) if \
                self._pending_qpref_pairs_ok else 0
            self.logger.record("train/qpref_pairs_available", pairs_avail)
            self.logger.record("train/qpref_active",           int(self._qpref_active))
            self.logger.record("train/qpref_guard_triggered",  int(self._qpref_guard_triggered))
            self.logger.record("train/qpref_guard_count",      self._qpref_guard_count)
            self.logger.record("train/qpref_skipped_updates",  self._qpref_skipped_updates)
            if self._pending_qpref_losses:
                interval_mean_delta = float(np.mean(self._pending_qpref_delta))
                self.logger.record("train/qpref_loss",
                                   float(np.mean(self._pending_qpref_losses)))
                self.logger.record("train/qpref_mean_q_pos",
                                   float(np.mean(self._pending_qpref_q_pos)))
                self.logger.record("train/qpref_mean_q_neg",
                                   float(np.mean(self._pending_qpref_q_neg)))
                self.logger.record("train/qpref_delta",        interval_mean_delta)
                self.logger.record("train/qpref_delta_std",
                                   float(np.std(self._pending_qpref_delta)))
                self._qpref_delta_deque.append(interval_mean_delta)
            if self._qpref_delta_deque:
                deque_list   = list(self._qpref_delta_deque)
                rolling_mean = float(np.mean(deque_list))
                recent_entries = deque_list[-self.qpref_guard_window:]
                guard_mean     = float(np.mean(recent_entries))
                self._qpref_peak_guard_mean = max(self._qpref_peak_guard_mean, guard_mean)
                self.logger.record("train/qpref_delta_rolling_mean", rolling_mean)
                self.logger.record("train/qpref_delta_guard_mean",   guard_mean)
                self.logger.record("train/qpref_peak_guard_mean",    self._qpref_peak_guard_mean)
                self.logger.record("train/qpref_deque_fill",
                                   len(self._qpref_delta_deque))
                deque_full    = (len(self._qpref_delta_deque)
                                 == self._qpref_delta_deque.maxlen)
                peak_positive = (self._qpref_peak_guard_mean
                                 >= self.qpref_guard_positive_threshold)
                if (self._qpref_active
                        and not self._qpref_guard_triggered
                        and deque_full
                        and peak_positive):
                    if guard_mean < self.qpref_guard_threshold:
                        self._qpref_guard_count += 1
                    else:
                        self._qpref_guard_count = 0
                    if self._qpref_guard_count >= self.qpref_guard_confirm:
                        self._qpref_active          = False
                        self._qpref_guard_triggered = True
                        print(
                            f"[QPREF GUARD] Triggered at step {self.num_timesteps}: "
                            f"guard mean delta {guard_mean:.4f} < "
                            f"{self.qpref_guard_threshold} for "
                            f"{self.qpref_guard_confirm} consecutive intervals "
                            f"(recent window={len(recent_entries)}/{self.qpref_guard_window} entries, "
                            f"full deque={len(self._qpref_delta_deque)}, "
                            f"peak_guard_mean={self._qpref_peak_guard_mean:.4f}). "
                            f"QPREF permanently disabled.",
                            flush=True,
                        )
                        self.logger.record("train/qpref_active",          0)
                        self.logger.record("train/qpref_guard_triggered", 1)
                        self.logger.record("train/qpref_guard_count",     self._qpref_guard_count)
            self._pending_qpref_losses   = []
            self._pending_qpref_q_pos    = []
            self._pending_qpref_q_neg    = []
            self._pending_qpref_delta    = []
            self._pending_qpref_pairs_ok = []

        # ---- Online RM logging ----
        if self.online_rm_manager is not None:
            for k, v in self.online_rm_manager.get_metrics().items():
                if np.isfinite(v) or isinstance(v, int):
                    self.logger.record(k, v)

        if self.soft_tac and self._pending_soft_tac_losses:
            self.logger.record("train/soft_tac_loss",
                               float(np.mean(self._pending_soft_tac_losses)))
            self.logger.record("train/tac_alignment",
                               float(np.mean(self._pending_tac_alignments)))
        self._pending_soft_tac_losses = []
        self._pending_tac_alignments  = []
        # Soft-TAC label distribution: counts + fractions
        if self.soft_tac:
            if self._pending_tac_label_pos:
                n_pos  = int(np.sum(self._pending_tac_label_pos))
                n_neg  = int(np.sum(self._pending_tac_label_neg))
                n_tie  = int(np.sum(self._pending_tac_label_zero))
                n_tot  = max(n_pos + n_neg + n_tie, 1)
                self.logger.record("soft_tac/n_pos",      n_pos)
                self.logger.record("soft_tac/n_neg",      n_neg)
                self.logger.record("soft_tac/n_tie",      n_tie)
                self.logger.record("soft_tac/y_pos_frac", n_pos / n_tot)
                self.logger.record("soft_tac/y_neg_frac", n_neg / n_tot)
                self.logger.record("soft_tac/y_tie_frac", n_tie / n_tot)
            self._pending_tac_label_pos  = []
            self._pending_tac_label_neg  = []
            self._pending_tac_label_zero = []
            self.logger.record("soft_tac/pool_size",
                               len(self.teacher_buffer.soft_tac_pool))
            self.logger.record("soft_tac/student_pool_size",
                               len(self.teacher_buffer.soft_tac_student_episodes))
        # pref_rank_disc J-diff stats: magnitude of J_pos - J_neg across all pairs this window
        if self.pref_rank_disc and self._pending_pref_j_diffs:
            diffs = np.array(self._pending_pref_j_diffs, dtype=np.float64)
            self.logger.record("pref_rank/j_diff_mean",   float(np.mean(diffs)))
            self.logger.record("pref_rank/j_diff_std",    float(np.std(diffs)))
            self.logger.record("pref_rank/pairs_sampled", int(len(diffs)))
            self.logger.record("pref_rank/pool_size",     len(self.teacher_buffer.pref_episodes))
            self.logger.record("train/pref_pool_size",    len(self.teacher_buffer.pref_episodes))
        self._pending_pref_j_diffs = []

    def _dump_logs(self) -> None:
        """Override SB3's _dump_logs to append normalized score metrics.

        Calls the parent first (which writes rollout/ep_rew_mean, time/fps, etc.),
        then records rollout/normalized_score and rollout/normalized_score_pct if
        expert_return is set.  The parent already calls logger.dump(), so we only
        need to record() here — the values are flushed by the parent's dump().
        """
        # Let SB3 handle all standard logging + logger.dump()
        super()._dump_logs()

        # Normalized score (additive only; no-op if expert_return is None)
        if self.expert_return is not None and len(self.ep_info_buffer) > 0:
            ep_rew = float(safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            if np.isfinite(ep_rew) and self.expert_return != 0.0:
                ns = ep_rew / self.expert_return
                self.logger.record("rollout/normalized_score",     float(ns))
                self.logger.record("rollout/normalized_score_pct", float(ns * 100.0))
                # Flush the two new records immediately (parent already dumped)
                self.logger.dump(step=self.num_timesteps)
