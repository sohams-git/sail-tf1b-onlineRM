# W&B / Logger Logging Upgrade — Analysis

**Date**: 2026-04-15  
**Files changed**: `sail_sb3/algorithms/sail.py`, `sail_sb3_online/algorithms/sail.py`, `sail_sb3/utils/callbacks.py`, `sail_sb3_online/utils/callbacks.py`

---

## Section 1: What Was Logged Before (Both Files)

### sail_sb3 (offline RM variant) — pre-upgrade metrics

| Group | Metric | Notes |
|-------|--------|-------|
| `train` | `n_updates` | Excluded from tensorboard |
| `train` | `disc_loss` | Mean of pending losses since last train() |
| `train` | `pref_loss` | Hard BT preference ranking loss (if pref_rank_disc) |
| `pref_reweight` | `pool_size` | Weighted teacher pool size |
| `pref_reweight` | `weight_max` | Max Boltzmann weight in teacher pool |
| `train` | `surrogate_reward_mean` | Mean discriminator reward over critic batch |
| `train` | `surrogate_reward_std` | Std of discriminator reward |
| `train` | `actor_loss` | TD3 actor loss (when policy_delay step) |
| `train` | `critic_loss` | Combined qf1+qf2 MSE loss |
| `train` | `qpref_source_teacher_pool_size` | If qpref enabled |
| `train` | `qpref_source_student_pool_size` | If qpref enabled |
| `train` | `qpref_pairs_available` | Count of gradient steps with valid pairs |
| `train` | `qpref_loss` | Bradley-Terry softplus loss |
| `train` | `qpref_mean_q_pos` | Mean Q of positive (higher-J) trajectory |
| `train` | `qpref_mean_q_neg` | Mean Q of negative trajectory |
| `train` | `qpref_delta` | Mean (Q_pos - Q_neg) / temp |
| `train` | `soft_tac_loss` | Weighted soft-TAC loss |
| `train` | `tac_alignment` | Fraction of pairs aligned with RM labels |
| `train` | `rm_critic_reward_mean` | Mean RM reward when rm_in_critic active |

### sail_sb3_online (online RM variant) — additional pre-upgrade metrics

All of the above minus `rm_critic_reward_mean`, plus:

| Group | Metric | Notes |
|-------|--------|-------|
| `soft_tac` | `n_pos` | Count of +1 labels per train() window |
| `soft_tac` | `n_neg` | Count of -1 labels |
| `soft_tac` | `n_tie` | Count of 0 labels |
| `pref_rank` | `j_diff_mean` | Mean J_pos - J_neg across pairs |
| `pref_rank` | `j_diff_std` | Std of J-diffs |
| `pref_rank` | `pairs_sampled` | Total pairs used this window |
| `pref_reweight` | `j_score_std` | Std of J scores after RM rescore |
| `pref_reweight` | `rescore_count` | Cumulative online RM rescore count |
| `online_rm/*` | (various) | From `online_rm_manager.get_metrics()` |

---

## Section 2: What Is Added by This Upgrade

### A. Both variants (sail_sb3 and sail_sb3_online)

**Discriminator component losses** (from `compute_loss()` return values):
- `train/disc_loss_exp` — Expert BCE component
- `train/disc_loss_gen` — Policy/generator BCE component  
- `train/disc_entropy` — Entropy regularization term
- `train/disc_grad_penalty` — WGAN gradient penalty term

**Discriminator logits, probabilities, and implicit rewards** (always-on forward pass after optimizer step):
- `train/disc_logits_exp_mean` — Mean logit on expert batch
- `train/disc_logits_pol_mean` — Mean logit on policy batch
- `train/disc_prob_exp_mean` — Mean sigmoid(logit) on expert batch (should be > 0.5)
- `train/disc_prob_pol_mean` — Mean sigmoid(logit) on policy batch (should be < 0.5)
- `train/disc_acc_exp` — Fraction of expert samples classified correctly (prob > 0.5)
- `train/disc_acc_pol` — Fraction of policy samples classified correctly (prob < 0.5)
- `train/disc_reward_exp_mean` — Mean -log(1-p+1e-8) reward on expert batch
- `train/disc_reward_pol_mean` — Mean -log(1-p+1e-8) reward on policy batch

**Critic internals**:
- `train/qf1_loss` — QF1 MSE loss independently
- `train/qf2_loss` — QF2 MSE loss independently
- `train/q1_mean` — Mean Q1 prediction on current batch
- `train/q2_mean` — Mean Q2 prediction on current batch
- `train/q_target_mean` — Mean Bellman target value
- `train/q_target_std` — Std of Bellman target values

**QPREF delta spread**:
- `train/qpref_delta_std` — Std of (Q_pos - Q_neg)/temp across the train() window

**Pref reweight extended weight stats**:
- `pref_reweight/weight_min` — Min Boltzmann weight
- `pref_reweight/weight_mean` — Mean Boltzmann weight
- `pref_reweight/weight_std` — Std of Boltzmann weights
- `pref_reweight/entropy` — Weight distribution entropy (-sum(w * log(w)))

**Pref reweight J-score stats**:
- `pref_reweight/j_mean` — Mean J across pref_episodes pool
- `pref_reweight/j_std` — Std of J scores
- `pref_reweight/j_min` — Min J score in pool
- `pref_reweight/j_max` — Max J score in pool

**LfD mixing indicator**:
- `train/lfd_active` — 1 if LfD mixing is currently active, 0 otherwise

**Soft-TAC label fractions** (both files; fractions are new, counts existed only in online):
- `soft_tac/n_pos`, `soft_tac/n_neg`, `soft_tac/n_tie` — Absolute counts
- `soft_tac/y_pos_frac`, `soft_tac/y_neg_frac`, `soft_tac/y_tie_frac` — Fractions summing to 1.0

**PrefRank J-diff pool size** (both files; existing metrics augmented):
- `pref_rank/pool_size` — Number of episodes in pref pool
- `train/pref_pool_size` — Same, under the train/ prefix for dashboard convenience

**Callbacks event counters** (both files):
- `events/promotions_total` — Cumulative student promotions
- `events/first_promotion_step` — Timestep of first promotion (-1 until first)
- `events/episodes_completed` — Cumulative completed episodes (logged every episode)
- `adaptive/has_promotions` — 1/0 flag from teacher_buffer._has_promotions

**Callbacks adaptive buffer stats** (both files, logged on each promotion):
- `adaptive/expert_scores_min` — Min score in FIFO threshold window
- `adaptive/expert_scores_max` — Max score in threshold window
- `adaptive/expert_scores_mean` — Mean score in threshold window
- `adaptive/teacher_buffer_capacity` — max_size if set
- `adaptive/teacher_buffer_fill_ratio` — current_size / max_size

### B. sail_sb3 only (backport from online + new RM-in-critic metrics)

**Soft-TAC labels** (backport from online):
- `_pending_tac_label_pos/neg/zero` accumulators added; logging as above in Section A

**PrefRank J-diff stats** (backport from online):
- `pref_rank/j_diff_mean`, `pref_rank/j_diff_std`, `pref_rank/pairs_sampled`
- `want_J` condition now also includes `self.pref_rank_disc` (was only `soft_tac`)

**RM-in-critic extended stats** (replaces single `rm_critic_reward_mean`):
- `train/r_disc_mean` — Mean discriminator reward in mixed reward batch
- `train/r_disc_std` — Std of discriminator reward
- `train/r_rm_mean` — Mean RM reward
- `train/r_rm_std` — Std of RM reward
- `train/r_mix_mean` — Mean blended reward r_mix = (1-alpha)*r_disc + alpha*r_rm
- `train/r_mix_std` — Std of blended reward
- `train/rm_critic_reward_mean` — Kept for backward compatibility (= r_rm_mean)
- `train/rm_in_critic_alpha` — Current alpha blend value
- `train/rm_in_critic_enabled` — Always 1 when rm_in_critic=True

### C. sail_sb3_online only (no rm_in_critic; RM-in-critic section not added)

Online RM-in-critic feature does not exist in this variant; those metrics are not added.

---

## Section 3: Complete Metric Name Table

| Group | Metric Name | Description | sail_sb3 | sail_sb3_online |
|-------|-------------|-------------|----------|-----------------|
| train | n_updates | Total gradient steps | Y | Y |
| train | disc_loss | Mean total disc loss | Y | Y |
| train | pref_loss | Hard BT pref ranking loss | Y | Y |
| train | disc_loss_exp | Expert BCE component | Y (new) | Y (new) |
| train | disc_loss_gen | Policy BCE component | Y (new) | Y (new) |
| train | disc_entropy | Entropy regularization | Y (new) | Y (new) |
| train | disc_grad_penalty | WGAN-GP gradient penalty | Y (new) | Y (new) |
| train | disc_logits_exp_mean | Mean expert logit | Y (new) | Y (new) |
| train | disc_logits_pol_mean | Mean policy logit | Y (new) | Y (new) |
| train | disc_prob_exp_mean | Mean expert prob sigmoid | Y (new) | Y (new) |
| train | disc_prob_pol_mean | Mean policy prob sigmoid | Y (new) | Y (new) |
| train | disc_acc_exp | Expert classification acc | Y (new) | Y (new) |
| train | disc_acc_pol | Policy classification acc | Y (new) | Y (new) |
| train | disc_reward_exp_mean | Implicit reward on expert | Y (new) | Y (new) |
| train | disc_reward_pol_mean | Implicit reward on policy | Y (new) | Y (new) |
| train | surrogate_reward_mean | Mean disc reward in critic | Y | Y |
| train | surrogate_reward_std | Std disc reward in critic | Y | Y |
| train | actor_loss | TD3 actor loss | Y | Y |
| train | critic_loss | Combined QF1+QF2 loss | Y | Y |
| train | qf1_loss | QF1 MSE loss alone | Y (new) | Y (new) |
| train | qf2_loss | QF2 MSE loss alone | Y (new) | Y (new) |
| train | q1_mean | Mean current Q1 | Y (new) | Y (new) |
| train | q2_mean | Mean current Q2 | Y (new) | Y (new) |
| train | q_target_mean | Mean Bellman target | Y (new) | Y (new) |
| train | q_target_std | Std Bellman target | Y (new) | Y (new) |
| train | lfd_active | LfD mixing active flag | Y (new) | Y (new) |
| train | qpref_source_teacher_pool_size | Teacher pref pool size | Y | Y |
| train | qpref_source_student_pool_size | Student pref pool size | Y | Y |
| train | qpref_pairs_available | Steps with valid pairs | Y | Y |
| train | qpref_loss | BT softplus ranking loss | Y | Y |
| train | qpref_mean_q_pos | Mean Q positive traj | Y | Y |
| train | qpref_mean_q_neg | Mean Q negative traj | Y | Y |
| train | qpref_delta | Mean (Q_pos-Q_neg)/temp | Y | Y |
| train | qpref_delta_std | Std of qpref delta | Y (new) | Y (new) |
| train | soft_tac_loss | Weighted soft-TAC loss | Y | Y |
| train | tac_alignment | Fraction aligned with RM | Y | Y |
| train | pref_pool_size | Size of pref episode pool | Y (new) | Y (new) |
| train | rm_critic_reward_mean | Mean RM reward (compat) | Y | N/A |
| train | r_disc_mean | Mean disc reward component | Y (new) | N/A |
| train | r_disc_std | Std disc reward component | Y (new) | N/A |
| train | r_rm_mean | Mean RM reward component | Y (new) | N/A |
| train | r_rm_std | Std RM reward component | Y (new) | N/A |
| train | r_mix_mean | Mean blended reward | Y (new) | N/A |
| train | r_mix_std | Std blended reward | Y (new) | N/A |
| train | rm_in_critic_alpha | Alpha blend value | Y (new) | N/A |
| train | rm_in_critic_enabled | Feature enabled flag | Y (new) | N/A |
| pref_reweight | pool_size | Pref episodes count | Y | Y |
| pref_reweight | weight_max | Max Boltzmann weight | Y | Y |
| pref_reweight | weight_min | Min Boltzmann weight | Y (new) | Y (new) |
| pref_reweight | weight_mean | Mean Boltzmann weight | Y (new) | Y (new) |
| pref_reweight | weight_std | Std Boltzmann weights | Y (new) | Y (new) |
| pref_reweight | entropy | Weight entropy | Y (new) | Y (new) |
| pref_reweight | j_mean | Mean J score in pool | Y (new) | Y (new) |
| pref_reweight | j_std | Std J scores in pool | Y (new) | Y (new) |
| pref_reweight | j_min | Min J score in pool | Y (new) | Y (new) |
| pref_reweight | j_max | Max J score in pool | Y (new) | Y (new) |
| pref_reweight | j_score_std | Std J after RM rescore | N/A | Y |
| pref_reweight | rescore_count | Online RM rescore count | N/A | Y |
| pref_rank | j_diff_mean | Mean J_pos - J_neg | Y (new) | Y |
| pref_rank | j_diff_std | Std of J-diffs | Y (new) | Y |
| pref_rank | pairs_sampled | Total pairs this window | Y (new) | Y |
| pref_rank | pool_size | Pref episode pool size | Y (new) | Y (new) |
| soft_tac | n_pos | Count +1 labels | Y (new) | Y |
| soft_tac | n_neg | Count -1 labels | Y (new) | Y |
| soft_tac | n_tie | Count 0 labels | Y (new) | Y |
| soft_tac | y_pos_frac | Fraction of +1 labels | Y (new) | Y (new) |
| soft_tac | y_neg_frac | Fraction of -1 labels | Y (new) | Y (new) |
| soft_tac | y_tie_frac | Fraction of 0 labels | Y (new) | Y (new) |
| adaptive | teacher_buffer_size | Current transitions in buffer | Y | Y |
| adaptive | expert_threshold | Current promotion threshold | Y | Y |
| adaptive | promoted_episodes | Promoted episode count | Y | Y |
| adaptive | pref_pool_size | Pref episodes available | Y | Y |
| adaptive | student_gt_score | GT score of promoted ep | Y | Y |
| adaptive | student_rm_score | RM score of promoted ep | Y | Y |
| adaptive | rm_threshold | RM promotion threshold | Y | Y |
| adaptive | has_promotions | 1 if any promotion occurred | Y (new) | Y (new) |
| adaptive | expert_scores_min | Min score in FIFO window | Y (new) | Y (new) |
| adaptive | expert_scores_max | Max score in FIFO window | Y (new) | Y (new) |
| adaptive | expert_scores_mean | Mean score in FIFO window | Y (new) | Y (new) |
| adaptive | teacher_buffer_capacity | max_size if ring buffer | Y (new) | Y (new) |
| adaptive | teacher_buffer_fill_ratio | current/capacity | Y (new) | Y (new) |
| events | promotions_total | Cumulative promotions | Y (new) | Y (new) |
| events | first_promotion_step | Step of first promotion | Y (new) | Y (new) |
| events | episodes_completed | Cumulative completed eps | Y (new) | Y (new) |
| online_rm | (various) | From online_rm_manager | N/A | Y |

---

## Section 4: Metrics That Were Impossible to Add Cleanly

### 4.1 Discriminator gradient norm

**What**: The L2 norm of discriminator parameter gradients during the backward pass.

**Why not added**: `disc_optimizer.step()` is called immediately after `total_loss.backward()`, consuming the gradients. To log gradient norms, we would need to call `parameters_to_vector(disc.parameters())` or iterate over params to compute the norm *between* `backward()` and `step()`. This is a minor restructuring of `_update_discriminator()` that falls outside pure logging — it requires inserting code between `backward()` and `step()`.

### 4.2 Per-episode discriminator assignment J_disc vs J_gt

**What**: The correlation between discriminator-assigned cumulative reward and ground-truth episode return.

**Why not added**: Computing this requires storing discriminator rewards for entire episodes, which are not tracked at the per-episode level inside `train()`. The discriminator reward is computed per-transition in the critic batch; there is no episode boundary tracking at this layer. Adding it would require either: (a) a separate episodic forward pass outside the training loop, or (b) propagating episode-boundary information into `train()`, neither of which is logging-only.

### 4.3 Delta-J between consecutive discriminator updates

**What**: How much the discriminator's reward assignment changed between successive `_update_discriminator()` calls (to detect convergence or oscillation).

**Why not added**: This requires caching the previous reward outputs per fixed probe set across `_update_discriminator()` calls. There is no persistent probe set; adding one would require changes to `adversary.py` to expose a `probe_rewards()` method and changes to `__init__` to allocate the probe buffer.

### 4.4 Actor gradient norm and Q-gradient signal quality

**What**: Whether the critic's Q-gradient signal to the actor is healthy (not vanishing or exploding).

**Why not added**: Same structural reason as discriminator gradient norm — gradients are consumed by `actor_loss.backward()` + `actor.optimizer.step()` before any logging point. A hook-based approach would be needed, which falls outside pure logging.

### 4.5 Exact per-component discriminator loss breakdown in the pref_rank/soft_tac loss

**What**: The discriminator's GAIL loss vs. the pref_rank_loss vs. soft_tac_loss as separate W&B metrics from the same `total_loss`.

**Status**: The GAIL components (`e_bce`, `p_bce`, `entropy`, `gp`) are now logged via `train/disc_loss_exp`, `train/disc_loss_gen`, etc. However, the **weighted** pref_rank contribution (`pref_rank_weight * pref_loss`) and soft_tac contribution (`soft_tac_weight * tac_loss`) are already logged separately as `train/pref_loss` and `train/soft_tac_loss`. The only thing not logged is the pre-weight raw pref_loss/soft_tac_loss before the weight is applied — but those are trivially derivable by dividing the logged value by the (constant) weight hyperparameter, so it was not added.

### 4.6 Discriminator logit distribution (per-step histogram)

**What**: Full logit distribution (not just mean) as a W&B histogram.

**Why not added**: SB3's `logger.record()` only supports scalar values. W&B histograms require calling `wandb.log({"hist": wandb.Histogram(data)})` directly, bypassing the SB3 logger abstraction. Adding this would require detecting whether a W&B run is active and calling the W&B API directly — a larger invasive change incompatible with the "logging only, no behavior change" constraint.
