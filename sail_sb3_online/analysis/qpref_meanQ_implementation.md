# QPREF Mean-Q Implementation — PyTorch SAIL

**Date:** 2026-04-09  
**Status:** Implemented, sanity-checked, smoke tests submitted (jobs 47203383, 47203384)

---

## Overview

QPREF adds a Q-value preference ranking loss to the TD3 critic update. The loss pushes `Q(s+, a+) > Q(s-, a-)` for episode pairs where the positive trajectory has a higher RM-scored quality than the negative.

This implementation uses **trajectory-mean-Q** rather than TF's single random timestep, because the preference label J is trajectory-level — mean Q over all valid timesteps is a more consistent estimator.

---

## Design Choices vs. TF Reference

| Aspect | TF reference (`stable-baselines/stable_baselines/td3/sail.py`) | This implementation |
|--------|--------------------------------------------------------------|---------------------|
| Q estimator | One random timestep per trajectory | Masked mean over all timesteps |
| Loss | `softplus(-(q_pos - q_neg) / T)` | Same formula, applied to mean Q |
| Preference pool (teacher) | Shared with pref_rank_disc | Reuses `pref_episodes` |
| Student pool | Built episode-by-episode in rollouts | `pref_student_episodes`, recency-pruned |
| Gating | Always feed dict; weight=0 disables | `qpref_active` condition check |
| Firing frequency | Every critic gradient step | Every `qpref_grad_interval`-th step (default 10) |

---

## Loss Formula

```
pair = sample_qpref_pairs_aggregate(B, source)   # B trajectory pairs
q_pos_flat = min(q1, q2)(pos_obs, pos_acs)       # [B*T_max] → [B, T_max]
q_neg_flat = min(q1, q2)(neg_obs, neg_acs)

q_pos_mean = masked_mean(q_pos_flat, pos_mask)   # [B]
q_neg_mean = masked_mean(q_neg_flat, neg_mask)   # [B]

delta = (q_pos_mean - q_neg_mean) / T            # Bradley-Terry scaled delta
qpref_loss = mean(softplus(-delta))              # BT ranking loss

critic_loss += λ * qpref_loss
```

The masked mean uses `(q * mask).sum(1) / (mask.sum(1) + 1e-8)` to handle variable-length episodes padded to T_max.

---

## Preference Pools

### Teacher pool (`pref_episodes`)
- Built at startup from **all 4 expert episodes**, before ring buffer truncation (SepPool pattern)
- Each entry: `{'obs': Tensor[T,18], 'acs': Tensor[T,6], 'J': float}` where J = cumulative RM score
- Also used by `pref_rank_disc` and `pref_reweight_teacher`
- Grows as adaptive promotion adds student episodes (via `add_episode`)
- Stays at ≥4 entries always

### Student pool (`pref_student_episodes`)
- Built incrementally: `SAILAdaptiveCallback` calls `add_student_episode(obs_ep, acs_ep)` at every episode end when `qpref_source='student'`
- Each entry scored by `pref_rm.reward(obs_ep, acs_ep)` at insert time
- Pruned by **recency**: keep last `pref_max_student_trajs` (default 500), TF parity (`_add_student_episode_to_pref_buffer` line 776)
- No quality filter — all student episodes included regardless of quality

---

## Files Changed

### `sail_sb3/datasets/teacher_buffer.py`
- `__init__`: added `pref_max_student_trajs: int = 500` param; initialized `self.pref_student_episodes = []`
- `add_student_episode(obs_ep, acs_ep)`: scores episode with pref_rm, appends to pool, prunes by recency
- `sample_qpref_pairs_aggregate(batch_size, source)`: samples B pairs from teacher or student pool; returns `(pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask)` all `[B, T_max, dim]` or `None` if < 2 episodes

### `sail_sb3/algorithms/sail.py`
- `__init__`: 8 new params (`qpref`, `qpref_weight`, `qpref_temp`, `qpref_batch_size`, `qpref_start_step`, `qpref_source`, `qpref_mean_trajectory_q`, `qpref_grad_interval`); 5 pending accumulator lists
- `train()`: QPREF block inside critic gradient loop, gated by `qpref_active` condition; accumulates metrics per firing; logs 7 metrics at end of `train()` call
- Gating condition: `qpref and qpref_weight > 0 and qpref_mean_trajectory_q and num_timesteps >= qpref_start_step and gradient_step % qpref_grad_interval == 0`

### `sail_sb3/scripts/train_sail.py`
- 8 new argparse flags (see CLI flags below)
- `need_pref_rm` updated to include `args.qpref`
- `TeacherBuffer()` call passes `pref_max_student_trajs`
- QPREF startup print block (pool size, warn if < 2 teacher eps)
- `SAIL(...)` call passes all 8 QPREF params including `qpref_grad_interval`

### `sail_sb3/utils/callbacks.py`
- `SAILAdaptiveCallback.__init__`: added `qpref_source: str = "teacher"` param
- At episode end: if `qpref_source == "student"`, calls `teacher_buffer.add_student_episode(obs_ep, acs_ep)`

---

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--qpref` | bool | False | Enable QPREF loss |
| `--qpref_weight` | float | 0.0 (→0.1 if --qpref) | λ weight on QPREF loss term |
| `--qpref_temp` | float | 1.0 | Temperature T in Bradley-Terry delta |
| `--qpref_batch_size` | int | 4 | Episode pairs per QPREF firing |
| `--qpref_start_step` | int | 0 | Delay QPREF until this many env steps |
| `--qpref_source` | str | "teacher" | Pool to sample from: "teacher" or "student" |
| `--pref_max_student_trajs` | int | 500 | Max student episodes to retain in student pool |
| `--qpref_grad_interval` | int | 10 | Fire QPREF every Nth gradient step (100×/train() at 1000 grad_steps) |

`--qpref_mean_trajectory_q` is always True in this implementation (hardcoded in SAIL constructor call); the random-single-step variant is not implemented.

---

## Logged Metrics

All metrics logged under `train/`:

| Metric | What it measures |
|--------|-----------------|
| `train/qpref_loss` | Mean QPREF loss per train() call (lower = pairs well-ranked) |
| `train/qpref_pairs_available` | Number of firings where `sample_qpref_pairs_aggregate` returned a valid pair |
| `train/qpref_source_teacher_pool_size` | `len(pref_episodes)` at end of train() |
| `train/qpref_source_student_pool_size` | `len(pref_student_episodes)` at end of train() |
| `train/qpref_mean_q_pos` | Mean Q over positive trajectories (batch-averaged) |
| `train/qpref_mean_q_neg` | Mean Q over negative trajectories (batch-averaged) |
| `train/qpref_delta` | Mean `(q_pos - q_neg) / T`; positive = model is learning the right direction |

**Healthy signal:** `qpref_delta` should increase over training as the critic learns to assign higher Q to preferred trajectories. `qpref_loss` should decrease toward `log(2) ≈ 0.693` (random-chance baseline) and ideally below.

---

## Performance Notes

**CPU cost:** With `gradient_steps=1000` (default) and `qpref_batch_size=4`, each QPREF firing processes `[4000, 24]` critic forward passes. Without `qpref_grad_interval`, this adds 1000 firings × 4000 forward passes per `train()` call (~10× wall-clock slowdown on CPU).

**With `qpref_grad_interval=10` (default):** 100 firings per `train()` call → overhead reduced ~10×. At 20fps before QPREF, this gives ~15–18fps with QPREF active. For 100k steps on CPU, expected runtime ~5–8h.

---

## Smoke Tests

| Job ID | Config | Seeds | Steps | Status |
|--------|--------|-------|-------|--------|
| 47203383 | QPREF + GT-Adaptive | 1–2 | 100k | Submitted |
| 47203384 | QPREF + GT-Adaptive + PrefRank | 1–2 | 100k | Submitted |

**What to check in logs:**
1. `[train_sail] QPREF: ...  teacher_pool=4 eps` at startup (confirms SepPool loaded)
2. `train/qpref_source_teacher_pool_size=4` at step 10k (first train call)
3. `train/qpref_pairs_available > 0` — pairs are being sampled
4. `train/qpref_loss` appears and is finite (0.5–0.9 range expected at start)
5. `train/qpref_delta` moving toward positive over time
6. `rollout/ep_rew_mean` should follow the same ~100k trajectory as `HC_TFParity_Adaptive` seeds

---

## Known Assumptions

1. `qpref_mean_trajectory_q=True` is required (the only implemented path). Passing `False` skips QPREF silently even if `--qpref` is set.
2. Teacher pool is populated at startup; QPREF is active immediately (unless `--qpref_start_step` delays it). No cold-start issue for teacher source.
3. Student source requires `qpref_source=student` AND `SAILAdaptiveCallback` in the training loop (automatic when `--adaptive` is set).
4. The student pool has no quality filter — early random-policy episodes will be included. This may cause noisy gradients early in training.

---

## Next Steps (Not Yet Done)

- [ ] Run 1M jobs once smoke tests confirm stability
- [ ] Evaluate `qpref_source=student` (requires adaptive promotion for interesting diversity)
- [ ] Tune `qpref_weight` (0.1 is a reasonable starting point; range 0.05–0.5 to explore)
- [ ] Check interaction with `pref_rank_disc`: two ranking losses simultaneously may need weight balancing
