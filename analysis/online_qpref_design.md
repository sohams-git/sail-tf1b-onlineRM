# Online QPREF Design — How It Works
**Codebase:** `sail_sb3_online` | **Date:** 2026-04-18  
**Config used:** `--qpref --qpref_source student --online_rm`

---

## 1. What QPREF Does

QPREF (Q-preference ranking) adds a soft ranking loss to the TD3 critic. The intuition: if episode A is better than episode B (higher return), the critic's value estimate for A's trajectories should be higher than for B's. The loss penalises violations of this ordering:

```
loss = mean( softplus( -(mean_Q_pos - mean_Q_neg) / T ) )
```

where `mean_Q_pos` = masked mean of `min(Q1, Q2)` over all valid timesteps in the higher-J episode, and `mean_Q_neg` = the same for the lower-J episode. `T` is temperature. The loss is zero when `mean_Q_pos >> mean_Q_neg` and positive when the critic inverts the ordering.

This loss is **added to the standard TD3 critic MSE loss**, so it backprops through the same critic weights. It does not touch the actor or discriminator.

---

## 2. Episode Pools: Two Sources

`qpref_source` controls which episode pool QPREF samples pairs from.

### `source='teacher'` (pref_episodes)

Pool contents: 1 ring-buffered expert episode (loaded at startup) + all episodes promoted by the adaptive SAIL quality gate.

J labels: GT at startup → overwritten by online RM on each rescore event (~every 20k steps, piggybacked on `pending_rescore`). By the time QPREF can sample valid pairs (requires N≥2, i.e. first promotion at ~280–430k), the J values have been RM-rescored ~15–20 times.

**Implication:** Teacher-source QPREF pairs are always RM-scored in the online setting before QPREF can fire.

### `source='student'` (pref_student_episodes) — online RM version

Pool contents: every completed rollout episode, recency-pruned to last 500. J labels are **online RM scores, no GT used**.

This is the newly implemented path. It differs from the offline version (which used a frozen offline pref_rm at insertion time) in two ways:
1. J is computed by the online RM, which improves over training.
2. There is no GT fallback: episodes are skipped before RM activation.

---

## 3. Exact Data Flow — Student Source

```
Episode ends
    │
    ▼
SAILAdaptiveCallback._on_step()
    │
    ├── online_rm_manager.is_active?
    │       NO  → skip (pool stays empty; no GT fallback)
    │       YES →
    │           rm_j = sum(online_rm_manager.reward(obs_ep, acs_ep))
    │           teacher_buffer.add_student_episode(obs_ep, acs_ep, J=rm_j)
    │               → pref_student_episodes.append({'obs', 'acs', 'J': rm_j})
    │               → recency-prune to last 500
    │
    └── (separately) online_rm_manager.add_student_episode(obs, acs, rew)
            → feeds segments into the RM training buffer
```

J is a point-in-time score: the RM's assessment of the episode quality at the moment the episode was collected. The pool is not retroactively rescored as the RM improves — the J label is frozen at insertion. This is intentional: retroactive rescoring would require storing all 500 episodes' obs/acs tensors and running a full rescore on each RM update, which is expensive and logistically equivalent to a second `rescore_soft_tac_pool` call. The recency-500 window naturally discards old episodes as the policy improves.

---

## 4. Timeline: When Does QPREF Start Working?

| Event | Step | QPREF student pool size |
|-------|------|------------------------|
| Training begins | 0 | 0 (RM not active) |
| RM activated | ~25,000 | 0 → starts filling |
| Pool has 2 episodes | ~26,000 | 2 — QPREF can sample pairs |
| Pool has 500 episodes | ~525,000 | 500 — full recency window |
| Guard deque fills (50 entries) | ~25k + 50k | guard eligibility |

After RM activation at ~25k, roughly one episode completes per ~1000 env steps (HalfCheetah episode length ≈ 1000 steps). So the pool reaches 2 episodes within ~1–2k steps of activation and 500 within ~500k steps.

**QPREF firing schedule:** every `qpref_grad_interval=10` gradient steps during each `train()` call. With the default `gradient_steps` setting, QPREF fires ~100 times per `train()` call. The guard deque fills after 50 `train()` calls where QPREF was active (≈ 50k env steps after RM activation ≈ step 75k).

---

## 5. Pair Sampling and Ranking Signal

`sample_qpref_pairs_aggregate(batch_size=4, source='student')` samples 4 pairs:

1. Sample two random indices `a, b` from `pref_student_episodes` (with tie-break on self-match).
2. Assign positive = episode with higher J, negative = lower J.
3. Pad both to the same `T_max` within the batch, build binary mask.
4. Return `[B, T_max, obs_dim]` tensors for critic evaluation.

The critic evaluates `min(Q1, Q2)` at every valid timestep, then takes the masked mean per trajectory. The QPREF loss penalises pairs where `mean_Q_pos < mean_Q_neg`.

**Early signal (steps 25k–100k):** pool contains low-quality episodes from the start of training through RM activation. J range is small (all episodes are bad; RM gives low scores to everything). The loss provides weak ordering signal.

**Mid signal (steps 100k–400k):** policy is improving rapidly. Pool spans a large quality range (early bad episodes + recent improving episodes). J differences are large → clear positive/negative labels → strong ordering signal for the critic. This is the most useful phase.

**Late signal (steps 400k+):** pool fills with expert-quality episodes (recency-500 window). J differences shrink as all recent episodes are near-expert. This is the analog of the J_std collapse seen in Soft-TAC, but less severe because: (a) the student pool covers the last 500 episodes of a monotonically improving policy (more history than Soft-TAC's 200), and (b) QPREF works in Q-space not J-space — even small J differences are meaningful if the critic has internalized the ordering.

---

## 6. Guard Mechanism

The guard prevents QPREF from harming the critic once the ordering signal inverts.

**Guard state:** `_qpref_active` (bool, starts True), `_qpref_delta_deque` (ring buffer of 50 per-train-call mean deltas), `_qpref_peak_guard_mean` (highest rolling mean ever seen).

**Guard check (runs once per `train()` call, inside QPREF logging block):**

```
Three conditions must ALL hold before guard can fire:
  1. deque_full: 50 train() calls have accumulated (prevents early-instability triggers)
  2. peak_positive: _qpref_peak_guard_mean >= 1.0 (QPREF must have shown genuine
     positive phase — critic ranked teacher above student in Q-space)
  3. guard_mean < -0.3: recent-10-entry mean delta has turned persistently negative
     for 3 consecutive train() calls
```

When all three hold for `confirm=3` consecutive intervals:
- `_qpref_active = False` (permanent)
- `_qpref_skipped_updates` incremented on every subsequent would-have-fired step
- Print: `[QPREF GUARD] Triggered at step ...`

**Why condition 2 matters for the online setting:** the online student pool starts empty and fills slowly. Early deltas may be noisy or negative simply because the pool has too few diverse pairs. Condition 2 prevents the guard from firing during this warm-up by requiring evidence that QPREF was genuinely useful before disabling it.

---

## 7. Logged Metrics

| Metric | Meaning |
|--------|---------|
| `train/qpref_source_student_pool_size` | Current `len(pref_student_episodes)` — confirms pool is growing |
| `train/qpref_pairs_available` | N gradient steps this train() call where pairs were found |
| `train/qpref_loss` | Mean QPREF loss this train() call |
| `train/qpref_mean_q_pos` | Mean Q of positive (higher-J) trajectories |
| `train/qpref_mean_q_neg` | Mean Q of negative (lower-J) trajectories |
| `train/qpref_delta` | mean(Q_pos − Q_neg) — positive = critic ranks correctly |
| `train/qpref_delta_guard_mean` | Mean of last 10 train-call deltas — the guard decision signal |
| `train/qpref_peak_guard_mean` | Highest guard_mean ever seen — must hit 1.0 before guard eligible |
| `train/qpref_active` | 1 = QPREF firing; 0 = guard triggered |
| `train/qpref_guard_triggered` | 1 after guard fires (latches permanently) |
| `train/qpref_skipped_updates` | Cumulative steps skipped post-guard |
| `[TeacherBuffer] qpref student pool: first episode added J=...` | Confirms RM activation and first pool entry |

**Key diagnostic:** watch `qpref_source_student_pool_size`. It should be 0 until ~25k (RM activation), then grow at ~1/episode. If it stays 0 past 50k, the RM is not activating or `online_rm_manager.is_active` is not being set.

---

## 8. Difference from Offline QPREF

| Aspect | Offline (sail_sb3) | Online (sail_sb3_online) |
|--------|-------------------|--------------------------|
| J label source | Frozen offline pref_rm at insertion | Online RM at insertion; no GT fallback |
| Pool fills from | Episode 0 (pref_rm always loaded) | Episode after RM activation (~25k) |
| J improves over run | No (pref_rm is frozen) | No (J frozen at insertion; RM improves but old entries keep old J) |
| Retroactive rescore | No | No (intentional — recency window discards stale entries naturally) |
| Guard mechanism | Yes (added 2026-04-18) | Yes (same implementation) |
| need_pref_rm flag | True when --qpref | False when --qpref + --online_rm |

---

## 9. Files Changed (2026-04-18 implementation)

| File | Change |
|------|--------|
| `sail_sb3_online/datasets/teacher_buffer.py` | `add_student_episode` accepts optional `J: float = None`; uses pref_rm only if J not provided; logs first episode |
| `sail_sb3_online/utils/callbacks.py` | Student QPREF branch: computes RM J via `online_rm_manager.reward()`, passes to `add_student_episode`; no GT fallback |
| `sail_sb3_online/scripts/train_sail.py` | `_qpref_needs_pref_rm = args.qpref and not _is_online_rm` (online QPREF no longer requires offline pref_rm) |
| `sail_sb3_online/algorithms/sail.py` | Guard mechanism added: 6 state vars, `qpref_would_run` split, full guard check in logging block |
| `sail_sb3_online/HC_OnlineRM_QPREF_Student.sbatch` | Updated echo header to document RM-only J; no --pref_rm flag needed |

---

## 10. sbatch Command

```bash
sbatch sail_sb3_online/HC_OnlineRM_QPREF_Student.sbatch
```

Key flags:
```
--qpref
--qpref_source student
--qpref_weight 0.1
--qpref_temp 1.0
--qpref_batch_size 4
--qpref_grad_interval 10
--qpref_guard_threshold -0.3
--qpref_guard_confirm 3
--qpref_guard_window 10
--qpref_guard_positive_threshold 1.0
--pref_max_student_trajs 500
--online_rm  [+ full RM config]
```

No `--pref_rm` needed. The online RM is the sole source of J labels for the student pool.
