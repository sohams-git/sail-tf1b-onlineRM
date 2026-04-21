# `--pref_reweight_teacher` Implementation Notes

**Date:** 2026-04-09  
**Status:** Implemented and sanity-checked. No jobs submitted yet.

---

## What Was Implemented

Boltzmann-weighted discriminator expert batches (PR-SAIL), faithful to the TF reference implementation in `stable-baselines/stable_baselines/td3/sail.py`.

**Core mechanism:**  
Each expert episode in the pref pool has a Boltzmann weight `w(τ) = softmax(J_phi(τ) / beta)` where `J_phi(τ) = sum_t R_phi(s_t, a_t)` is the cumulative offline RM score. At each discriminator update, expert transitions are drawn **uniformly by episode** (TF parity — NOT proportionally to w), then the episode's softmax weight is attached as a per-sample multiplier to the expert BCE loss, computed as `sum(w_i * L_i) / (sum(w_i) + 1e-8)`. This biases the discriminator gradient toward higher RM-quality episodes without changing sampling frequency.

---

## Files Changed

### 1. `sail_sb3/datasets/teacher_buffer.py`

**New `__init__` params:**
- `pref_max_teacher_trajs: int = None` — max episodes in pref pool before quantile pruning
- `pref_promote_quantile: float = 0.75` — quantile threshold for pruning low-J episodes

**New fields stored in `__init__`:**
- `self.pref_max_teacher_trajs`, `self.pref_promote_quantile` — stored from args
- `self.pref_teacher_weights = None` — np.array[N_eps], softmax(J/beta), sums to ~1
- `self._pref_reweight_beta = 1.0` — temperature; overwritten by `train_sail.py` after construction

**New method: `_recompute_pref_weights(beta=None)`**  
Computes softmax weights over `pref_episodes` using `_pref_reweight_beta` (or explicit `beta`).  
TF parity: max-subtraction before `exp()`, denominator has `1e-8`.  
Called at: end of `_build_pref_episodes()` (initial expert build); end of `add_episode()` (after each promotion).

**New method: `sample_batch_weighted(batch_size)`**  
Returns `(states, actions, expert_w)` tensors of shape `(B, obs_dim)`, `(B, act_dim)`, `(B, 1)`.  
TF parity (sail.py:780): episodes sampled uniformly (`np.random.randint(0, N)`, NOT proportional), per-sample weight = `pref_teacher_weights[ep_idx]`.  
Falls back to `sample_batch()` + `ones` weights when pref pool is empty or weights not computed.

**Updated `add_episode()`:**  
After appending a promoted student episode to `pref_episodes`, now:
1. Applies quantile-based pruning if `pref_max_teacher_trajs` is set and pool exceeds it (TF parity: keep J >= quantile(scores, q), always keep best).
2. Calls `_recompute_pref_weights()` to refresh weights immediately.

**`_build_pref_episodes()`:** Added `self._recompute_pref_weights()` call at the end so weights are ready immediately after expert pool initialization.

---

### 2. `sail_sb3/reward_models/adversary.py`

**New `__init__` param:**
- `use_expert_weights: bool = False` — when True, expert BCE loss is computed as weighted mean

**Updated `compute_loss()` signature:**
- Added `expert_w: torch.Tensor = None` parameter (shape `(N, 1)`)

**Updated expert loss computation:**
```python
sample_expert_loss = F.binary_cross_entropy_with_logits(
    expert_logits, torch.ones_like(expert_logits), reduction='none')  # (N, 1)
if self.use_expert_weights and expert_w is not None:
    expert_loss = (sample_expert_loss * expert_w).sum() / (expert_w.sum() + 1e-8)
else:
    expert_loss = sample_expert_loss.mean()
```

TF parity: `adversary.py:749-756` — weighted sum normalized by sum of weights.  
When `use_expert_weights=False` (flag off), `expert_w` is accepted but silently ignored — fallback is identical to pre-implementation behavior.

---

### 3. `sail_sb3/algorithms/sail.py`

**New `__init__` param:**
- `pref_reweight_teacher: bool = False`

**New field:**
- `self.pref_reweight_teacher = pref_reweight_teacher`

**New pending-metric fields for logging:**
- `self._pending_reweight_pool_size: list`
- `self._pending_reweight_w_max: list`

**Updated `_update_discriminator()`:**  
Expert batch construction now branches on `pref_reweight_teacher`:
```python
use_weighted = (self.pref_reweight_teacher
                and len(self.teacher_buffer.pref_episodes) >= 2)
if use_weighted:
    expert_states, expert_actions, expert_w = \
        self.teacher_buffer.sample_batch_weighted(self.disc_batch_size)
else:
    expert_batch = self.teacher_buffer.sample_batch(self.disc_batch_size)
    ...
    expert_w = torch.ones(...)
```
`expert_w` is always passed to `compute_loss(expert_w=expert_w)`.

Logging: when `use_weighted`, accumulates `pref_reweight/pool_size` and `pref_reweight/weight_max` per disc gradient step.

**Updated `train()`:**  
Drains `_pending_reweight_pool_size` / `_pending_reweight_w_max` and logs them via `self.logger.record`.

---

### 4. `sail_sb3/scripts/train_sail.py`

**New CLI args (4 total):**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pref_reweight_teacher` | flag | False | Enable PR-SAIL weighted disc expert batch |
| `--pref_beta` | float | 1.0 | Boltzmann temperature |
| `--pref_max_teacher_trajs` | int | 500 | Max pref pool size before pruning |
| `--pref_promote_quantile` | float | 0.75 | Quantile for pruning low-J episodes |

**Updated `need_pref_rm`:**  
Added `or args.pref_reweight_teacher` — ensures RM is loaded when reweighting is on.

**Updated `TeacherBuffer` construction:**  
Passes `pref_max_teacher_trajs` and `pref_promote_quantile` to constructor (gated: only pass `pref_max_teacher_trajs` when flag is on).

**Post-construction beta setup:**
```python
if args.pref_reweight_teacher:
    teacher_buffer._pref_reweight_beta = args.pref_beta
    teacher_buffer._recompute_pref_weights(beta=args.pref_beta)
    # prints: weights min/max/sum
```

**Updated `Adversary` construction:**  
`use_expert_weights=args.pref_reweight_teacher` — coupled to the same flag (no separate footgun flag).

**Updated `SAIL` construction:**  
`pref_reweight_teacher=args.pref_reweight_teacher`

---

## Sanity Checks Passed

All 9 checks passed (run locally, not on cluster):

| # | Check | Result |
|---|-------|--------|
| 1 | `_recompute_pref_weights`: sum≈1, monotone ordering | PASS |
| 2 | `sample_batch_weighted`: correct shapes, weights from pref_teacher_weights | PASS |
| 3a | Ones-weight expert loss = unweighted expert loss | PASS |
| 3b | Unequal weights produce different expert loss | PASS |
| 3c | `use_expert_weights=False` ignores `expert_w` (fallback intact) | PASS |
| 4 | `pref_reweight_teacher` in `SAIL.__init__` signature | PASS |
| 5 | All 4 new CLI flags in `--help` output | PASS |
| 6 | Fallback returns `ones` weights when pref pool empty | PASS |
| 7 | Pool pruned correctly (size ≤ pref_max_teacher_trajs) after 14 episodes, max=10 | PASS |
| 8 | `add_episode` has no GT params (no leakage) | PASS |
| 9 | `sample_pref_pairs` (pref_rank_disc) and `sample_batch_weighted` (pref_reweight) both work on same `pref_episodes` pool | PASS |

---

## Deviations from Original Plan

**None.** All 7 steps from `pref_reweight_teacher_pytorch_plan.md` implemented as specified.

Minor decisions made during implementation:

1. **Beta recompute on construction:** `train_sail.py` calls `_recompute_pref_weights(beta=args.pref_beta)` after setting `_pref_reweight_beta`. The initial call in `_build_pref_episodes()` used `beta=1.0` (default). This means weights are recomputed once with the correct user-supplied beta at startup. Cost: one extra numpy pass over N_expert episodes (negligible).

2. **`_pending_reweight_pool_size` accumulates per disc gradient step, logged as mean:** The pool size doesn't change within a single `train()` call so all values will be identical. The `mean()` is harmless and consistent with how `_pending_disc_losses` is drained.

3. **`sample_batch_weighted` fallback threshold is `>= 2` episodes in SAIL but returns fallback when pool is empty in TeacherBuffer.** `SAIL._update_discriminator` gates `use_weighted` on `len(pref_episodes) >= 2` (need at least 2 for meaningful weights; also matches `sample_pref_pairs` requirement). `TeacherBuffer.sample_batch_weighted` independently falls back when pool is empty. Both guards are present.

---

## Ambiguities Resolved

1. **Uniform sampling (NOT proportional) confirmed as TF behavior.** TF line 804: `ep_idx = np.random.randint(0, n_eps)` with comment "instead of np.random.choice(..., p=weights)". This is intentional. Implemented faithfully.

2. **`disc_use_expert_weights` coupled to `pref_reweight_teacher`** — in TF these are separate flags. In PyTorch we couple them: `use_expert_weights=args.pref_reweight_teacher`. This prevents the footgun where weights are computed and sampled but discarded in the loss.

3. **`pref_teacher_weights` type: numpy array, not tensor** — consistent with TF (scores/weights are Python-side, only the final scalar per sample is handed to PyTorch as `expert_w` tensor). `sample_batch_weighted` converts to tensor in the return.

---

## Logging to Expect When Active

At startup (when `--pref_reweight_teacher` and `--pref_rm` set):
```
[TeacherBuffer] Pref pool: 4 expert episodes (built from full dataset before ring truncation)
[TeacherBuffer] Pref pool RM scores: mean=1353.5 min=1333.1 max=1374.9 spread=41.8
[TeacherBuffer] Ring buffer: truncated transition buffer to last 1000 transitions ...
[train_sail] PrefReweight: beta=1.0  pool=4 eps  weights min=X max=Y sum=1.0000
```

During training (every ~1000 env steps, logged by SB3):
```
| pref_reweight/pool_size  | 4        |
| pref_reweight/weight_max | 0.9997   |
```

`weight_max` will be near 1.0 early (4 expert episodes with small spread → almost all weight on best). It should decrease as diverse student promotions add episodes with different J values.

---

## What NOT to Expect

- No change to `pref_loss` (pref_rank_disc path is unchanged)
- No change to `disc_loss` scaling (weighted mean preserves comparable magnitude)
- No change to LfD mixing (critic batch path unchanged)
- No change to transition ring buffer behavior

---

## How to Enable in a New sbatch

Add these flags to an existing AdaptPref sbatch:
```bash
--pref_reweight_teacher \
--pref_beta 1.0 \
--pref_max_teacher_trajs 500 \
```

`--pref_rm` must also be set (already required by existing AdaptPref sbatches).

The `--pref_promote_quantile 0.75` default is fine for most runs.
