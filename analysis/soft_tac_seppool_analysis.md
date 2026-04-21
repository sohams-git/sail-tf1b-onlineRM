# Soft-TAC Separate Pool: Design, Bugs, Fixes, and Analysis

**Date:** 2026-04-17  
**Codebases:** `sail_sb3/` (offline RM) and `sail_sb3_online/` (online RM)  
**WANDB groups:** `HalfCheetah-v2_softtac_seppool_v2` / `HalfCheetah-v2_online_rm_softtac_seppool_v2`

---

## 1. Why the Original Soft-TAC Was Broken

### The Pool Homogeneity Problem

The original Soft-TAC implementation used `pref_episodes` as the pool for pair sampling. `pref_episodes` is built at init from all expert episodes, scored by the offline RM, and grows only when a student episode is promoted (quality-gated).

For HalfCheetah with 4 expert episodes and `teacher_buffer_size=1000`:

- Ring buffer truncation leaves only 1 expert episode in the transition buffer
- `_build_pref_episodes()` runs on the **full pre-truncation dataset**, so `pref_episodes` retains all 4 expert episodes
- All 4 experts have RM scores: mean=1353.5, min=1333.1, max=1374.9, **spread=41.8**
- Relative spread = 41.8 / 1353.5 = **3.1%**

With near-identical J values across all pool members:

```
diff = J_pos - J_neg  ≈  41.8 at most (usually much less)
tanh(diff / T) = tanh(41.8 / 1.0) ≈ 1.0   → always saturated
loss = 1 - mean(y * tanh(delta_disc / T))
```

When all pairs are near-tied in J-space, `y` labels are almost all `0` (filtered by `tac_tie_eps`), or all `+1` with very similar confidence. The Soft-TAC loss produces near-zero gradient throughout training. The discriminator receives no useful alignment signal from this path.

**Confirmed by log evidence:** Both broken jobs (48179500 offline, 48179317 online) showed `soft_tac/pool_size=0` at startup — the init-order bug meant the pool never populated at all.

---

## 2. The Fix: Dedicated `soft_tac_pool`

### Design

Replace the `pref_episodes`-based pool with a dedicated `soft_tac_pool` that is:

- **Expert component** (`_soft_tac_expert_episodes`): permanent, all 4 expert episodes from the full pre-truncation dataset, J = RM score (offline path) or GT return (online path). Set once at init, never changed.
- **Student component** (`soft_tac_student_episodes`): every completed student episode is added unconditionally (no quality filter). Recency-pruned to `soft_tac_max_student_trajs=200`.

```python
@property
def soft_tac_pool(self):
    return self._soft_tac_expert_episodes + self.soft_tac_student_episodes
```

**Diversity timeline:**
- Step ~0: pool = 4 expert episodes (spread = 41.8 RM units or ~247 GT return units)
- Step ~1k (first episode ≈ 1000 steps): first low-quality student added (J ≈ -600 GT)
- Step ~5k: pool has 5 episodes spanning expert quality down to random policy
- Step ~200k: pool has 200 students + 4 experts; full J range [-600, 7000+]

This gives the discriminator pairs with large, clear J differences from step ~1k onwards.

---

## 3. Bug 1 — Init-Order Overwrite (Critical)

### Root Cause

In `teacher_buffer.py __init__`, the call order was:

```python
# ---- BEFORE fix ----
if pref_rm_path:
    self._build_pref_episodes()          # sets self._soft_tac_expert_episodes = list(pref_episodes)
    ...

# Later in __init__ (these lines ran AFTER _build_pref_episodes):
self._soft_tac_expert_episodes = []      # ← OVERWRITES the assignment above
self.soft_tac_student_episodes = []
```

Result: `soft_tac_pool` was always empty at startup regardless of whether `_build_pref_episodes()` ran correctly.

**Log evidence:** `soft_tac/pool_size=0` at every step in both broken jobs.

### Fix (both codebases)

Move initialization of `_soft_tac_expert_episodes` and `soft_tac_student_episodes` to **before** the `if pref_rm_path:` block:

```python
# ---- AFTER fix ----
# Must be initialized BEFORE _build_pref_episodes() so the assignment inside
# that method is not overwritten by later __init__ code.
self._soft_tac_expert_episodes = []
self.soft_tac_student_episodes = []

if pref_rm_path:
    self._build_pref_episodes()          # now correctly persists
```

And remove the duplicate initializations that were after the block.

**Verified by unit test:** `_soft_tac_expert_episodes: 4` at init for both offline and online paths.

---

## 4. Bug 2 — Student Pool Never Grows in Online Path

### Root Cause

In the online RM path there is no offline `pref_rm` object. The original `add_soft_tac_student_episode` required `pref_rm` to score student episodes:

```python
def add_soft_tac_student_episode(self, obs_ep, acs_ep):
    if self.pref_rm is None:
        return           # ← always returns in online path
    J = sum(pref_rm.reward(obs_ep, acs_ep))
    ...
```

Result: `soft_tac_student_episodes` stayed empty for the entire 1M-step run. Pool remained 4 expert-only episodes — diverse in GT space (spread=247) but too small for useful pair sampling at scale.

### Fix

Added optional `J: float = None` parameter to `add_soft_tac_student_episode` in both codebases:

```python
def add_soft_tac_student_episode(self, obs_ep, acs_ep, J: float = None):
    if J is None:
        if self.pref_rm is None:
            return           # no-op only when BOTH J and pref_rm are absent
        J = sum(pref_rm.reward(obs_ep, acs_ep))
    # append with provided or computed J
    self.soft_tac_student_episodes.append({'obs': ..., 'acs': ..., 'J': J})
```

Updated both callbacks to pass `J=gt_score`:

```python
# sail_sb3/utils/callbacks.py and sail_sb3_online/utils/callbacks.py
if self.soft_tac and obs_ep is not None and len(obs_ep) > 0:
    self.teacher_buffer.add_soft_tac_student_episode(obs_ep, acs_ep, J=gt_score)
```

**Verified by unit test:**
- Online GT path: `_soft_tac_expert_episodes: 4`, pool grows with each student call, J spread = 7338.3 (expert GT=6988 to student GT=-350).

---

## 5. J Signal Design Decision: Always GT

### What J Does in Soft-TAC

From `sail.py` and `adversary.py`:

```python
diff = (tpos_J - tneg_J).detach().cpu().numpy()    # stored J difference
y[diff >  tac_tie_eps] =  1.0                        # pair label
y[diff < -tac_tie_eps] = -1.0

# In compute_soft_tac_loss:
delta = J_disc_pos - J_disc_neg       # discriminator-derived, NOT stored J
tac_term = y * tanh(delta / T)
loss = 1 - mean(tac_term)
```

**Stored J is only used for:**
1. Determining which episode is "positive" in `sample_soft_tac_pairs`
2. Tie detection: `|J_pos - J_neg| < tac_tie_eps` → `y=0` → zero gradient

The `tanh` argument is `delta = disc_J_pos - disc_J_neg` — entirely discriminator-derived. J does not enter the tanh.

### Why GT Wins Over RM

| Criterion | GT return | Online RM (at activation, acc=0.60) |
|-----------|-----------|--------------------------------------|
| Ordering accuracy | 100% | ~60% (40% wrong labels) |
| Scale consistency across pool | Always (all GT return units) | Breaks (old episodes have GT, new have RM) |
| RM compute per episode | 0 | 1 full forward pass |
| Rescoring needed | No | Yes, every `rescore_freq` |
| Stability early training | Full | Degraded (wrong labels hurt disc) |

**Scale mismatch is the fatal problem for RM-based J:** Expert episodes have `J = GT return ≈ 6860`. If students get `J = RM_score ≈ 300-1374`, comparing them is meaningless. `sign(6860 - 500) = +1` is spurious — the expert just happens to be in GT units while the student is in RM units. The ordering is random.

**Wrong labels are the other fatal problem:** At acc=0.60, 40% of pairs get reversed `y` labels. The Soft-TAC loss actively pushes the discriminator in the wrong direction for those pairs.

**Blending is worse than either:** `J = (1-β)*gt + β*rm` requires normalization first (different scales). Running z-score normalization over a 200-episode recency window is unstable. The blend adds noise for no benefit over pure GT.

**Future-proofing:** The `J` parameter design supports the reward-free case (no GT access) by allowing any caller to provide J. For HalfCheetah with full env access, GT is always the right choice.

**Current implementation is correct.** Callbacks pass `J=gt_score`. No further changes needed.

---

## 6. Isolation Analysis: Does Soft-TAC Break Original SAIL Augmentation?

### Two Independent Parallel Systems

#### System A — Original SAIL Adaptive Augmentation

**Trigger:** `student_score > threshold_list[0]` (quality-gated)

**Code path in `_on_step`:**
```
student_score > threshold
  → teacher_buffer.add_episode(obs_ep, acs_ep)
      writes: self.states / actions / next_states / dones (transition ring buffer)
              self._has_promotions = True
  → teacher_buffer.add_pref_episode(obs_ep, acs_ep, J_pref)   [online path]
      writes: self.pref_episodes
              self.pref_teacher_weights (recomputed)
  → threshold FIFO updated
```

**Effect on sail.py training:**
- `lfd_active = lfd_mixing and not teacher_buffer._has_promotions` → LfD turns off on first promotion
- `sample_batch()` / `sample_batch_weighted()` → samples from transition ring buffer
- `sample_pref_pairs()` → samples from `pref_episodes` for PrefRank loss
- `rescore_pref_episodes()` → updates J in `pref_episodes` in-place

#### System B — Soft-TAC Pool

**Trigger:** Every episode end, unconditional (no quality filter)

**Code path in `_on_step`:**
```
if soft_tac and obs_ep is not None:
  → teacher_buffer.add_soft_tac_student_episode(obs_ep, acs_ep, J=gt_score)
      writes: self.soft_tac_student_episodes (and nothing else)
```

**Effect on sail.py training:**
- `sample_soft_tac_pairs()` → samples from `soft_tac_pool = _soft_tac_expert_episodes + soft_tac_student_episodes`
- Contributes `soft_tac_weight * tac_loss` to `total_disc_loss`

### Complete State Isolation Table

| Attribute | System A `add_episode` | System A `add_pref_episode` | System B `add_soft_tac_student_episode` |
|-----------|----------------------|----------------------------|----------------------------------------|
| `self.states / actions / next_states / dones` | **Writes** | No | No |
| `self._has_promotions` | **Sets True** | No | **Never touched** |
| `self.pref_episodes` | No (online; offline RM path only) | **Writes** | No |
| `self.pref_teacher_weights` | No | **Writes** | No |
| `self.soft_tac_student_episodes` | No | No | **Writes** |
| `self._soft_tac_expert_episodes` | No | No | No (set once at init) |

### Answers to Each Concern

**Q: Does adding to `soft_tac_student_episodes` affect the adaptive augmentation path?**  
No. `add_soft_tac_student_episode()` writes only to `soft_tac_student_episodes`. It never touches `_has_promotions`, the transition ring buffer, or `pref_episodes`.

**Q: Does the original SAIL augmentation still happen exactly as before?**  
Yes. The promotion check runs independently. `add_episode()` still fires on promotion. `_has_promotions` still gates LfD mixing. The transition ring buffer still grows. `pref_episodes` still grows via `add_pref_episode()`.

**Q: Can Soft-TAC cause the model to stop using normally promoted student trajectories?**  
No. Promoted transitions flow into `self.states` via `add_episode()`. `sample_batch()` reads from `self.states`. Neither path is aware of `soft_tac_pool`. Even if `soft_tac_pool` grew to 10,000 episodes, it would not affect `sample_batch()` or `sample_pref_pairs()`.

**Q: Does a promoted episode appear in both systems?**  
Yes. When a student is promoted, the callback calls `add_episode()` (System A) and then `add_soft_tac_student_episode()` (System B) for the same episode. The promoted episode's transitions go into the ring buffer; its obs/acs tensors with GT J go into `soft_tac_student_episodes`. These are separate consumers; there is no double-counting or conflict.

**Verdict:** `soft_tac_pool` is a pure auxiliary structure. Zero shared state with the original augmentation path. The original SAIL adaptive augmentation is completely preserved.

---

## 7. Full-Run sbatch Commands

### Cancel Broken Jobs First

```bash
scancel 48179500 48179317
```

### `sail_sb3` (Offline RM + Soft-TAC SepPool)

**File:** `sail_sb3/HC_TFParity_SoftTAC_SepPool.sbatch`

```
sbatch sail_sb3/HC_TFParity_SoftTAC_SepPool.sbatch
```

**Key parameters:**
- `--array=1-2` (seeds 1 and 2)
- `--time=12:00:00`
- `--adaptive --lfd_mixing --teacher_buffer_size 1000`
- `--adaptive_score_source rm --pref_rm <offline_RM_path>`
- `--soft_tac --soft_tac_weight 0.5 --soft_tac_temp 1.0 --tac_tie_eps 0.0`
- `--soft_tac_max_student_trajs 200`
- `--entcoeff 0.05`
- WANDB group: `HalfCheetah-v2_softtac_seppool_v2`
- **No** `--pref_rank_disc`

### `sail_sb3_online` (Online RM + Soft-TAC SepPool)

**File:** `sail_sb3_online/HC_OnlineRM_SoftTAC_SepPool.sbatch`

```
sbatch sail_sb3_online/HC_OnlineRM_SoftTAC_SepPool.sbatch
```

**Key parameters:**
- `--array=1-3` (seeds 1, 2, 3)
- `--time=8:00:00`
- `--adaptive --lfd_mixing --teacher_buffer_size 1000 --pref_max_teacher_trajs 500`
- `--soft_tac --soft_tac_weight 0.5 --soft_tac_temp 1.0 --tac_tie_eps 0.1`
- `--soft_tac_max_student_trajs 200`
- `--online_rm` with gates: `--rm_min_segments 500 --rm_min_updates 50 --rm_min_acc 0.60`
- `--rm_rescore_freq 20000 --rm_segment_len 50`
- `--entcoeff 0.05`
- WANDB group: `HalfCheetah-v2_online_rm_softtac_seppool_v2`
- **No** `--pref_rank_disc`
- J signal for Soft-TAC: `gt_score` throughout (passed by callback)

---

## 8. Summary of All Code Changes

| File | Change | Bug Fixed |
|------|--------|-----------|
| `sail_sb3/datasets/teacher_buffer.py` | Moved `_soft_tac_expert_episodes = []` and `soft_tac_student_episodes = []` before `if pref_rm_path:` block | Init-order overwrite (Bug 1) |
| `sail_sb3/datasets/teacher_buffer.py` | Added `J: float = None` to `add_soft_tac_student_episode` | Enables GT J path |
| `sail_sb3_online/datasets/teacher_buffer.py` | Same init-order fix | Init-order overwrite (Bug 1) |
| `sail_sb3_online/datasets/teacher_buffer.py` | Same `J: float = None` fix | Student pool never grew (Bug 2) |
| `sail_sb3/utils/callbacks.py` | Pass `J=gt_score` to `add_soft_tac_student_episode` | Bug 2 (offline) |
| `sail_sb3_online/utils/callbacks.py` | Pass `J=gt_score` to `add_soft_tac_student_episode` | Bug 2 (online) |
| `sail_sb3/algorithms/sail.py` | Decoupled PrefRank and Soft-TAC into two independent sampling blocks; added `soft_tac/pool_size` and `soft_tac/student_pool_size` logging | Architecture |
| `sail_sb3_online/algorithms/sail.py` | Same decoupling | Architecture |
| `sail_sb3/scripts/train_sail.py` | Added `--soft_tac_max_student_trajs` CLI arg; plumbed into TeacherBuffer and SAILAdaptiveCallback | Wiring |
| `sail_sb3_online/scripts/train_sail.py` | Same | Wiring |

---

## 9. Expected Behavior After Fix

### Offline (`sail_sb3`) — Soft-TAC SepPool v2

- At startup: `soft_tac/pool_size=4`, `soft_tac/student_pool_size=0`
- After step ~1k: first student episode added, `pool_size=5`
- Pool J spread: starts at 41.8 RM units (4 experts only), grows to thousands as student quality varies
- Soft-TAC loss: initially `≈1.0` (4 experts nearly tied), drops as diverse student episodes enter pool
- `tac_tie_eps=0.0` — all pairs with any J difference produce gradient

### Online (`sail_sb3_online`) — Online RM + Soft-TAC SepPool v2

- At startup: `soft_tac/pool_size=4`, `soft_tac/student_pool_size=0`
- J for expert episodes: GT return (mean=6860, spread=247)
- J for students: `gt_score` from Monitor callback — always in GT return space, always consistent
- After step ~1k: first student added, pool J spread immediately includes low-quality returns
- `tac_tie_eps=0.1` — filters pairs with nearly-identical GT returns
- Online RM activation has zero effect on Soft-TAC J (GT used throughout)
- Original adaptive augmentation path (promotion → transition ring buffer + pref_episodes) runs completely independently
