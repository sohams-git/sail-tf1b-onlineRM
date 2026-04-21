# Separate Preference Episode Pool — Implementation Notes

**Date:** 2026-04-06  
**Jobs:** 46847221 (TFParity_AdaptPref_SepPool), 46847222 (RMAdaptPref_SepPool)  
**Seeds:** 1–2 each, all running

---

## Problem Being Solved

With `teacher_buffer_size=1000` (TF ring buffer parity), loading 4 expert episodes (4000 transitions) into a 1000-slot ring leaves only the last episode (episode 3) in the transition buffer. The original code called `_build_pref_episodes()` **after** ring truncation, so `pref_episodes` contained exactly 1 episode after init. This caused:

- `sample_pref_pairs()` to raise `ValueError("Not enough expert episodes for preference ranking")` until the first student promotion (~290–460k steps)
- pref_loss completely inactive for the first ~30–46% of training
- RM threshold for `score_source=rm` calibrated on only 1 episode (bimodal failure risk)

---

## Code Changes

### `sail_sb3/datasets/teacher_buffer.py`

**What changed:** Moved RM loading and `_build_pref_episodes()` call to **before** the ring truncation block.

**Old order in `__init__`:**
1. Load all 4000 transitions
2. Apply ring truncation (→ 1000 transitions)
3. Set `self.initial_size`, `self._has_promotions = False`
4. Load RM and build pref pool (only 1 episode visible after truncation)

**New order:**
1. Load all 4000 transitions
2. Load RM and build pref pool from full data (all 4 episodes captured)
3. Apply ring truncation (→ 1000 transitions, pref_episodes unaffected)
4. Set `self.initial_size`, `self._has_promotions = False`

**Other fix:** Added `.ravel()` in `_build_pref_episodes()` to flatten the `(N, 1)` dones tensor before `np.where()`:
```python
dones_np = self.dones.cpu().numpy().ravel()
```

**`add_episode()` unchanged** — already appends to `pref_episodes` without ring capping.

### `sail_sb3/utils/callbacks.py`

Added `pref_pool_size` metric logging on each adaptive promotion:
```python
self.logger.record("adaptive/pref_pool_size",
                   len(self.teacher_buffer.pref_episodes))
```

---

## New Jobs

| Job ID | Name | WANDB Group | Seeds | Submitted |
|--------|------|-------------|-------|-----------|
| 46847221 | HC_TFP_AP_SP | HalfCheetah-v2_tfparity_adaptpref_seppool | 1–2 | 2026-04-06 18:09 EDT |
| 46847222 | HC_RMAP_SP | HalfCheetah-v2_rmadaptpref_seppool | 1–2 | 2026-04-06 18:09 EDT |

**Key difference vs prior runs (46817057, 46818417):** pref pool initialized from all 4 expert episodes instead of 1.

---

## Early Verification (from logs at ~12k–36k steps)

### Startup messages — all 4 seeds confirmed:
```
[TeacherBuffer] Pref pool: 4 expert episodes (built from full dataset before ring truncation)
[TeacherBuffer] Pref pool RM scores: mean=1353.5 min=1333.1 max=1374.9 spread=41.8
[TeacherBuffer] Ring buffer: truncated transition buffer to last 1000 transitions
                             (pref pool retains all 4 expert episodes)
```

### TFParity_AdaptPref_SepPool seed1 (job 46847221_1):
```
[train_sail] Expert episodes: 1 | Total transitions: 1000
[train_sail] Adaptive mode: expert score threshold initialized to 6988.3
```
- pref_loss at 12k: **0.246** → 13k: 0.00611 → 34k: 1.13e-4
- disc_loss at 36k: 0.325 (healthy, non-saturated)
- surrogate_reward_mean at 36k: 1.12 (normal range)

### TFParity_AdaptPref_SepPool seed2 (job 46847221_2):
- pref_loss at 12k: **0.0308** → 13k: 1.07e-4

### RMAdaptPref_SepPool seed1 (job 46847222_1):
```
[train_sail] RM adaptive: rm_expert_scores[0] (threshold) = 1366.5
[train_sail] RM adaptive: rm_expert_scores = ['1366.5', '1339.3', '1333.1', '1374.9']
```
- pref_loss at 12k: **0.322** → 13k: 0.0278 → 30k: 3.99e-5

### RMAdaptPref_SepPool seed2 (job 46847222_2):
- pref_loss at 12k: **0.664** → 13k: 0.00795 → 14k: 7.05e-4

### Summary checklist

| Check | Result |
|-------|--------|
| Main transition buffer capped at 1000 | ✓ confirmed from startup |
| Pref pool starts with 4 expert episodes (not 1) | ✓ all 4 seeds |
| pref_loss appears before first promotion | ✓ at 12k steps (vs 290–460k before) |
| No ValueError / sampling / shape crashes | ✓ clean logs |
| GT threshold TF-parity (6988.3) | ✓ TFParity seeds |
| RM threshold uses all 4 episodes (not just 1) | ✓ rm_expert_scores = [1366.5, 1339.3, 1333.1, 1374.9] |
| RM threshold = first-loaded episode (1366.5) | ✓ insertion-order parity preserved |

---

## pref_loss Behavior: Before vs After

| Phase | Before SepPool | After SepPool |
|-------|---------------|---------------|
| 0–290k (pre-promotion) | **N/A** — ValueError, pref_loss not logged | **Active from 12k** — 0.03–0.66 at first disc update |
| 10k–30k | Completely inactive | 1e-4 to 1e-3 (4 similar-quality episodes → easy ranking) |
| 300k+ after promotions | 0.1–0.5 (only then active) | Expected to grow similarly as diverse student promos added |

**Interpretation:** The 4 expert episodes have RM spread=41.8 points — relatively small, so after the initial burst at 12k, pref_loss quickly settles near 0 (ranking task solved early). This is correct behavior: pref_loss will grow again once student promotions add quality-diverse episodes to the pool (same Phase 2/3 pattern as before, but the pool now has 4 starting anchors instead of 1, giving better coverage of the quality space).

---

## Open Issues (Not Fixed Here)

1. **`adaptive/promoted_episodes` always logs 0** — `get_growth_stats()` returns `added_transitions = current_size - initial_size = 1000 - 1000 = 0` with ring buffer. Fix: add `self._total_promotions` counter. Not yet implemented.
2. **Final performance comparison** — deferred pending run completion.
