# TF Parity Buffer Plan — Demo Buffer + RM Adaptive Promotion

**Date:** 2026-04-06  
**Status:** Parity audit complete. Ready to implement.  
**Scope:** Step 1 audit + implementation plan for Steps 2–5

---

## Part A — Parity Audit

### A.1 Teacher / Demo Buffer

#### TF behavior (confirmed from code + logs)

`ReplayBufferExtend` is a plain ring/FIFO buffer:

```python
# deepq/replay_buffer.py — ReplayBufferExtend.add()
if self._next_idx >= len(self._storage):
    self._storage.append(data)      # fill up first
else:
    self._storage[self._next_idx] = data   # overwrite once full
self._next_idx = (self._next_idx + 1) % self._maxsize
```

For HalfCheetah, confirmed from `settings.py` and both TF log files:
```
'demo_buffer_size': 1000  (int(1e3))
'n_episodes': 4
```

Loading 4000 expert transitions (4 × 1000 steps) into a 1000-slot ring:
- idx 0–999: fills buffer, `_next_idx` wraps to 0
- idx 1000–1999: overwrites slots 0–999, `_next_idx` wraps to 0
- idx 2000–2999: overwrites all again
- idx 3000–3999: overwrites all again

**After loading: ring contains ONLY the last 1000 transitions = episode 3 (return 6988.28).**

This is confirmed by the TF log pattern and the fact that `_next_idx` = 0 after every full sweep.

#### PyTorch current behavior

`TeacherBuffer` uses unbounded append-only tensors:
```python
self.states = torch.cat([self.states, new_obs], dim=0)   # grows indefinitely
```

After initial load: 4000 transitions (all 4 episodes) in memory.  
After N promotions: 4000 + N × 1000 transitions in memory.  
At 200 promotions (typical 1M run): **204,000 transitions** in the teacher buffer.

**Mismatch: TF sees ≤ 1000 transitions at all times. PyTorch sees 4000–204,000.**

---

### A.2 Discriminator Sampling — What It Actually Sees

#### TF
- At init: disc samples from 1000 transitions = episode 3 only (return 6988.28, the best expert)
- After first promotion: disc samples from 1000 transitions = the PROMOTED student episode
- After second promotion: disc samples from 1000 transitions = the second PROMOTED student episode (episode 3 is gone)
- The ring always contains exactly the most recently written 1000 transitions (= most recently promoted episode, or episode 3 before any promotion)

#### PyTorch
- At init: disc samples from 4000 transitions spanning all 4 expert episodes
- After 200 promotions: disc samples from 204,000 transitions spanning all expert + all student episodes ever promoted

---

### A.3 Initial Promotion Threshold — The "Unsorted" Behavior

#### TF

`initialize_expert_buffer()` appends to `expert_scores` in episode arrival order:
```python
# Line 1363–1368 in TF sail.py:
if demo_dones[idx+1] == 1:
    episode_idx += 1
    self.expert_scores.append(episode_score)
```

Processing order for 4-episode dataset:
- Episode 0 ends at idx 998: `expert_scores.append(6932.27)`
- Episode 1 ends at idx 1998: `expert_scores.append(6780.64)`
- Episode 2 ends at idx 2998: `expert_scores.append(6741.27)`
- Episode 3 ends at idx 3997: `expert_scores.append(6988.28)`

**Result: `expert_scores = [6932.27, 6780.64, 6741.27, 6988.28]` — NOT sorted.**

Threshold check uses `expert_scores[0]` = **6932.27** (the first episode's return, not the minimum).

Confirmed from TF log:
```
Adding new trajectory with score 7154.933702 to replay buffer, expert-score 6932.27001953125
```

After first promotion, `expert_scores` gets sorted and the minimum drops to 6741.27:
```
Adding new trajectory with score 6907.162589 to replay buffer, expert-score 6741.2734375
```

#### PyTorch current behavior

```python
# train_sail.py line 212:
expert_scores = sorted(returns)  # [6741.27, 6780.64, 6932.27, 6988.28]
```

**Initial threshold = 6741.27 (the minimum), not 6932.27.**

**Mismatch: PyTorch's initial promotion threshold is ~191 points lower than TF's.**

---

### A.4 LfD Mixing with Capped Buffer

#### TF
LfD mixing draws 128 samples from `demo_replay_buffer` (1000 transitions = episode 3 only).  
Before promotion: the expert half of every critic batch comes from episode 3 exclusively.  
After promotion: the 1000-slot ring holds the newly promoted episode; expert half comes from that.

#### PyTorch current behavior
LfD mixing draws 128 samples from `teacher_buffer` (4000 transitions = all 4 expert episodes).  
The expert half is drawn uniformly from all 4 episodes simultaneously.

**Mismatch: TF LfD mixes from a single episode; PyTorch mixes from all 4.**

This is not necessarily a critical bug (all 4 episodes are expert quality), but it is a parity gap.

#### `lfd_active` flag

Current check: `teacher_buffer.num_transitions == teacher_buffer.initial_size`

With ring buffer: `num_transitions` is always ≤ `max_size` = 1000 (constant after first fill). This check would always be True. **Must be replaced.**

Fix: add `teacher_buffer._has_promotions` boolean, set True in `add_episode()`.  
New check: `lfd_mixing and not teacher_buffer._has_promotions`

---

### A.5 PrefRank — Buffer Interaction

#### TF

PrefRank samples pairs from `demo_replay_buffer` (the same 1000-slot ring). Before promotion, this contains only episode 3. The 4-episode ranking task is never presented — TF PrefRank only ever ranks within the single episode in the ring (rank of identical transitions → trivial → pref_loss near zero).

After first promotion: ring contains the student episode. TF PrefRank now ranks the student episode against itself — still trivial until a SECOND diverse episode appears.

This explains why the baseline PrefRank results in PyTorch converged despite pref_loss ≈ 0.001: **both TF and PyTorch PrefRank are in the same "trivial ranking" regime for most of training**.

#### PyTorch current behavior

`sample_pref_pairs()` samples from `self.pref_episodes` which tracks all episodes added via `add_episode()`. Initially this is 4 episodes. All 4 are scored and pairs are sampled between them.

**This is actually harder than TF's regime** (4 episodes with real score spread vs 1 episode). Explains why PyTorch's initial pref_loss is higher (0.756) before collapsing.

After the ring buffer fix: PyTorch will initialize pref_episodes with 1 episode only (episode 3), which matches TF exactly.

---

### A.6 Complete Mismatch Table

| # | Component | TF Behavior | PyTorch Current | Impact |
|---|-----------|-------------|-----------------|--------|
| B1 | Demo buffer size | 1000 (ring) | 4000 initial, unbounded growth | **CRITICAL**: discriminator sees 200× more data |
| B2 | Expert data in buffer after init | Episode 3 only (1000 transitions) | All 4 episodes (4000 transitions) | Disc trained on different distribution |
| B3 | Initial promotion threshold | `expert_scores[0]` = 6932.27 (insertion order) | `sorted(returns)[0]` = 6741.27 | 191-point higher bar in TF |
| B4 | LfD mixing source | Episode 3 only (1000 transitions) | All 4 episodes (4000 transitions) | Critic expert batches from different distribution |
| B5 | `lfd_active` check | `mix` flag set False on first promotion | `num_transitions == initial_size` | **Breaks with ring buffer** |
| B6 | Post-promotion buffer | Ring: always 1 episode | Append: accumulates all | Disc always sees different "expert" |
| B7 | PrefRank pair source at init | Episode 3 only (trivial pairs) | 4 episodes with score spread | Pref loss behavior differs |

---

## Part B — Implementation Plan

### B.1 File-Level Modifications

| File | Change | Reason |
|------|--------|--------|
| `sail_sb3/datasets/teacher_buffer.py` | Add ring buffer mode with `max_size` param | B1, B2, B4, B6 |
| `sail_sb3/algorithms/sail.py` | Fix `lfd_active` check | B5 |
| `sail_sb3/utils/callbacks.py` | Add `score_source` param + RM scoring logic | New RM feature |
| `sail_sb3/scripts/train_sail.py` | Add `--teacher_buffer_size` + `--adaptive_score_source` args; fix expert_scores init order | B3 + new RM feature |

---

### B.2 TeacherBuffer Ring Buffer (teacher_buffer.py)

Add `max_size` parameter. When set, use pre-allocated tensors and a ring write pointer.

**New fields:**
```python
self.max_size = max_size          # None = unbounded (backward compat)
self._next_idx = 0                # ring write pointer
self._has_promotions = False      # replaces num_transitions == initial_size check
```

**Init behavior:**
- Pre-allocate tensors: `self.states = torch.zeros(max_size, obs_dim, ...)`
- Track valid count: `self._count = 0` (grows to max_size, then fixed)
- Load expert data through ring: call internal `_ring_write(obs, act, rew, nobs, done)` for each transition
- After loading 4000 into 1000-slot ring: `_count = 1000`, `_next_idx = 0`, contains episode 3 only

**`add_episode()` ring write:**
```python
for each transition in episode:
    slot = self._next_idx
    self.states[slot] = obs_t
    ...
    self._next_idx = (self._next_idx + 1) % self.max_size
    self._count = min(self._count + 1, self.max_size)
self._has_promotions = True
```

**`sample_batch()` unchanged** — still random indices into `[0, self._count)`.

**`num_transitions` property** — returns `self._count` (always ≤ max_size).

**`initial_size` property** — returns `max_size` (for logging/compatibility; the ring is always "full" once loaded).

**`pref_episodes` with ring buffer:**  
Only build pref_episodes from what's in the ring at init = episode 3 only. After `add_episode()`, replace pref_episodes[0] with the new episode (ring semantics for the episode-level list too).

---

### B.3 SAIL lfd_active Fix (sail.py)

```python
# BEFORE:
lfd_active = (self.lfd_mixing and
              self.teacher_buffer.num_transitions == self.teacher_buffer.initial_size)

# AFTER:
lfd_active = (self.lfd_mixing and not self.teacher_buffer._has_promotions)
```

One line change. No other logic affected.

---

### B.4 Expert Scores — Insertion Order Fix (train_sail.py)

```python
# BEFORE:
expert_scores = sorted(returns)  # → [6741.27, 6780.64, 6932.27, 6988.28]

# AFTER (TF parity):
expert_scores = list(returns)    # → [6932.27, 6780.64, 6741.27, 6988.28] (insertion order)
```

The `returns` list is already built in episode order (ep_slices iteration), so `list(returns)` directly matches TF's append order.

**Effect:** Initial promotion threshold becomes 6932.27 (matching TF), not 6741.27.

---

### B.5 RM-Based Adaptive Promotion (New Feature)

#### New CLI argument:
```
--adaptive_score_source {gt, rm}    default: gt
```

#### GT mode (default): unchanged behavior.

#### RM mode:
- Student trajectory score = `sum(rm.reward(obs_ep, acs_ep))`
- Threshold = `rm_scores[0]` where `rm_scores` is initialized from expert episodes in insertion order
- Promotion condition: `student_rm_score > rm_threshold`
- After promotion: `rm_scores` updated and sorted (same logic as GT)

**Initialization in train_sail.py (RM mode):**
```python
# Compute RM score for each expert episode in insertion order
rm_expert_scores = []
ep_start = 0
for ep_i, ep_end in enumerate(ep_slice_indices):
    obs_ep = teacher_buffer.states[ep_start:ep_end+1].cpu().numpy()
    acs_ep = teacher_buffer.actions[ep_start:ep_end+1].cpu().numpy()
    rm_score = float(teacher_buffer.pref_rm.reward(obs_ep, acs_ep).sum())
    rm_expert_scores.append(rm_score)
    ep_start = ep_end + 1
# rm_expert_scores[0] = RM score of episode 0 = initial threshold
```

**Note:** With ring buffer, the physical buffer only contains episode 3. But `rm_expert_scores` tracks all 4 episodes in insertion order, matching TF's `expert_scores` tracking pattern.

**Callback changes (callbacks.py):**
```python
class SAILAdaptiveCallback:
    def __init__(self, ..., score_source='gt', rm_scores_list=None):
        self.score_source = score_source
        self.rm_scores = rm_scores_list  # only used in RM mode

    def _on_step(self):
        if done:
            if self.score_source == 'gt':
                student_score = info['episode']['r']
                threshold_list = self.expert_scores
            else:  # rm
                # Compute RM score over collected trajectory
                obs_ep = np.array([t[0] for t in traj])
                acs_ep = np.array([t[1] for t in traj])
                student_score = float(
                    self.teacher_buffer.pref_rm.reward(obs_ep, acs_ep).sum())
                threshold_list = self.rm_scores

            if student_score > threshold_list[0]:
                # promote
                self.teacher_buffer.add_episode(obs_ep, acs_ep, ...)
                # update threshold_list (sort after promotion)
                threshold_list.append(student_score)
                if len(threshold_list) >= 10:
                    threshold_list.pop(0)
                threshold_list.sort()
```

**No GT leakage in RM mode:** student score uses only `rm.reward()`, threshold uses only `rm_scores`. The `expert_scores` list (GT) is not touched in RM mode.

---

### B.6 RM-Mode Storage

RM score of the student trajectory is computed on demand at episode end using the episode_buffer's trajectory data. No persistent storage required — compute and discard. This avoids memory overhead.

For threshold tracking: `rm_scores` list uses same 10-element cap + sort logic as `expert_scores`.

---

### B.7 Validation Plan

From logs, verify:

**Buffer size (GT mode):**
```
[TeacherBuffer] Ring buffer: max_size=1000, loaded 4000 → count=1000
```
- `adaptive/teacher_buffer_size` should stay at 1000 throughout (no growth!)
- Discriminator should log different `disc_loss` behavior than the unbounded baseline

**Threshold (GT mode):**
```
[train_sail] Expert score threshold initialized to 6932.27 (first episode, insertion order)
```
- First promotion should be at a score > 6932.27 (not 6741.27)

**Promotion count:**
- Fewer promotions expected at 1M steps (higher initial threshold)
- Each promotion still overwrites the entire ring buffer

**RM mode:**
- `[SAIL-Adaptive] score_source=rm student_rm_score=... threshold_rm=...`
- No `episode['r']` in promotion log (confirms no GT leakage)
- Log format: `adaptive/student_rm_score`, `adaptive/rm_threshold`

---

## Part C — Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| Ring buffer LfD sampling: episode 3 dominates critic batches | Medium | This IS the TF behavior — document it |
| Higher initial threshold delays first promotion | Low | Expected — TF parity by design |
| RM scores may have different scale than env returns | Medium | Log both for comparison; flag scale difference |
| `pref_episodes` ring semantics for sample_pref_pairs | Low | Only 1 episode at init → pref_loss trivially small (matches TF) |
| RM mode requires `--pref_rm` path | Medium | Require `--pref_rm` if `--adaptive_score_source rm`; error cleanly if not provided |
| lfd_active with ring buffer (B5) | Low | Fixed by `_has_promotions` flag |

---

## Part D — sbatch Scripts Required

| Script | Config | Flags |
|--------|--------|-------|
| `HC_TFParity_Adaptive_LfD.sbatch` | TF-parity Adaptive, no PrefRank | `--adaptive --lfd_mixing --teacher_buffer_size 1000 --entcoeff 0.01` |
| `HC_TFParity_AdaptPref_LfD.sbatch` | TF-parity Adaptive + PrefRank | above + `--pref_rank_disc --pref_rank_weight 0.1 --pref_rm ...` |
| `HC_RMAdaptive_LfD.sbatch` | RM-scored Adaptive, no PrefRank | `--adaptive --lfd_mixing --teacher_buffer_size 1000 --adaptive_score_source rm --pref_rm ...` |
| `HC_RMAdaptPref_LfD.sbatch` | RM-scored Adaptive + PrefRank | above + `--pref_rank_disc --pref_rank_weight 0.1` |

All use `--entcoeff 0.01` (TF default), `--array=1-2`.

---

## Part E — What This Does NOT Change

- Reward formula (`-log(1 - sigmoid(logits) + 1e-8)`)
- Discriminator architecture (2×256 tanh)
- TD3 hyperparameters
- Gradient penalty
- Obs normalization in discriminator
- Post-promotion disc behavior
- 2× disc cadence (from previous TFDisc change)

---

*Sources: `stable-baselines/stable_baselines/deepq/replay_buffer.py`, `stable-baselines/stable_baselines/td3/sail.py` (lines 56–75, 246–247, 1343–1376, 1540–1569), `stable-baselines/run/settings.py` (line 70), TF logs `slurm_43780075_0.out` and `slurm_pref_rank_only_44405407_0.out`.*
