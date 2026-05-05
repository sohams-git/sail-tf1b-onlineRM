# QPREF Cross-Pool Pairing — Implementation Plan
**Goal:** Fix weak QPREF signal in early training by sampling (teacher_ep, student_ep) pairs
instead of (student_ep, student_ep) pairs when the student pool is small and low-diversity.  
**Scope:** QPREF-only. Zero changes to PrefRank, Soft-TAC, Reweight, or the existing
`teacher` and `student` source modes.

---

## 1. Current QPREF Structure Relevant to This Change

### Data flow today

```
Callback (every episode, qpref_source='student')
  └─ teacher_buffer.add_student_episode(obs, acs)
       └─ scores with pref_rm.reward() → J
       └─ appends to teacher_buffer.pref_student_episodes[]  (FIFO cap: 500)

train() inner loop (every qpref_grad_interval gradient steps)
  └─ teacher_buffer.sample_qpref_pairs_aggregate(batch_size, source=self.qpref_source)
       └─ if source='teacher': pools from pref_episodes[]          (expert + promoted)
       └─ if source='student': pools from pref_student_episodes[]  (all students)
       └─ samples B pairs within one pool, assigns pos/neg by J, pads to T_max
       └─ returns (pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask)
  └─ eval critic on padded trajectories → masked-mean Q per trajectory
  └─ delta = Q(pos) - Q(neg) ; qpref_loss = softplus(-delta).mean()
  └─ critic_loss += qpref_weight * qpref_loss
```

### Files involved

| File | Role in QPREF |
|------|--------------|
| `datasets/teacher_buffer.py` | Stores both pools; `add_student_episode()` populates student pool; `sample_qpref_pairs_aggregate()` samples within one pool |
| `algorithms/sail.py` | Calls `sample_qpref_pairs_aggregate()`, runs critic forward pass, computes loss, accumulates log metrics, runs guard check |
| `utils/callbacks.py` | Calls `add_student_episode()` after every student episode when `qpref_source='student'` |
| `scripts/train_sail.py` | Parses `--qpref_source`, passes to SAIL and callback constructors |

### The specific gap

`sample_qpref_pairs_aggregate()` currently reads from **one pool only**. There is no path that can draw one episode from `pref_episodes` and another from `pref_student_episodes`. The fix requires a new method and a new source string that routes to it.

---

## 2. Proposed Cross-Pool Design

### New source value: `'cross_pool'`

Extend the existing `--qpref_source` argument with a third value:

| Value | Pair composition | When to use |
|-------|-----------------|-------------|
| `'teacher'` | teacher_ep vs teacher_ep (within `pref_episodes`) | existing mode, unchanged |
| `'student'` | student_ep vs student_ep (within `pref_student_episodes`) | existing mode, unchanged |
| **`'cross_pool'`** | **teacher_ep vs student_ep** (one from each pool) | **new mode** |

**The key invariant:**  
With `cross_pool`, J(teacher) ≈ 1333–1374 (RM scores) and J(student) ≈ −300 to −500 early, or growing as policy improves. The spread is ~1600–1800 from step 3k, compared to ~50–100 within a homogeneous early student pool. In late training when student episodes exceed teacher quality (student J > teacher J), the assignment naturally flips: student becomes `pos`, teacher becomes `neg`. Both directions are correct — the QPREF loss is symmetric.

### No new pool structures

`cross_pool` draws from `pref_episodes` and `pref_student_episodes`, which already exist. No new data structures are needed.

---

## 3. Early-Training Gating Strategy

### Option analysis

| Option | Mechanism | Pro | Con |
|--------|-----------|-----|-----|
| **Permanent cross-pool** | Always sample (teacher, student) | Simple, no new params, predictable | In late training when policy converges, cross-pool J-spread may shrink (student J approaches teacher J); mild issue |
| Timestep threshold | Switch at step T | Predictable | Arbitrary, not adaptive to pool growth rate |
| J-spread adaptive | Switch when `std(J_student) > σ` | Adapts to policy speed | Harder to implement, more moving parts |
| Pool-size warmup | Cross-pool until `len(student_pool) >= N_min`, then switch to student | Adapts to pool growth, interpretable | Needs one extra parameter |

### Recommendation: permanent `cross_pool` mode + optional warmup gate

**Primary design:** `--qpref_source cross_pool` is a **permanent mode** — it always samples one teacher and one student episode. This is the simplest possible thing that addresses the early-training problem.

**Optional extension:** add `--qpref_cross_pool_min_student N` (default 0, i.e., no effect). When set, cross-pool sampling is used when `len(student_pool) < N`, and pure student-student when `len(student_pool) >= N`. This is an optional warmup gate that defaults to off.

**Why permanent is recommended:**  
In the smoke test log, the student pool reaches 500 (cap) at step ~550k and stays there. At that point the teacher pool has J ~1333 and student pool has J ~1000–1400+ (late-phase policy). Cross-pool J-spread is still ~0–400, which is smaller than the productive phase but still non-trivial. The guard handles late-phase delta collapse regardless of source. Making cross-pool permanent avoids the question of "what happens after the switch" and removes the student-pool homogeneity late-phase failure mode simultaneously.

If warmup-then-switch is preferred, `--qpref_cross_pool_min_student 50` is the right value based on the smoke test (the pool has meaningful diversity and the disc is mature by step ~50k when pool=49).

---

## 4. Isolation from Other Variants

### PrefRank discriminator loss
Uses `teacher_buffer.sample_pref_pairs()` which reads only `pref_episodes`. **No change.**

### Soft-TAC discriminator loss
Uses `teacher_buffer.sample_soft_tac_pairs()` and `_soft_tac_expert_episodes`. **No change.**

### Pref-reweight teacher (Boltzmann)
Uses `teacher_buffer.sample_pref_weighted_expert()` which reads `pref_episodes`. **No change.**

### Existing QPREF 'teacher' mode (`--qpref_source teacher`)
Routes to `sample_qpref_pairs_aggregate(..., source='teacher')`. That method is unchanged. **No change.**

### Existing QPREF 'student' mode (`--qpref_source student`)
Routes to `sample_qpref_pairs_aggregate(..., source='student')`. That method is unchanged. **No change.**

### New cross-pool mode
Only activates when `qpref_source == 'cross_pool'`. Isolated to the new routing branch and new method.

### Teacher pool (`pref_episodes`) read access
`cross_pool` reads `pref_episodes` for sampling but **never writes to it**. No conflict with promotion logic or reweight weight computation.

---

## 5. Required Code Changes

### Change 1 — `teacher_buffer.py`: new method `sample_qpref_pairs_cross_pool()`

**Location:** After `sample_qpref_pairs_aggregate()` (line ~530).

**Logic:**
```
def sample_qpref_pairs_cross_pool(self, batch_size: int):
    """
    Sample batch_size pairs where each pair is one teacher episode (from pref_episodes)
    vs one student episode (from pref_student_episodes).
    Assign pos/neg by J comparison (either can be positive depending on quality).
    Returns None if either pool is empty.
    Returns same 6-tuple shape as sample_qpref_pairs_aggregate().
    """
    if len(self.pref_episodes) == 0 or len(self.pref_student_episodes) == 0:
        return None

    # Sample teacher indices from pref_episodes (size N_t)
    N_t = len(self.pref_episodes)
    N_s = len(self.pref_student_episodes)
    t_idx = np.random.randint(0, N_t, size=batch_size)
    s_idx = np.random.randint(0, N_s, size=batch_size)

    pos_eps_sel, neg_eps_sel = [], []
    j_spreads = []
    for ti, si in zip(t_idx, s_idx):
        ep_t = self.pref_episodes[ti]
        ep_s = self.pref_student_episodes[si]
        spread = abs(float(ep_t['J']) - float(ep_s['J']))
        j_spreads.append(spread)
        if float(ep_t['J']) >= float(ep_s['J']):
            pos_eps_sel.append(ep_t)
            neg_eps_sel.append(ep_s)
        else:
            pos_eps_sel.append(ep_s)
            neg_eps_sel.append(ep_t)

    # Pad to T_max — identical to sample_qpref_pairs_aggregate padding logic
    # (copy-paste the padding block; no change in output shape)
    T_max   = max(max(ep_len(e) for e in pos_eps_sel),
                  max(ep_len(e) for e in neg_eps_sel))
    ... [same padding block as sample_qpref_pairs_aggregate] ...

    return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, np.mean(j_spreads)
    # Note: returns 7-tuple when cross-pool (extra j_spread_mean scalar for logging).
    # Caller checks tuple length to distinguish from 6-tuple.
```

**Return shape consideration:**  
To avoid changing the existing call path, prefer returning the J-spread as a separate attribute rather than appending to the tuple. Simplest: return the 6-tuple and let the caller compute spread from the logged delta. Or: add a `_last_qpref_j_spread` attribute that the method sets internally before returning, which the caller reads for logging. The latter avoids any interface change for the existing tuple-unpack in `sail.py`.

**Preferred design:** Set `self._last_qpref_j_spread_mean = np.mean(j_spreads)` as a side effect inside the method. Return the same 6-tuple as before. The caller reads `self.teacher_buffer._last_qpref_j_spread_mean` for logging after the call.

---

### Change 2 — `teacher_buffer.py`: optional warmup gate in `sample_qpref_pairs_cross_pool()`

If `--qpref_cross_pool_min_student N` is implemented, the method receives the threshold and switches internally:

```python
def sample_qpref_pairs_cross_pool(self, batch_size, min_student=0):
    if min_student > 0 and len(self.pref_student_episodes) >= min_student:
        # Enough student episodes — fall back to student-student
        return self.sample_qpref_pairs_aggregate(batch_size, source='student')
    # else: cross-pool
    ...
```

This keeps the switch logic inside the buffer method rather than in `sail.py`.

---

### Change 3 — `algorithms/sail.py`: route `cross_pool` source

**Location:** Lines 509–511 (the `if qpref_active:` block).

**Current:**
```python
pair = self.teacher_buffer.sample_qpref_pairs_aggregate(
    self.qpref_batch_size, source=self.qpref_source)
```

**Change to:**
```python
if self.qpref_source == 'cross_pool':
    pair = self.teacher_buffer.sample_qpref_pairs_cross_pool(
        self.qpref_batch_size,
        min_student=self.qpref_cross_pool_min_student,
    )
else:
    pair = self.teacher_buffer.sample_qpref_pairs_aggregate(
        self.qpref_batch_size, source=self.qpref_source)
```

Nothing else in the `train()` function changes. The `pair` variable is still `None | 6-tuple`, and the downstream critic forward pass is identical.

**New attribute on SAIL:**  
Add `self.qpref_cross_pool_min_student: int = 0` in `__init__`. This is 0 by default (permanent cross-pool) and is only set when `--qpref_cross_pool_min_student N` is passed.

---

### Change 4 — `algorithms/sail.py`: new log metric in the QPREF logging block

**Location:** After the `if self._pending_qpref_losses:` block (~line 668).

Add:
```python
if self.qpref_source == 'cross_pool':
    j_spread = getattr(self.teacher_buffer, '_last_qpref_j_spread_mean', None)
    if j_spread is not None:
        self.logger.record("train/qpref_j_spread_mean", float(j_spread))
    # Also log which sub-mode is active (cross vs student fallback)
    cross_active = (len(self.teacher_buffer.pref_student_episodes)
                    < max(self.qpref_cross_pool_min_student, 1))
    self.logger.record("train/qpref_cross_pool_active", int(cross_active))
```

---

### Change 5 — `utils/callbacks.py`: populate student pool for `cross_pool` source

**Location:** Line 192.

**Current:**
```python
if self.qpref_source == "student" and obs_ep is not None and len(obs_ep) > 0:
    self.teacher_buffer.add_student_episode(obs_ep, acs_ep)
```

**Change to:**
```python
if self.qpref_source in ("student", "cross_pool") \
        and obs_ep is not None and len(obs_ep) > 0:
    self.teacher_buffer.add_student_episode(obs_ep, acs_ep)
```

This is the only change in the callback. `add_student_episode` is already a no-op if `pref_rm` is not loaded, so this is safe.

---

### Change 6 — `scripts/train_sail.py`: flag additions

**Existing flag to extend:**
```python
parser.add_argument("--qpref_source", type=str, default="teacher",
                    choices=["teacher", "student"],   # ADD "cross_pool"
                    help="...")
```
Change to `choices=["teacher", "student", "cross_pool"]`.

**New optional flag:**
```python
parser.add_argument("--qpref_cross_pool_min_student", type=int, default=0,
    help="Cross-pool → student-student switch threshold. When qpref_source=cross_pool "
         "and student pool reaches this size, switches to student-student sampling. "
         "0 = permanent cross-pool (default).")
```

**Startup validation (near line 462):** Add a warning when `qpref_source='cross_pool'` and `pref_rm` is not loaded (student episodes won't be scored and pool will stay empty).

**Pass to SAIL constructor (near line 566):**
```python
qpref_cross_pool_min_student=args.qpref_cross_pool_min_student,
```

**Pass to callback constructor (near line 642):**
The callback already receives `qpref_source=args.qpref_source`. No additional parameter needed since `add_student_episode` routing is purely on the source string.

---

## 6. New Logging

| Metric | Where | What it shows |
|--------|-------|---------------|
| `train/qpref_j_spread_mean` | `sail.py`, QPREF logging block | Mean \|J_pos − J_neg\| per QPREF batch. Key diagnostic: should be ~1600 early (cross-pool) vs ~50-100 (student-student early). Confirms cross-pool is giving larger spread. |
| `train/qpref_cross_pool_active` | `sail.py`, QPREF logging block | 1 when cross-pool sampling is active, 0 when fallen back to student-student (only relevant if `min_student > 0`). Shows the transition moment. |
| `train/qpref_source_teacher_pool_size` | already exists | Already logged — teacher pool size. Should be 4 at start, growing if adaptive promotions happen. |
| `train/qpref_source_student_pool_size` | already exists | Already logged — student pool size. Watch this grow from 0 toward the switch threshold. |
| `train/qpref_delta` | already exists | Should be strongly positive from step ~3k with cross-pool. Compare against baseline (same run with `student` source) to confirm improvement. |

**No new log metrics are needed for validation** beyond `qpref_j_spread_mean` and `qpref_cross_pool_active`. All other diagnostic metrics (`qpref_delta`, `qpref_loss`, `qpref_mean_q_pos/neg`, guard metrics) are already logged and will reveal the improvement directly.

---

## 7. Recommended Final Design

### Primary recommendation: `--qpref_source cross_pool` (permanent, no warmup gate)

**One new flag:** `--qpref_source cross_pool`  
**One new flag (optional):** `--qpref_cross_pool_min_student 0` (default: permanent cross-pool)  
**No other new flags.**

**Rationale:**

The smoke test (48463312) shows the specific timeline:
- Steps 3k–60k: student pool J-spread ~50–100 → delta −0.114 to −1.31 → Q degrades 43%
- Steps 60k+: teacher J ~1333 − student J ~−400 to +1300 → spread ~33 to ~1733 → delta +2 to +15

Permanent cross-pool provides this large spread from step 3k, not step 60k. The expected early delta at step 3k with cross-pool (teacher J=1353, student J≈−300): `|ΔJ| ≈ 1650`. Even with only 1000 critic updates and Q values of ~1.4, this large contrast gives the Q function a much clearer gradient direction. The Q function does not need to be well-calibrated to know that a 1650-point J difference is meaningful.

In late training (step 500k+), student episodes approach teacher quality (student J → 1200–1400 RM). Cross-pool spread drops to ~0–200. This is when the guard provides protection, which it already does correctly (fires at 757k in the smoke test). So cross-pool + guard handles both failure phases.

**Total code change footprint:**
- `teacher_buffer.py`: 1 new method (~30 lines), 1-line `_last_qpref_j_spread_mean` side effect
- `sail.py`: 4-line routing branch, 6-line logging addition, 1 new `__init__` attribute
- `callbacks.py`: 1-line condition change (`"student"` → `"student", "cross_pool"`)
- `train_sail.py`: extend `choices`, 1 new argument, 1 new startup validation, pass to SAIL

**Sbatch change for next smoke test:**
```bash
# Replace
--qpref_source student
# With
--qpref_source cross_pool
# (optionally add)
--qpref_cross_pool_min_student 0   # permanent (default)
```

### Expected observables to confirm the fix

After implementing and running a 60k smoke test with `--qpref_source cross_pool`:

1. `train/qpref_j_spread_mean` at step 3k: should be ~1600 (vs current ~50)
2. `train/qpref_delta` at step 3k: should be **positive** (vs current −0.114)
3. `train/qpref_mean_q_pos > qpref_mean_q_neg` from first update: teacher episodes get higher Q
4. `rollout/ep_rew_mean` at step 30k: should be better than −572 (current smoke test) because Q is not degraded during 3k–30k window
5. `train/critic_loss` at step 30k: should be lower (~0.1–0.15 vs current ~0.265)
6. `train/q1_mean` trajectory: should NOT show the 43% drop (22.6 → 12.9) seen in the current run

If `qpref_j_spread_mean` is high and `qpref_delta` turns positive early, the early Q degradation is eliminated. The guard can then focus on the (less severe) late-phase saturation problem.
