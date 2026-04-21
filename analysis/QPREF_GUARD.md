# QPREF Negative-Delta Safety Guard

## Background

QPREF adds a Bradley-Terry ranking loss on the TD3 critic that pushes Q-values to rank teacher
(positive) trajectories above student (negative) trajectories.  The signal is summarised by:

```
delta = (Q_pos_mean - Q_neg_mean) / temp
```

- **delta > 0**: critic correctly ranks teacher above student — QPREF is helping
- **delta ≈ 0**: no useful gradient signal — QPREF is neutral
- **delta < 0**: critic ranks student above teacher — QPREF is injecting harmful gradients

Empirically (run 48001996), delta starts strongly positive (~+18 at 150k steps) and declines
gradually to negative (~−0.5 at 900k steps) as the policy improves and the critic's Q-values
for student trajectories catch up to — or exceed — the teacher's.  Continuing to apply QPREF
after delta turns negative hurts the critic and degrades final performance.  The guard detects
this transition and permanently disables QPREF.

---

## Design

### Three-condition trigger gate

All three conditions must be satisfied simultaneously before the guard can fire:

```
Condition 1 — Warm-up complete (deque_full):
  deque_fill == 50  (50k env steps elapsed)
  Prevents spurious triggers from early Q warm-up noise.
  Q-values are not yet meaningful in the first ~50k steps.

Condition 2 — Positive phase demonstrated (peak_positive):
  _qpref_peak_guard_mean >= qpref_guard_positive_threshold  (default 1.0)
  QPREF must have been demonstrably useful at some earlier point.
  If guard_mean never reached 1.0, the negative delta is critic instability,
  NOT the intended late-stage policy-improvement scenario.

Condition 3 — Sustained negative signal (consecutive trigger):
  guard_mean < qpref_guard_threshold  for N consecutive train() calls
  (default: guard_mean < -0.3 for 3 consecutive calls)
  The recent 10-entry window has turned and stayed negative.
```

The guard fires — QPREF is permanently disabled — only when all three conditions hold.

### Why three conditions are needed

Empirical evidence from runs 48121988 and 48136319 revealed two failure modes of simpler guards:

**Failure mode A (fixed by Condition 1): Q warm-up noise**
Early in training (steps 1–50k), Q-values are untrained.  delta can be transiently negative
(-0.5 to -0.8) with no relationship to actual policy quality.  Condition 1 waits 50k steps
before arming.

**Failure mode B (fixed by Condition 2): Critic instability / divergence**
In run 48136319, a critic instability spike occurred at step 27k (critic_loss tripled in one
call).  From step 32k onward, Q_neg diverged far above Q_pos (Q_neg = 20–27, Q_pos = 9–18),
producing massive negative deltas (−7 to −14) with huge variance (std = 8–13).  The guard
armed at 52k with guard_mean = −10.5, fired at 53k — at only 53k/1M steps, before QPREF had
any meaningful positive impact.  The peak guard_mean in this run was only 0.35 (weak positive
signal during 12k–30k).  Condition 2 correctly blocks the trigger: 0.35 < 1.0 threshold.

The key diagnostic difference between the two root causes:
| Cause | Max guard_mean before going negative | Delta magnitude |
|-------|--------------------------------------|-----------------|
| Critic instability (run 48136319) | **0.35** | Large (−7 to −14), high std |
| Late-stage policy improvement (ref 48001996) | **~18** | Gradual decline (−0.5), low std |

A peak guard_mean threshold of 1.0 cleanly separates these cases.

### Rolling deque

One `float` entry is appended to `_qpref_delta_deque` (maxlen=50) per `train()` call.  With
default settings (`qpref_grad_interval=50`, `gradient_steps=1000`), QPREF fires 20 times per
`train()` call and their deltas are averaged into one entry.  One `train()` call ≈ 1000 env
steps, so the 50-entry deque covers the last 50k env steps.

### Guard decision window

The guard compares `np.mean(deque[-qpref_guard_window:])` (default: last 10 entries = last 10k
env steps) against the threshold — not the full 50-entry mean.

**Why not the full 50-entry mean?**  If delta is +18 for the first 40 calls and −1 for the
last 10, the full-deque mean stays positive and the guard never fires.  The 10-entry window
detects the recent decline immediately.  The full 50-entry mean is still logged as
`qpref_delta_rolling_mean` for trend visualisation but does not control the trigger.

### Counter and reset

After all arm conditions are met, each `train()` call with `guard_mean < threshold` increments
`_qpref_guard_count`.  A single call with `guard_mean >= threshold` resets it to 0 (recovery).
When the counter reaches `qpref_guard_confirm`, QPREF is permanently disabled.

---

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--qpref_guard_threshold` | `-0.3` | Recent-window mean delta below which the guard increments its counter |
| `--qpref_guard_confirm` | `3` | Consecutive `train()` calls below threshold needed to trigger (~3k env steps) |
| `--qpref_guard_window` | `10` | Recent deque entries used for guard decision (~10k env steps) |
| `--qpref_guard_positive_threshold` | `1.0` | Peak guard_mean that must be reached before the guard can ever fire |

The arm condition (full 50-entry deque) is not user-configurable.

---

## State variables (in `sail.py`)

| Variable | Type | Purpose |
|----------|------|---------|
| `_qpref_active` | `bool` | `True` until guard fires; gating flag for QPREF loss computation |
| `_qpref_guard_triggered` | `bool` | Latched `True` when guard fires; prevents re-arming |
| `_qpref_guard_count` | `int` | Consecutive below-threshold intervals so far |
| `_qpref_skipped_updates` | `int` | Cumulative QPREF gradient steps skipped after disable |
| `_qpref_delta_deque` | `deque(maxlen=50)` | Circular buffer of per-`train()`-call mean deltas |
| `_qpref_peak_guard_mean` | `float` | Maximum guard_mean ever observed; must reach `positive_threshold` before guard can fire |

---

## Logged metrics

All metrics are under the `train/` prefix in W&B / SB3 logger.

| Metric | When logged | Meaning |
|--------|-------------|---------|
| `qpref_active` | always (when `--qpref`) | 1 = QPREF running, 0 = disabled by guard |
| `qpref_guard_triggered` | always | 1 = guard has fired at least once |
| `qpref_guard_count` | always | Current consecutive-below-threshold counter |
| `qpref_skipped_updates` | always | Cumulative gradient steps skipped post-disable |
| `qpref_delta` | when QPREF ran | Mean delta for this `train()` interval |
| `qpref_delta_std` | when QPREF ran | Std of delta within this interval |
| `qpref_delta_rolling_mean` | when deque non-empty | Full 50-entry deque mean (trend context only) |
| `qpref_delta_guard_mean` | when deque non-empty | Recent `guard_window`-entry mean (actual trigger signal) |
| `qpref_peak_guard_mean` | when deque non-empty | Running maximum of `guard_mean`; tracks whether positive phase was reached |
| `qpref_deque_fill` | when deque non-empty | Current deque length (0–50; guard fully arms at 50) |
| `qpref_loss` | when QPREF ran | Mean Bradley-Terry loss this interval |
| `qpref_mean_q_pos` | when QPREF ran | Mean Q of positive (teacher) trajectories |
| `qpref_mean_q_neg` | when QPREF ran | Mean Q of negative (student) trajectories |
| `qpref_pairs_available` | always | Number of intervals where a valid pair was sampled |
| `qpref_source_teacher_pool_size` | always | Episodes in teacher pool |
| `qpref_source_student_pool_size` | always | Episodes in student pool |

`qpref_delta_rolling_mean`, `qpref_delta_guard_mean`, and `qpref_peak_guard_mean` continue to
be logged after the guard fires (using stale deque values), so the last known state remains
visible in W&B.

---

## Code locations

| File | Lines | What |
|------|-------|------|
| `sail_sb3/algorithms/sail.py` | constructor params ~44–47 | `qpref_guard_threshold`, `qpref_guard_confirm`, `qpref_guard_window`, `qpref_guard_positive_threshold` |
| `sail_sb3/algorithms/sail.py` | `__init__` ~80–91 | State variable initialisation including `_qpref_peak_guard_mean` |
| `sail_sb3/algorithms/sail.py` | `train()` ~508–517 | `qpref_would_run` / `_qpref_active` gate |
| `sail_sb3/algorithms/sail.py` | `train()` ~690–750 | Deque append, rolling means, peak tracker, guard check, trigger |
| `sail_sb3/scripts/train_sail.py` | ~214–240 | CLI arg definitions |
| `sail_sb3/scripts/train_sail.py` | ~530–534 | Passed to SAIL constructor |

---

## Guard trigger timeline (expected for a healthy 1M-step run)

```
Step 0–50k:   deque_fill 0→50.  Guard disarmed (Condition 1 not met).
              delta noisy: early negative during Q warm-up (1–25k), then recovering.
              peak_guard_mean accumulates but likely < 1.0 during this phase.

Step 50k–150k: Guard armed (Condition 1 met).  delta strongly positive (+5 to +18).
              peak_guard_mean rises quickly above 1.0 (Condition 2 met).
              guard_mean > 0, guard_count stays 0.

Step ~300–600k: delta begins declining toward 0 as policy improves.
              peak_guard_mean stable at its maximum (no longer updated).

Step ~700–900k: delta crosses 0 → becomes negative.
              guard_mean < -0.3 starts accumulating guard_count.
              After 3 consecutive intervals: all 3 conditions satisfied →
              → _qpref_active = False, _qpref_guard_triggered = True
              → "[QPREF GUARD] Triggered at step X" printed to stdout
              → QPREF loss no longer computed; skipped_updates increments.
```

---

## Known design decisions and trade-offs

**Warm-up period (50 entries = 50k steps):**  Fixed at `deque.maxlen`.  Calibrated to match
Q warm-up duration on HalfCheetah-v2 with these hyperparameters.

**Positive threshold default (1.0):**  Chosen empirically from comparing run 48136319 (critic
instability, peak=0.35) versus run 48001996 (healthy, peak≈18).  A value of 1.0 cleanly
separates the two cases with headroom on both sides.  If delta is consistently weak-positive
(0.1–0.5) for a long run, the guard will never fire even if it turns negative — which is
correct, since a weak positive phase indicates QPREF was never strongly beneficial.

**Permanent disable:**  Once triggered, QPREF cannot re-activate.  A recovery after a long
negative phase is treated as transient noise.  The guard is a one-way safety latch.

**Recovery resets the counter:**  One `train()` call with `guard_mean >= threshold` resets
`guard_count` to 0.  Adjust `qpref_guard_confirm` for more/less tolerance.

**`qpref_delta_guard_mean` vs `qpref_delta_rolling_mean` vs `qpref_peak_guard_mean`:**  Three
complementary views.  Peak tracks the historical maximum.  Guard mean is the current trigger
signal.  Rolling mean is the long-horizon trend.  All three should be plotted together in W&B.
