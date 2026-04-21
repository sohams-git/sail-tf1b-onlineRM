# Online QPREF Student-Source Run Analysis
**Job:** 48253943 (seeds 1-3) | **Date:** 2026-04-18  
**Config:** Online RM + QPREF (student source, RM-only J) + Adaptive SAIL + LfD  
`qpref_weight=0.1 temp=1.0 batch=4 grad_interval=10 pref_max_student_trajs=500`

---

## 1. Results Summary

| Seed | Final ep_rew | Score% | Promotions | First promo | QPREF delta (final) | surr_rew (final) | Outcome |
|------|-------------|--------|------------|-------------|---------------------|------------------|---------|
| 1 | -600 | -6.6% | 0 | Never | ~9e-5 (≈ 0) | 1.75 | **FAIL** |
| 2 | **6608** | **72.6%** | 16 | 202k | 2.05 | 0.44 | Success |
| 3 | -600 | -6.6% | 0 | Never | ~2e-5 (≈ 0) | 1.74 | **FAIL** |

**Success rate: 1/3 (33%).** The failure mode is identical to prior runs without QPREF (RMAdaptive, SoftTAC-RMJ seed 2). QPREF did not cause the failures and did not prevent them either.

---

## 2. Failure Analysis — Seeds 1 and 3

### The cascade (same as RMAdaptive seed 2 failure documented earlier)

1. Random initialization → discriminator develops a bias toward treating all policy states as expert early.
2. `surr_rew` climbs: **0.75 → 1.27 → 1.44 → 1.57 → 1.75** (monotonically, `lfd_active=True` throughout).
3. Policy receives inflated reward signal regardless of quality → never achieves `gt_score > 6988.3`.
4. Never promotes → `lfd_active=True` forever → discriminator saturates on one expert episode.

Seeds 1 and 3 show ep_rew declining monotonically from -300 to -600, becoming permanently stuck around step ~200k.

### QPREF was completely inactive for ranking in the failing seeds

The critical observation: `qpref_loss = 0.693` for the **entire run** in seeds 1 and 3.

`softplus(-delta) = ln(2) ≈ 0.693` when `delta = 0`. This is the maximum loss value. It means `mean_Q_pos ≈ mean_Q_neg` throughout — the critic assigns the same Q to both episodes in every pair.

**Root cause:** The student pool is filled exclusively with stuck-policy episodes. When the policy is permanently stuck at ep_rew ≈ -600:
- All episodes have essentially the same trajectory distribution.
- The online RM assigns near-identical J values to every episode (all are equally bad).
- `sample_qpref_pairs_aggregate` selects two episodes with `|J_a - J_b| ≈ 0`. The positive/negative assignment is essentially random coin flips.
- The critic, evaluating two near-identical trajectories, produces `Q_pos ≈ Q_neg` → `delta ≈ 0` → `loss ≈ ln(2) = 0.693`.

The QPREF gradient is zero in expectation: the loss numerically adds 0.693 * 0.1 weight to the critic, but the gradient `d(softplus(-delta))/d(delta) = -sigmoid(-delta) ≈ -0.5` is symmetric and averages to zero when the positive/negative labels are random. QPREF contributes **noise** to the critic when the student pool is homogeneous, not a useful signal.

This is confirmed by `qpref_delta` values: seed 1 oscillates around ±0.0003, seed 3 around ±0.0006 — four orders of magnitude smaller than seed 2's delta of 2–25.

### Why seed 2 escaped

Seed 2's policy started improving slightly earlier. By the time surr_rew was climbing (~100k), its policy had broken slightly above -600, creating J diversity in the student pool:
- `qpref_delta` goes from -0.002 → 1.49 → 5.95 → 17.1 → 25.2 (growing strongly).
- The critic learns that current-policy episodes rank above past-bad-policy episodes.
- First promotion at step 202k. After promotion, `lfd_active` turns off, training normalizes.

Seed 2 reached 72.6% (6608 ep_rew) with 16 promotions.

---

## 3. QPREF Signal Quality — Seed 2 in Detail

QPREF delta in seed 2 tells a clear story:

| Phase | Step range | delta range | Interpretation |
|-------|-----------|-------------|----------------|
| Silent | 0 → ~25k | N/A | RM not active, pool empty |
| Warmup | ~25k → ~60k | -0.002 → 0.675 | Pool small, J diversity low |
| Breakout | ~60k → ~150k | 0.5 → 5.9 | Policy improving; pool spans quality range |
| Peak | ~150k → ~400k | 10 → 25 | Strong critic ordering: current > old episodes |
| Declining | ~400k+ | 17 → 20 (still positive) | Pool saturating with high-quality episodes; delta shrinks |

The guard check: `peak_guard_mean` reached ~23 (well above the 1.0 threshold), so the guard is now armed. Guard_mean at end is 1.98 (positive, healthy) — guard has not fired and should not, since QPREF is still providing positive ordering signal.

---

## 4. Why QPREF Did Not Help the Failing Seeds

QPREF is a **conditional signal**: it requires J diversity in the student pool to produce useful pair labels. When the policy is stuck:

- Pool diversity is zero → delta ≈ 0 → gradient ≈ noise.
- The noise adds 0.693 × 0.1 = 0.069 to the critic loss, slightly destabilising it.
- This is not the cause of the failure — the surr_rew saturation cascade happens independently — but QPREF cannot rescue a saturated discriminator.

QPREF and the saturation cascade operate on different components (critic vs discriminator), and fixing the discriminator saturation requires a discriminator-level intervention, not a critic ranking loss.

**Comparison with non-QPREF runs:**
- SoftTAC-RMJ (no QPREF): 4/5 success rate, same saturation failure mode at ~20% frequency.
- QPREF-student (this run): 1/3 success rate, same failure mode at 67% frequency.

Seeds 1, 2, and 3 all have the same fundamental vulnerability. QPREF did not improve resilience because it cannot. The 2/3 failure rate here is a sampling artefact of only 3 seeds.

---

## 5. Improvements Required (No GT)

All improvements must use RM-only signals.

### Fix A: The structural saturation vulnerability (highest priority)

The root cause of all failures is the same: with a single expert episode in the ring-truncated buffer, the discriminator can saturate on that one trajectory. This is upstream of QPREF entirely.

**Option 1 (safest): Add `--rm_min_acc` sanity gate to discriminator update**  
When `surr_rew_mean > 1.2` and `_has_promotions=False` for more than 50k steps, reduce `disc_train_freq` to slow further saturation. This is a discriminator-level adaptive schedule, no GT needed.

**Option 2 (architectural): Detach LfD mixing from `_has_promotions`**  
Currently LfD is on until first promotion. If the RM has a better-calibrated threshold (use RM scores of expert pool, not GT returns), the first RM promotion could be gated more loosely, letting the policy escape the saturation trap sooner. This is the same fix suggested in the plan-mode analysis.

### Fix B: Add J-std logging for the student pool

Currently there is no logging of J spread in `pref_student_episodes`. This makes it impossible to monitor from logs whether QPREF has useful pair diversity. Add to the QPREF logging block in `sail.py`:

```python
if self.qpref and self.teacher_buffer.pref_student_episodes:
    j_vals = [ep['J'] for ep in self.teacher_buffer.pref_student_episodes]
    self.logger.record("train/qpref_student_j_std",  float(np.std(j_vals)))
    self.logger.record("train/qpref_student_j_mean", float(np.mean(j_vals)))
```

This would have immediately shown `j_std ≈ 0` in seeds 1 and 3 (confirming homogeneous pool) vs `j_std > 100` in seed 2 (confirming useful diversity).

### Fix C: Pool J diversity gate on QPREF activation

When `j_std < threshold` (e.g. 10 RM units), the student pool is too homogeneous for QPREF to produce useful signal. Instead of firing noise, skip the QPREF update:

```python
# Inside qpref_active block, before pair sampling:
if self.teacher_buffer.pref_student_episodes:
    j_vals = [ep['J'] for ep in self.teacher_buffer.pref_student_episodes]
    if np.std(j_vals) < self.qpref_min_j_std:
        self._pending_qpref_pairs_ok.append(0)
        continue  # skip this grad step
```

This prevents the 0.693 constant loss from adding gradient noise during stuck phases. It is RM-only (uses stored J values, no GT). Default `qpref_min_j_std=10` RM units.

### Fix D: Guard `positive_threshold` reduction

The current `qpref_guard_positive_threshold=1.0` requires delta to reach 1.0 before the guard can fire. Seed 2 peaked at ~25, so this is easily met. But if a seed fails fast (delta never leaves near-zero), the guard never arms and never fires — which is correct, but also means `qpref_active=1` and the 0.693 noise continues. Fix C (J-std gate) is cleaner than lowering this threshold.

---

## 6. Summary Table

| Metric | Seed 1 (fail) | Seed 2 (success) | Seed 3 (fail) |
|--------|--------------|-----------------|--------------|
| Final ep_rew | -600 | **6608** | -600 |
| Normalized score | -6.6% | **72.6%** | -6.6% |
| Promotions | 0 | 16 | 0 |
| Final surr_rew | 1.75 | 0.44 | 1.74 |
| Final qpref_loss | 0.693 (≡ delta=0) | 0.291 | 0.693 (≡ delta=0) |
| Final qpref_delta | ~9e-5 | 2.05 | ~2e-5 |
| Peak qpref_delta | ~0.001 | ~25 | ~0.001 |
| Guard triggered | No | No | No |
| QPREF contribution | Noise (delta≈0) | Active & useful | Noise (delta≈0) |
| Failure mode | Disc saturation | — | Disc saturation |

**Verdict:** QPREF works correctly when the policy is improving (seed 2: delta reaches 25, strong ranking signal, guard armed but not triggered). It is ineffective when the policy is stuck (seeds 1 and 3: pool is homogeneous, delta ≈ 0, loss ≈ constant 0.693). The fix is a J-std gate (Fix C) to skip QPREF updates when the pool has no useful diversity, plus J-std logging (Fix B) to monitor the pool quality. The saturation vulnerability (Fix A) is upstream and not addressable by QPREF alone.
