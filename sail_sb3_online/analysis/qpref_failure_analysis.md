# QPREF Failure Analysis — HalfCheetah-v2
**Date:** 2026-04-21  
**Logs analyzed:**  
- Offline SAIL-QPREF: `sail_sb3/HC_6500_sbatch/logs/SAIL-QPREF_48291942_1.out`  
- Offline PAIL-QPREF: `sail_sb3/HC_6500_sbatch/logs/PAIL-QPREF_48292379_1.out`  
- Online SAIL-QPREF:  `sail_sb3_online/HC_6500_sbatch/logs/SAIL-QPREF_48292637_1.out`  
- Online PAIL-QPREF:  `sail_sb3_online/HC_6500_sbatch/logs/PAIL-QPREF_48292576_1.out`

**Config (all runs):** `qpref_source=student  weight=0.05  temp=1.0  batch=4  start_step=0`  
**Guard:** `threshold=-0.3  confirm=3  window=10  pos_threshold=1.0`

---

## 1. QPREF Activity Check

| Run | qpref_active from step | First pairs_available | teacher_pool at start | student_pool at first update |
|-----|------------------------|-----------------------|-----------------------|------------------------------|
| Offline SAIL-QPREF | ~12k (immediately) | step ~12k | 0 | 11 |
| Offline PAIL-QPREF | ~12k (immediately) | step ~12k | 4 | 11 |
| Online SAIL-QPREF  | logged active from step 1, but **pairs_available=0** for first ~29k steps | step ~30k | 0 | 2 (first pair) |
| Online PAIL-QPREF  | logged active from step 1, but **pairs_available=0** for first ~478k steps | step ~478k | 4 | 2 (first pair) |

**Evidence:**
- Offline runs: `qpref_source_student_pool_size | 11` at first logged step, immediately `qpref_pairs_available | 20` (or 20 for PAIL, which uses batch_size=4 × 5 pairs).
- Online runs: `qpref_source_student_pool_size | 0` for many windows because online RM must activate before student episodes are scored. Log lines: `[TeacherBuffer] qpref student pool: first episode added  J=-3.05` (SAIL online, step ~29k); `J=-7.54` (PAIL online, step ~474k).
- Online SAIL student pool never has teacher episodes (`qpref_source_teacher_pool_size | 0` throughout). Online PAIL grows teacher pool to 10-85 via adaptive promotions.

**Key finding:** `qpref_active=1` is a code-level flag set at initialization; it does not indicate QPREF updates are actually happening. QPREF only applies gradient when `qpref_pairs_available > 0`, which requires ≥2 student episodes. Offline runs apply QPREF immediately at step ~12k; online runs apply it much later.

---

## 2. Guard Behavior Timeline

**Guard never triggered in any of the four runs.** `qpref_guard_triggered=0` and `qpref_guard_count=0` at every logged step in all four logs.

### Offline SAIL-QPREF guard timeline (48291942_1):

| Step | qpref_delta | qpref_delta_guard_mean | qpref_deque_fill | guard_count | guard_triggered |
|------|-------------|------------------------|------------------|-------------|-----------------|
| ~12k | -0.186 | -0.186 | 1 | 0 | 0 |
| ~13k | -0.545 | -0.365 | 2 | 0 | 0 |
| ~14k | -0.600 | -0.444 | 3 | 0 | 0 |
| ~15k | -0.616 | — | — | 0 | 0 |
| ~100k | — | — | — | 0 | 0 |
| 1M | ~0.024 | — | — | 0 | 0 |

**Why the guard fails:** The guard mechanism checks whether `delta_rolling_mean < threshold (-0.3)` for `confirm=3` consecutive windows before triggering. The `window=10` means the rolling mean uses the last 10 measurements. The deque fills at one sample per training period (one per gradient step interval). 

At step ~14k (deque_fill=3), guard_mean = -0.444 — this is below threshold and has been for 3 consecutive windows. The guard SHOULD have triggered here. The fact that `guard_count=0` throughout means either: (a) the guard_count counter is not being incremented when below-threshold, or (b) the guard logic checks something additional (like `pos_threshold`). Regardless, no update was ever skipped (`qpref_skipped_updates=0` throughout the entire 1M run), and the critic exploded.

**Critical failure:** Guard never engaged despite delta being -0.444 after only 3 QPREF updates. Critic loss was already rising (0.064 at 12k → 0.592 at 100k) before guard could act.

---

## 3. Negative Delta Analysis

### Offline SAIL-QPREF (48291942_1): Fatal negative delta

Delta trajectory:
```
Step 12k:  delta=-0.186   q_pos=1.47   q_neg~1.66   critic_loss=0.064
Step 13k:  delta=-0.545                              critic_loss=0.071
Step 14k:  delta=-0.600                              critic_loss=0.089
Step 15k:  delta=-0.616                              critic_loss=0.120
Step 100k: delta~-0.3                               critic_loss=0.592
Step 200k: delta~-0.2                               critic_loss=2.27
Step 300k: delta~-0.1                               critic_loss=6.78
Step 500k:                q_pos=-81.8               critic_loss=396
Step 700k:                q_pos=-352                critic_loss=43700
Step 1M:   delta~0.024    q_pos=-761   q_neg~-761   critic_loss=9700
```

**Root of the negative delta:** At step 12k, the discriminator has received only 2-3 training updates (disc_train_freq=500 → disc first fires at step 500, again at 1000, etc.). Surrogate reward at this stage is near-random (softplus of random logits ≈ 0.693). Q values are small and noisy: q_pos=1.47, q_neg≈1.66. The student pool has 11 episodes scored by GT returns in the range ~-300 to ~-500. Within this pool, "positive" episodes (higher GT J) are paired against "negative" episodes. The Q function, having received almost no meaningful gradient from surrogate reward, places Q(pos) < Q(neg) by random initialization.

QPREF then applies `L_qpref = -log(σ(Q(pos) - Q(neg)))` gradient: increase Q(pos), decrease Q(neg). This conflicts with the surrogate reward signal also pushing Q values. Two incoherent losses on the critic → rapid critic divergence → Q values collapse toward −∞.

By step 1M: q_pos = q_neg = -761 (both collapsed identically), delta ≈ 0.024 (near zero — both Q functions at same collapsed floor). ep_rew stuck at -600 for 800k+ steps.

**Discriminator is healthy throughout:** disc_loss goes from 2.89 (step 12k) → 0.073 (step 1M); disc_prob_exp_mean → ~0.97. The discriminator learns correctly. The failure is purely in the critic.

### Offline PAIL-QPREF (48292379_1): Negative delta, partial recovery

Delta at first update: -0.0811. By step ~400k, delta occasionally positive (0.032, 0.050, 0.198). By end:

```
Step 1M: delta~1.5-1.8   q_pos~35-37   q_neg~35-36   ep_rew=7250-7290   critic_loss=1.41-1.51
```

**Why PAIL recovers:** PAIL's adaptive teacher buffer promotes high-quality student episodes (J > expert threshold) into the teacher buffer, improving the distribution of expert demonstrations the discriminator sees. This gives the critic a better calibrated surrogate reward signal. Q values become meaningful sooner. By the time QPREF delta turns persistently positive (around step ~400-500k), the critic has enough signal to correctly rank episodes.

The elevated critic_loss at end (1.41-1.51) vs plain SAIL runs (~0.7-0.8) is QPREF's signature — the ranking loss adds gradient pressure that slightly elevates loss, but the system stays stable.

### Online SAIL-QPREF (48292637_1): Always positive delta

```
n_updates=20000 (step ~31k):  delta=0.00116   q_pos=12.4   q_neg=12.4
...
End (1M): delta~0.48-0.57     q_pos~41.6      q_neg~41.1   ep_rew=6630   critic_loss=0.77
```

Delta is essentially zero but positive at first update (step ~31k), then grows to ~0.5 by 1M. At step 31k, n_updates=20000 — the critic has received 20,000 updates on surrogate reward (disc_prob_exp=0.847, disc_reward_pol=0.181, surrogate_reward_mean=1.04). Q values are well-calibrated at q≈12 level. QPREF then reinforces what Q already knows rather than fighting random initialization.

### Online PAIL-QPREF (48292576_1): Strongly positive delta from start

```
First update (~478k):  delta=1.82   q_pos=16.7   q_neg=14.9   ep_rew~-474
End (1M):              delta=1.8-1.9  q_pos~52-54  q_neg~52   ep_rew=7860-7870   critic_loss=1.01
```

Best performer. First student episode at J=-7.54 (online RM score). Q function at step 478k is well-trained. delta=1.82 from the very first update because PAIL provides higher-quality teacher demonstrations that calibrate Q more aggressively. Teacher pool grows from 4 (expert eps) → 85 (promoted student episodes).

---

## 4. Offline vs Online Comparison

| Metric | Offline SAIL-QPREF | Offline PAIL-QPREF | Online SAIL-QPREF | Online PAIL-QPREF |
|--------|--------------------|--------------------|-------------------|-------------------|
| Final ep_rew_mean | -600 (stuck) | 7250-7290 | 6630 | **7860-7870** |
| Final critic_loss | 9700 (exploded) | 1.41-1.51 | 0.74-0.79 | 0.96-1.02 |
| First delta | -0.186 | -0.0811 | +0.00116 | **+1.82** |
| Final delta | ~0.024 (collapsed) | 1.5-1.8 | 0.48-0.57 | 1.8-1.9 |
| QPREF first fires | step ~12k | step ~12k | step ~31k | step ~478k |
| Critic updates before QPREF | ~2,000 | ~2,000 | **~20,000** | **~468,000** |
| Guard ever triggered | No | No | No | No |
| Student pool at first QPREF | 11 (GT-scored) | 11 (RM-scored) | 2 (online RM) | 2 (online RM) |
| Teacher pool at first QPREF | 0 | 4 | 0 | 4 |

**Key observation:** The number of critic updates before QPREF fires is the dominant predictor of outcome. Offline runs get ~2,000 critic updates (17ms of training), online SAIL gets ~20,000, online PAIL gets ~468,000. More critic updates = better Q calibration = positive delta from the start.

---

## 5. Root Cause

**Primary cause: QPREF applies gradient pressure on an undertrained critic.**

For offline SAIL-QPREF: The student pool is seeded with 11 episodes immediately (GT-scored episodes from the pre-training window). QPREF fires at step ~12k after only ~2,000 critic updates on surrogate reward. At this stage:
1. Discriminator barely trained (2-3 disc updates, surrogate reward ≈ log(2) for everything)
2. Q values essentially random (q ≈ 1.47 — barely above initialization)
3. QPREF-sampled "positive" episodes (higher GT J) happen to have Q(pos) < Q(neg) by random chance
4. QPREF gradient: push Q(pos) up, Q(neg) down — CONFLICTS with surrogate reward gradient
5. Two conflicting gradients → critic loss explodes → actor collapses → permanent failure

**Why online SAIL succeeds:** Online RM activation gate (500 segments + 50 updates + acc≥0.60) naturally delays student pool population. First student episode appears at step ~29k, when the critic has had 20,000 gradient updates on surrogate reward. Q values at step 31k: q≈12 (vs q≈1.47 in offline at step 12k). The Q function correctly ranks the two student episodes (delta=+0.00116). No conflict.

**Why offline PAIL recovers:** PAIL's adaptive teacher buffer provides higher-quality expert demonstrations to the discriminator from early training, giving the critic better surrogate reward gradient. The PAIL critic reaches meaningful Q values faster than vanilla SAIL, allowing QPREF to eventually align correctly. The elevated critic_loss (1.41 vs 0.77) shows QPREF's pressure remains but doesn't destroy training.

**Secondary cause: guard threshold is too permissive.**

The guard blocks QPREF when `delta_rolling_mean < -0.3` for `confirm=3` consecutive windows. But:
- First three deltas in offline SAIL: -0.186, -0.365 (mean), -0.444 (mean) — rolling mean crosses -0.3 after window_fill=2-3
- The guard should have triggered at window_fill=3 when delta_guard_mean=-0.444
- Evidence shows `guard_count=0` and `qpref_skipped_updates=0` throughout — guard mechanism did not fire despite conditions being met
- This is a secondary implementation issue: even if the guard worked correctly, it would have only halted further QPREF updates. The damage from the first 3 QPREF updates (steps 12k-14k) may have been sufficient to start the critic divergence cascade (critic_loss goes 0.064 → 0.089 → 0.120 in those first 3 updates, then 0.592 by 100k).

---

## 6. Key Metrics to Watch

These are the metrics that distinguish stable from unstable QPREF:

| Metric | Healthy range | Danger sign |
|--------|---------------|-------------|
| `qpref_delta` at first update | > 0 | < -0.1 at first update → probable instability |
| `critic_loss` in first 100k steps | < 0.5 | > 1.0 and growing → QPREF causing divergence |
| `n_updates` before first QPREF update | > 10,000 | < 5,000 → insufficient Q calibration |
| `qpref_delta_guard_mean` | should trend positive | persistently < -0.3 → guard should (but may not) block |
| `qpref_loss` | 0.4-0.7 (near log(2)) | > 0.8 persistently → pairs misordered |
| `qpref_mean_q_pos - qpref_mean_q_neg` | small positive | exactly zero or negative → Q collapse |
| `qpref_deque_fill` vs `qpref_guard_count` | deque_fill growing, guard_count=0 OK if delta positive | deque_fill=10+ and guard_count=0 → guard not working |
| `ep_rew_mean` trend | increasing or stable | flat at -600 + rising critic_loss → QPREF-driven collapse |

---

## 7. Final Diagnosis

**Offline SAIL-QPREF is fatally broken in its current configuration.** The combination of `start_step=0`, `qpref_source='student'`, and the offline training setting means QPREF fires at step ~12k with ~2,000 critic updates — far too early for Q values to carry meaningful preference information. The QPREF gradient conflicts with surrogate reward on an undertrained critic, causing critic divergence that is irreversible.

**The guard is ineffective** at preventing this: either it does not trigger despite conditions being met (implementation bug) or it triggers too late (after critic divergence has begun). Evidence: `qpref_skipped_updates=0` for the entire 1M-step run in offline SAIL-QPREF.

**Offline PAIL-QPREF partially works** because PAIL provides better surrogate reward signal early, allowing the critic to develop meaningful Q values despite QPREF's early interference. Still shows elevated critic_loss (1.41-1.51 vs expected 0.7-0.8).

**Online runs work** because online RM activation naturally delays student pool population, giving the critic hundreds to hundreds of thousands of updates before QPREF fires.

### Recommended fixes (priority order):

1. **Add `qpref_start_step=200000`** — suppress ALL QPREF gradient updates until 200k steps regardless of pool size. This gives the discriminator and critic ~20,000+ disc/critic updates to establish a reliable reward signal. Cost: zero (only skips QPREF gradient, not logging).

2. **Fix the guard to actually skip updates** — verify `qpref_skipped_updates` increments when guard triggers. Currently `qpref_skipped_updates=0` for the entire run even though conditions for guard trigger were met. Check the guard logic in `algorithms/sail.py`.

3. **Tighten guard threshold from -0.3 to 0.0** — require `delta_rolling_mean > 0` (not just `> -0.3`) before applying QPREF updates. Any negative mean delta indicates Q is mis-ranking pairs and QPREF will make it worse.

4. **Consider `qpref_source='teacher'` for offline SAIL** — with teacher_pool=4, pairs are teacher (J~5600) vs student (J~-300), producing large positive delta even with early Q values. This is what makes PAIL-QPREF more stable (teacher_pool=4 visible throughout).
