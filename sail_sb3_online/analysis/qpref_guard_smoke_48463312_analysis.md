# QPREF Guard Smoke Test — Detailed Analysis
**Log:** `sail_sb3/sail_sb3/logs/HC_QPREF_Guard_smoke_48463312_1.out`  
**Job:** 48463312, seed 1  
**Config:**  
```
qpref_source=student  weight=0.05  grad_interval=50  start_step=0
guard: threshold=-0.3  confirm=3  window=10  pos_threshold=1.0
adaptive=gt  lfd_mixing=True  entcoeff=0.05  total_timesteps=1M
teacher_pool: 4 expert episodes  pref_pool RM scores: mean=1353.5 min=1333.1 max=1374.9
```

---

## 1. Early-Phase QPREF Behavior

QPREF fires at step **3,000** (the very first training log point) with only **2 student episodes** in the pool. The Q function at this point has approximately 1,000 critic updates and Q values of ~1.4-1.6. This is far too early.

**The early phase has four sub-phases:**

**Sub-phase A (steps 3k–6k): Tiny pool, negligible signal**
- Pool size: 2 → 4 student episodes
- Student episodes all have GT J around –300 to –450 (early training quality)
- J-spread within those 2-4 episodes: tiny (all poor, all similar quality)
- Q values: 1.39–9.26 (barely trained)
- Delta oscillates: −0.114, −0.085, **+0.256** (step 5k, pool=4), **+0.002** (step 6k)
- The single positive reading at step 5k is a fluke: q_pos=7.19 vs q_neg=6.93 (a 0.26 difference out of noise)
- **QPREF is producing only noise in this phase — no meaningful gradient**

**Sub-phase B (steps 7k–30k): Persistent negative delta, active Q degradation**
- Pool size: 6 → 29 student episodes
- Delta turns negative from step 7k (−0.18) and **stays negative continuously for ~45k steps**
- q_pos < q_neg in every recorded measurement:
  - step 7k: q_pos=10.3, q_neg=10.5
  - step 15k: q_pos=13.8, q_neg=14.4
  - step 20k: q_pos=12.7, q_neg=13.4
  - step 27k: q_pos=9.78, q_neg=11.1
- Q1_mean (bulk critic): **rises from 1.61 → 22.6 (steps 3k–15k)** from surrogate reward learning
- Then **drops from 22.6 → 12.9 (steps 15k–45k)** — a 43% decline in average Q values
- This drop is QPREF-driven: with persistent negative delta, QPREF gradient pushes Q(pos) down (or Q(neg) up), actively fighting the surrogate reward signal on the critic
- Critic loss rises from 0.046 (step 3k) to 0.455 (step 50k) — not catastrophic, but clearly elevated by the conflict
- ep_rew declines monotonically: −322 → −431 → −512 → −542 → −557 → −566 → −572 → −576 → −580 → −582 → −593

**Sub-phase C (steps 30k–60k): Gradual recovery**
- Pool size: 29 → 59 episodes
- Delta oscillates with higher amplitude, begins having positive readings
- step 40k: delta=−0.217, q1=14.9 (still falling)
- step 50k: delta=+0.358, q1=25.5 (Q rebounds sharply)
- step 55k: delta=−0.475, q1=40.8
- step 60k: delta=+2.16, q1=53.5
- Discriminator is now well-trained: disc_prob_exp=0.93, disc_loss=0.163 at step 64k
- ep_rew slowly improves: −593 → −580 → −573

**The recovery is driven by the base SAIL/LfD algorithm overcoming QPREF, not by QPREF helping.**

---

## 2. Delta Timeline

Complete timeline from start to guard fire:

| Step | ep_rew | delta | delta_guard_mean | delta_rolling | q1 | critic_loss | peak_gm | pool |
|------|--------|-------|-----------------|---------------|----|-------------|---------|------|
| 3k   | −322   | **−0.114** | −0.114 | −0.114 | 1.61 | 0.046 | −0.114 | 2 |
| 4k   | −390   | −0.085 | −0.099 | −0.099 | 3.94 | 0.046 | −0.099 | 3 |
| 5k   | −431   | **+0.256** | +0.019 | +0.019 | 7.45 | 0.092 | +0.019 | 4 |
| 6k   | −457   | +0.002 | +0.015 | +0.015 | 10.2 | 0.142 | +0.019 | 5 |
| 7k   | −477   | −0.180 | −0.024 | −0.024 | 12.1 | 0.192 | 0.019 | 6 |
| 10k  | −512   | −0.370 | −0.167 | −0.167 | 15.5 | 0.185 | 0.019 | 9 |
| 15k  | −542   | −0.601 | −0.465 | −0.353 | 22.6 | 0.276 | 0.019 | 14 |
| 20k  | −557   | −0.759 | −0.780 | −0.507 | 22.0 | 0.131 | 0.019 | 19 |
| 25k  | −566   | −1.19 | −0.889 | −0.586 | 21.1 | 0.282 | 0.019 | 24 |
| 30k  | −572   | −0.762 | −0.975 | −0.674 | 19.2 | 0.265 | 0.019 | 29 |
| 35k  | −576   | −0.670 | −0.906 | −0.683 | 17.4 | 0.200 | 0.019 | 34 |
| 40k  | −580   | −0.217 | −0.583 | −0.650 | 14.9 | 0.186 | 0.019 | 39 |
| 45k  | −582   | −0.314 | −0.372 | −0.611 | 12.9 | 0.177 | 0.019 | 44 |
| 50k  | −593   | +0.358 | −0.028 | −0.521 | 25.5 | 0.455 | 0.019 | 49 |
| 55k  | −580   | −0.475 | +0.048 | −0.517 | 40.8 | 0.874 | 0.090 | 54 |
| **60k** | −573 | **+2.16** | +0.337 | −0.406 | 53.5 | 1.36 | 0.337 | 59 |
| 100k | −367   | +3.52 | +1.23 | +2.20 | 75.4 | 6.81 | 4.70 | 99 |
| 150k | +728   | +8.71 | +6.54 | +4.68 | 58.7 | 11.2 | 6.54 | 149 |
| 200k | +2630  | +11.2 | +9.18 | +8.15 | 37.8 | 5.61 | 9.43 | 199 |
| 250k | +4570  | +14.0 | +12.6 | +11.2 | 32.7 | 3.99 | 12.6 | 249 |
| 300k | +5940  | +11.9 | +12.3 | +12.3 | 30.0 | 2.38 | 12.9 | 299 |
| 350k | +6480  | +12.2 | +11.3 | +12.1 | 22.7 | 1.47 | 13.3 | 349 |
| 400k | +6520  | +12.9 | +12.8 | +11.8 | 29.1 | 1.32 | 13.3 | 399 |
| 450k | +6330  | +15.1 | +14.6 | +13.5 | 38.3 | 1.29 | 14.8 | 449 |
| 500k | +6420  | +14.0 | +14.2 | +14.2 | 41.2 | 1.18 | 14.8 | 499 |
| 550k | +6580  | +15.1 | +14.7 | +14.5 | 44.5 | 1.31 | 15.5 | **500 (cap)** |
| 600k | +6630  | +9.87 | +9.29 | +11.4 | 46.2 | 1.31 | 15.5 | 500 |
| 650k | +6750  | +5.10 | +4.02 | +6.29 | 49.8 | 1.18 | 15.5 | 500 |
| 700k | +6700  | +0.413 | +1.03 | +1.94 | 51.5 | 1.15 | 15.5 | 500 |
| 750k | +6660  | −0.794 | −0.256 | +0.259 | 50.6 | 1.17 | 15.5 | 500 |
| **757k** | — | **Guard fires** | −0.400 | — | — | — | 15.5 | 500 |
| 800k | +6530  | — | −0.400 | +0.136 | 51.0 | 1.10 | 15.5 | 500 |
| 1M   | +6610  | — | −0.400 | +0.136 | 47.4 | 0.993 | 15.5 | 500 |

**Three distinct phases:**
1. **Harmful early (3k–60k):** delta mostly negative, Q values artificially depressed, ep_rew declining
2. **Strongly helpful (60k–550k):** delta +2 → +15, policy goes from −573 to 6580
3. **Degrading late (550k–757k):** delta 15 → 0 → negative as pool saturates, guard fires at 757k

---

## 3. Guard Behavior

### What the guard does and when it fires

**Guard configuration:**
- `threshold = −0.3` (guard triggers if `delta_guard_mean < −0.3`)
- `confirm = 3` (must be below threshold for 3 consecutive measurement windows)
- `window = 10` (use most recent 10 deque entries, not the full 50)
- `pos_threshold = 1.0` (guard is INELIGIBLE to trigger until `peak_guard_mean ≥ 1.0`)

**What the guard did in this run:**

The `pos_threshold=1.0` gate **completely blocks the guard for the first 60k steps.** Evidence:
- `peak_guard_mean = 0.019` from step 3k all the way through step 59k (the maximum guard_mean ever seen stays at 0.019 — the brief flicker at step 5k)
- `peak_guard_mean` first exceeds 1.0 at step **~64k** (when delta=3.47, guard_m=1.22)
- Before that: even when `delta_guard_mean = −0.975` (step 30k), guard cannot trigger because `peak_guard_mean < 1.0`

**Timeline of guard eligibility:**
- Steps 3k–60k: **Guard ineligible** (peak_guard_mean < 1.0, despite delta_guard_mean plunging to −0.975)
- Steps 60k–550k: Guard eligible (peak_guard_mean growing 0.337 → 1.22 → 15.5)
- Steps 550k–757k: Guard eligible; delta slowly turns negative again
- **Step 757k: Guard fires.** Log: `[QPREF GUARD] Triggered at step 757000: guard mean delta -0.3997 < -0.3 for 3 consecutive intervals (recent window=10/10 entries, full deque=50, peak_guard_mean=15.5462). QPREF permanently disabled.`

**After guard fires:**
- `qpref_active = 0` — all subsequent QPREF gradient attempts skipped
- `qpref_skipped_updates` increments by 20 per window (grad_interval=50, 1000 steps/window → 1000/50=20 updates/window)
- Final value at 1M: `qpref_skipped_updates = 4840` ≈ (1M − 757k)/50 = 4860 expected. ✓ Correct.
- ep_rew after guard: 6630 → 6530 → 6560 → 6500 → 6350 → 6610 (flat, no major change)

**Is the guard helping?**

The guard is **technically correct but strategically limited:**
1. **It does NOT protect early training** — the most harmful QPREF period (3k–60k) is completely invisible to the guard because `pos_threshold=1.0` keeps it ineligible. During those 60k harmful steps: `guard_count=0`, `guard_triggered=0`, `qpref_skipped_updates=0`.
2. **It fires correctly at 757k** to stop the late-phase degradation (delta went from +15 at 550k to −0.4 at 757k).
3. **The policy did not recover after guard fire:** ep_rew was 6660 at 750k and 6610 at 1M — essentially flat. The guard stopped further harm but did not rescue performance.

**Bottom line:** The guard works in the late phase but offers zero protection during the early phase where the damage most matters.

---

## 4. Why Performance is Weak

### The two actual failure modes

**Failure 1 (early, 3k–60k): Wrong ordering on an undertrained critic with homogeneous episodes**

The student pool fills from the bottom up: the first 50+ episodes are all from a policy with ep_rew −300 to −600. Their GT J-scores span a narrow range (~−300 to ~−600 total return, all poor). Within this band, J-assignment to "positive" vs "negative" is near-random from the critic's perspective.

At step 3k, the critic has ~1,000 updates. Q values are 1.4–1.6 — barely above random initialization. The QPREF loss tells it: "this episode (J=−350) should have higher Q than that episode (J=−430)". But the 80-point J difference over a 1000-step episode is nearly indistinguishable given the noise in the surrogate reward signal (discriminator has had only 6 updates by step 3k). The Q function places Q(pos) < Q(neg) by chance, QPREF pushes Q(pos) up and Q(neg) down — but since the discriminator has not yet established stable reward structure, this push conflicts with the Bellman-target gradient.

Evidence from log: q1_mean rises from 1.61 to 22.6 (steps 3k–15k, surrogate reward learning), then **falls from 22.6 to 12.9 (steps 15k–45k)** — a 43% drop directly caused by QPREF pulling Q values in conflicting directions.

**Failure 2 (late, 550k–757k): Pool saturation and collapsed J-spread**

At step 550k, the student pool caps at 500 episodes. By this point, the policy is at ep_rew ~6500. As training continues, new episodes (ep_rew ~6000–7000) push out older ones (ep_rew ~−400 to +2000) via FIFO rotation. By step 600k+, the pool is dominated by high-quality episodes with similar GT J-scores. 

Pairs sampled from 500 similar-quality episodes have tiny J-differences → the margin `J_pos − J_neg` is small and noisy → Q ordering is random relative to this noise → delta begins declining: 15.1 → 9.87 → 5.10 → 0.413 → −0.794.

Evidence: `qpref_delta_std` at step 550k is still reasonable. But the mean delta is declining smoothly and monotonically from 550k onward, consistent with pool homogeneity rather than random noise.

### No catastrophic critic explosion

Unlike the previous SAIL-QPREF runs, critic_loss here does NOT explode unrecoverably. It peaks at 11.2 at step 150k but then declines steadily (5.61 → 3.99 → 2.38 → 1.47 → 1.32 → 1.18 → 1.31). This is because:
1. LfD mixing (teacher expert transitions) stabilizes the replay buffer with high-quality data
2. entcoeff=0.05 keeps the discriminator well-calibrated
3. QPREF weight=0.05 is low enough that the harmful gradient doesn't dominate the total loss

The QPREF harm is real but sublethal — it delays and slightly suppresses learning rather than killing it.

---

## 5. Why QPREF is Not Helping Early

Three compounding reasons:

### Reason 1: Pool diversity is near zero at start

At step 3k (first QPREF update), the pool has **2 student episodes** with J scores drawn from GT returns at ep_rew ~−322. The two episodes differ by maybe 50-100 points out of a scale of ~9000 (expert). That ~0.5–1% J-spread is indistinguishable from surrogate reward noise, especially when the discriminator has had <10 training updates.

The J-labels are correct in principle but the signal-to-noise is catastrophically low. QPREF requires meaningful J-spread to produce consistent delta. With only 2 similar-quality episodes, every pair is essentially a coin flip.

### Reason 2: Q function is not calibrated when QPREF first fires

At step 3k: q1_mean = 1.61. The critic has had exactly 1,000 gradient updates on surrogate reward (train_freq=1000, grad_steps=1000 → first update at step 1k, but since the first disc update is at step 500, surrogate reward at step 1k is still near-random at ~log(2)=0.693).

A Q value of 1.61 means the critic knows almost nothing. It cannot meaningfully rank episodes. The QPREF loss imposes an ordering constraint on a random function, creating conflicting gradient with Bellman targets.

At step 60k (when QPREF first becomes genuinely useful): q1_mean = 53.5, disc_prob_exp = 0.93, disc_loss = 0.163. The Q function at this point reflects a meaningful and consistent reward structure. Same QPREF loss → large positive delta (+2.16) → consistent, helpful gradient.

**The difference between harmful and helpful QPREF is purely timing relative to Q function calibration.**

### Reason 3: `qpref_source='student'` means no anchor signal in early training

The teacher pool has 4 expert episodes with J ~1333–1374 (RM scores). The student pool starts with J ~−300 to −600 (GT returns). A teacher-vs-student pair would have J-spread of ~1700+ points — a crystal-clear ranking signal even with only 2-3 updates on the Q function.

But `qpref_source='student'` means QPREF only samples from the student pool. With student J spread of ~50–100 points among similarly-poor episodes, the ranking signal is useless in early training.

The teacher pool is there, the J-labels are loaded, but QPREF never uses them.

---

## 6. Best Improvements to Make QPREF Useful from the Beginning

Ranked by estimated impact:

### Rank 1: Pool-size warmup gate (highest impact, zero cost)

**Change:** Do not apply any QPREF gradient until `qpref_source_student_pool_size ≥ N_min` (suggest 50) AND `qpref_delta_guard_mean > 0` for at least one window.

**Why it helps:** At step 50k (pool=49), delta turns positive (+0.358) for the first time since step 5k. The discriminator is well-trained, Q values are meaningful, pool has reasonable diversity. If QPREF had been dormant from 3k to 50k and activated at step 50k, the harmful Q degradation phase (Q drops from 22.6 to 12.9, steps 15k–45k) never happens. The base SAIL algorithm would have built a better Q foundation during that period.

The guard's `pos_threshold=1.0` is trying to do this but via a different mechanism. A direct pool-size gate is simpler, more predictable, and fires at the right moment.

**Implementation:** In `_update_discriminator` or `train()`, add:
```python
if self.qpref and self.qpref_source == 'student':
    if len(student_pool) < self.qpref_pool_min_size:
        return  # skip QPREF updates
```

### Rank 2: Mixed pair source in early training (very high impact)

**Change:** During the warmup phase (or always), sample QPREF pairs as `(teacher_episode, student_episode)` — a cross-pool pair — rather than only `(student_i, student_j)`.

**Why it helps:** Teacher episodes have J ~1333–1374 (RM scores). The earliest student episodes have J ~−300 to −600 (GT). The pair `(teacher_ep, student_ep)` has J-spread ~1700, which produces delta of order 10–20 even with a weakly-trained Q function. This gives the Q function a strong, clear preference signal from step 3k: "teacher episodes should have much higher value than random student episodes." This is the same signal PAIL-QPREF gets from its teacher_pool.

This does not change the QPREF loss function — only the pair source. Teacher J values are already available (`teacher_pool_size=4` appears in every log line throughout the run). The pairing logic just never uses them.

**Implementation:** Add a `qpref_cross_pool` flag. When enabled, for each QPREF batch, sample half pairs as (teacher, student) and half as (student, student). Or, in early training only (pool_size < 50), always use cross-pool pairs.

### Rank 3: Positive-first gate design (medium impact, replaces current pos_threshold)

**Change:** Replace `pos_threshold=1.0` on `peak_guard_mean` with a minimum-positive-count gate: "QPREF updates are enabled once delta has been positive for ≥ K consecutive windows (e.g., K=3)." Disable immediately when delta turns persistently negative (current confirm=3 logic).

**Why it helps:** The current `pos_threshold=1.0` waits for peak_guard_mean to exceed 1.0, which only happens after the policy is already performing well (~60k steps). The gate does not adapt to the learning speed. A "K consecutive positive readings" gate would fire as soon as QPREF shows its first consistent positive signal — which in this run would have been around step 50k (the first genuine positive after the noisy early phase), rather than waiting until Q has naturally recovered.

### Rank 4: Entropy-based pair filtering (medium impact)

**Change:** Only submit a pair for QPREF gradient if `|J_pos − J_neg| > margin_threshold` (e.g., `|ΔJ| > 100`). Pairs with tiny J-difference are noise — they contribute large variance and small signal to the loss.

**Why it helps:** In early training, most pairs have |ΔJ| < 50 (similar-quality poor episodes). Filtering these out suppresses the harmful gradient while allowing any high-signal pairs to contribute. In late training (pool saturation), same filter would suppress most pairs as J-spread collapses — effectively auto-disabling QPREF when signal is gone, without relying on the guard.

**Evidence from log:** At step 5k (the one positive reading), q_pos=7.19 vs q_neg=6.93. The J-difference between those two student episodes was probably small (~50 points). At step 750k (delta going negative), same issue — pool full of similar-quality episodes. A |ΔJ| > 100 filter would have silenced QPREF in both phases.

### Rank 5: Adaptive pool with quality diversity (medium impact)

**Change:** Instead of FIFO capping at 500, maintain a pool that spans the full quality range. Keep the K best, K worst, and K median episodes by J-score. Evict from the dominant quality bucket when pool is full.

**Why it helps:** The late-phase failure (550k–757k) is directly caused by pool saturation with same-quality episodes. A diversity-aware pool would maintain J-spread by always keeping episodes from different performance levels, ensuring QPREF always has meaningful contrast pairs even when the policy has converged.

### Rank 6: Increase QPREF weight after confirmed positive phase (lower priority)

**Change:** Start at weight=0.01 (currently 0.05) during the warmup/uncertain phase, then ramp to weight=0.05 (or 0.1) once delta has been consistently positive for K steps.

**Why it helps:** weight=0.05 is substantial enough to cause the 43% Q degradation in early training. Starting lower (0.01) would reduce damage from early mis-ordered pairs. Once QPREF is clearly helpful (delta consistently positive), increasing weight would amplify the beneficial effect.

### Rank 7: Bellman-target integration instead of a separate loss (architectural, high impact if done)

**Change:** Instead of adding `L_qpref` as a separate loss term to the critic update, integrate the preference constraint into the Bellman target directly: for a positive episode, add a small preference bonus to the target Q; for a negative episode, subtract it. This avoids the gradient conflict between `L_qpref` and `L_bellman`.

**Why it matters:** The fundamental conflict is that QPREF applies a ranking gradient that may point in the opposite direction to the Bellman-target gradient. By modifying the target instead of adding a separate loss, both objectives share the same gradient direction. The effect is a modified reward signal: `r̂ = r_surrogate + λ * (J_normalized)` for episodes in the QPREF pool.

This is architecturally larger than the other fixes but would make QPREF robust to the Q-calibration issue since it acts through the reward shaping layer, not the loss function.

---

## 7. Final Diagnosis

**What happened in this run, concisely:**

QPREF fired immediately at step 3k with a 2-episode student pool. The J-spread within those early poor-quality episodes is negligible, and the Q function is random. For 45k+ steps, QPREF applied gradient that depressed Q values (q1_mean dropped 43%, from 22.6 to 12.9) and slowed ep_rew progress. The base SAIL+LfD algorithm eventually overcame QPREF's interference around step 50k–60k when the discriminator matured and the pool grew to ~50 episodes with better diversity.

Once QPREF became genuinely useful (60k–550k), it was **strongly helpful**: delta grew to +15, and the policy reached ep_rew 6750 by step 650k. This is the "correct" QPREF behavior and confirms the algorithm works when conditions are right.

The pool capped at 500 episodes by step 550k, which gradually filled with similar high-quality episodes. J-spread collapsed, delta declined from +15 to −0.794, and the guard correctly fired at step 757k. After the guard, ep_rew stabilized at ~6500–6610 with no further change.

**The guard worked correctly in its intended scope** (late-phase protection), but its `pos_threshold=1.0` design made it completely invisible during the early harmful phase. The guard is not broken — its design scope is wrong.

**Root causes, ranked:**
1. `start_step=0` + tiny pool = QPREF fires before Q function or pool have enough signal (primary cause of early harm)
2. `qpref_source='student'` with no teacher-anchor = no high-contrast pairs in early training (amplifies cause 1)
3. FIFO pool capping = late-phase J-spread collapse (cause of late degradation)
4. Guard `pos_threshold=1.0` = guard ineligible during the harmful early phase (design scope issue, not implementation bug)

**Expected gains from Rank 1 + Rank 2 fixes:**
- Eliminating the 45k harmful steps would let Q values remain at their natural peak (~22) instead of falling to 12.9
- Earlier convergence: policy would likely reach ep_rew +700 by step 100k (vs step 150k now, a 30% speedup)
- Better final performance: fewer harmful updates during the critical 10k–60k window means the Q function enters the productive phase with a stronger foundation

**Proof of concept in this run:** Between steps 45k–60k, when QPREF delta was near zero (not harmful, not helpful), the policy improved from −582 to −573. At step 60k when QPREF turned strongly positive, it then rocketed to +728 by step 150k. If QPREF had been dormant until step 50k (pool=49, delta first genuinely positive), that same ramp would have started 50k steps earlier, from a higher baseline.
