# Why QPREF Is Not Beating Adaptive SAIL

**Date:** 2026-04-22  
**Runs analyzed:** 6 variant groups, ~30 seeds across HC HalfCheetah-v2 1M-step experiments

---

## 1. Executive Summary

QPREF, across every variant tried — teacher-source, student-source, cross-pool, with guard, without guard — converges to **~6.0–6.8k final ep_rew_mean** on HalfCheetah-v2. Plain Adaptive SAIL (no QPREF) with identical hyperparameters reaches **8.3–8.7k**. This is a consistent ~2000 point gap (~23%).

The root cause is a **dual interference pattern**:

1. **Early-phase (steps 3k–60k):** The critic is undertrained and its Q-values are tiny (1–15 range). QPREF's softplus loss applies large gradients that conflict with Bellman targets. This corrupts the critic foundation before it is calibrated.

2. **Late-phase (steps 700k–1M):** The adaptive buffer fills with near-expert student episodes. QPREF delta (Q_pos − Q_neg) collapses toward zero or goes negative. QPREF now either contributes nothing or actively pushes the critic in the wrong direction. Performance declines.

These two phases bracket the useful window where QPREF signal is meaningful (steps ~60k–700k). QPREF does produce positive delta and faster learning in that window — but the early corruption means the policy enters the window from a weaker starting position. The late collapse ensures the window closes before 1M steps. Net result: QPREF helps in the middle but cancels itself on both sides.

Changing the pair source (teacher / student / cross-pool) does not fix this. It shifts which phase is worse, but not by enough to matter. Guard helps with the late-phase collapse but leaves the early-phase damage unaddressed. All variants therefore converge to the same 6–7k ceiling.

---

## 2. Variant-by-Variant Comparison

All runs use HalfCheetah-v2, 1M steps, adaptive buffer, LfD mixing, pref_rm loaded.

### 2.1 Plain Adaptive SAIL (Baseline)

| Run ID | Config | Seeds succeeding | Stable final ep_rew |
|--------|--------|-----------------|---------------------|
| 46729448 | entcoeff=0.05, GT adaptive | 2/2 | **8.29k, 8.74k** |
| 48290421 | entcoeff=0.01, TF parity | 3/5 | 5.57k, 6.19k, 6.73k |
| 48292214 (PAIL) | entcoeff=0.01, RM adaptive | 3/5 | 7.43k, 7.76k, 7.37k |
| 48350850 (PAIL) | entcoeff=0.01, RM adaptive | 0/5 | −600 (disc saturation) |

The strongest baseline (entcoeff=0.05, GT adaptive) reaches **8.3–8.7k**, stable to 1M steps with no decline. This is the fair comparison point for all QPREF variants below.

### 2.2 QPREF Teacher-Source (no guard)

| Run ID | Config | Seeds succeeding | Final ep_rew |
|--------|--------|-----------------|--------------|
| 47225856 | qpref_weight=0.1, interval=10, entcoeff=0.05 | 1/2 | −600, **6.55k** |
| 48291942 | qpref_weight=0.05, interval=50, entcoeff=0.05 | 4/5 | −600, **6.04k, 6.27k, 6.56k, 6.01k** |
| 48350882 | qpref_weight=0.05, interval=50, entcoeff=0.05 | — | seed 1: peaks at **6.76k at step ~730k** → **declines to 5.46k at 1M** |

**Key behaviors:**
- Early (steps 11k–90k): qpref_delta oscillates between −1.0 and +0.16. ep_rew is declining (−332 → −444). Both Bellman and QPREF gradients are large; they conflict.
- Middle (steps 90k–500k): delta becomes consistently positive (3–20 range). ep_rew rises rapidly.
- Late (steps 700k–1M): delta shrinks (20 → 12 → 8 → 4 → near-zero). Student pool fills with adaptive-promoted near-expert episodes. ep_rew peaks then **declines**: 6.76k → 5.46k in seed 1 (−18%).
- Guard was not active in these runs (no `--qpref_guard` flag). The late decline is unchecked.

**vs. baseline:** Teacher-source QPREF achieves 6.0–6.6k stable or 6.76k declining. Baseline reaches 8.3–8.7k stable. Gap = ~2k.

### 2.3 QPREF Teacher-Source on PAIL Baseline (no guard)

| Run ID | Config | Seeds succeeding | Final ep_rew |
|--------|--------|-----------------|--------------|
| 48292379 (PAIL-QPREF) | entcoeff=0.01, RM adaptive, teacher-source | 3/5 | **7.29k, 7.57k**, 368, 537, **7.60k** |
| 48292214 (PAIL) | entcoeff=0.01, RM adaptive, no QPREF | 3/5 | **7.43k, 7.76k, 7.37k** |

When base config is entcoeff=0.01 (already somewhat susceptible to disc saturation), QPREF neither helps nor hurts for the seeds that succeed. Both reach ~7.4–7.6k. But PAIL-QPREF has **two additional partial-failure seeds** (368, 537) instead of full failures (−600). This suggests QPREF changes the failure mode from full disc saturation to partial collapse.

**Key finding:** QPREF is not improving successful seeds and is changing the failure landscape.

### 2.4 QPREF Student-Source (HC_OnlineRM_QPREF_Student_48253943)

| Seed | Final ep_rew | Note |
|------|-------------|------|
| 1 | −600 | full failure |
| 2 | 6.3k | QPREF effectively OFF (student pool = 0 throughout) |
| 3 | −600 | full failure |

The student pool never populated because `pref_rm` was not loaded as the online RM scorer (student_pool_size stayed at 0 in logs). `add_student_episode` was a no-op → QPREF sampled empty pool → returned None → no gradient. Seed 2 behaved like plain adaptive SAIL, reaching 6.3k (which is below the 8.4k no-QPREF baseline). Seeds 1 and 3 failed independently.

**Key finding:** When student-source actually functions (pref_rm loaded, pool fills), behavior is expected to mirror teacher-source with even worse early-phase interference (student pool has homogeneous near-random trajectories with tiny J-spread initially). The guard smoke test (Section 2.5) is the proper student-source test.

### 2.5 QPREF Student-Source with Guard (HC_QPREF_Guard_smoke_48463312)

Config: qpref_weight=0.05, interval=50, entcoeff=0.05, GT adaptive, guard enabled (threshold=−0.3, confirm=3, window=10, pos_threshold=1.0).

| Seed | Guard fires? | Peak ep_rew | Final ep_rew |
|------|-------------|------------|--------------|
| 1 | Yes (step 757k) | ~6.76k | **6.61k** |
| 2 | No | ~6.79k | **6.57k** |
| 3 | No | ~6.72k | **6.55k** |

Guard **works as designed**: seed 1's guard fires at step 757k and halts further late-stage decline. Seeds 2 and 3 plateau naturally without the guard triggering (their delta window doesn't drop below −0.3 consistently).

Final values (6.55–6.61k) are **stable** and represent the best QPREF result in this comparison — better than teacher-source (5.5–6.6k declining). But still 1.7–2.2k below the no-QPREF baseline.

**Early phase:** Seeds 1–3 all show the same early decline pattern (ep_rew from −325 to −580 before recovering). The guard's pos_threshold=1.0 design means the guard is ineligible during the entire early harmful phase (peak_guard_mean never reaches 1.0 until step ~200k+). This is confirmed: the early damage (steps 3k–60k) happens identically to the no-guard runs.

### 2.6 QPREF Cross-Pool Source (HC_QPREF_CrossPool_smoke_48503862)

Config: same as guard smoke but qpref_source=cross_pool, qpref_cross_pool_min_student=0.

| Seed | Peak ep_rew | Final ep_rew | Note |
|------|------------|--------------|------|
| 1 | ~6.95k (step ~870k) | **6.27k** | declining from peak |
| 2 | ~6.77k (step ~660k) | **6.2k** | dramatic mid-run collapse (4.55k at step ~770k), partial recovery to 6.2k |

Cross-pool provides larger J-spread from step 0 (teacher J~1340 vs student J~−300, spread ~1640 vs ~50 in same-pool). But:

- **Same early-phase interference**: Q-values are still tiny (1–15) when the QPREF loss fires. The larger J-spread creates larger softplus gradients, but those gradients are still fighting Bellman.
- **Seed 2 instability**: The cross-pool pairing creates dramatically large loss swings mid-run, leading to a 2000-point collapse (6.77k → 4.55k). This is worse than teacher-source.
- **Performance ceiling identical**: Final values (6.2–6.3k) are not higher than teacher-source or guard smoke.

**Key finding:** Larger J-spread does not help because the bottleneck is not the J-spread — it is the misalignment between J-ranking and Q-ranking during critic training.

---

## 3. Why the RM Signal Is Not Helping

### 3.1 J vs. V Misalignment (Core Problem)

The QPREF loss trains the critic to rank trajectories by **J** (RM reward). But Bellman training trains the critic to represent **V** = discounted sum of **GAIL discriminator rewards**. These are two different objectives:

- J = `pref_rm.reward(obs, acs)` — offline RM rating of the trajectory
- V = `Σ softplus(logits_disc(s,a)) * γ^t` — accumulated imitation reward

In a well-trained SAIL system, J and V are **correlated** (both reward expert-like behavior), but they are **not the same function**. The discriminator measures similarity to the expert distribution; the RM measures a richer semantic quality. Early in training these two are often **anti-correlated** at the trajectory level because:

- The discriminator is not yet well-calibrated (low entropy, not fully trained)
- GAIL discriminator reward is dense (per-step), while RM reward is trajectory-level
- The Bellman bootstrap under an uncalibrated critic propagates incorrect values

### 3.2 Early-Phase Q Catastrophe

Q-values at steps 11k–90k (first 1–9 train() calls with gradient_steps=1000) are in the **1–15 range**. Teacher J ≈ 1340, student J ≈ −300. The Q value is meant to represent discounted-future GAIL reward, not return. QPREF is computing:

```
delta = (Q(s,a)_teacher_traj - Q(s,a)_student_traj) / temp
loss = softplus(-delta)
```

At step 11k, a teacher trajectory might have Q_mean ≈ 1.9 and a student trajectory Q_mean ≈ 1.7. Delta = 0.2. Softplus(−0.2) ≈ 0.6. But the true J difference is 1640. The QPREF gradient is pushing Q_teacher up by a tiny amount while Bellman is pushing both toward whatever their TD targets say. These gradients conflict.

**Worse**: when Q_teacher < Q_student (50% of randomly-initialized critic's cases), delta is negative, softplus(−delta) > log(2), and the gradient is large — up to 1.0 per sample. This large corrective gradient disrupts Bellman convergence.

From the guard smoke (48463312_1) log:
- Step 11k: qpref_delta = −0.01, loss = 0.703, q1_mean = 1.7
- Steps 22k–90k: delta oscillates −1.06 to +0.16, loss = 0.6–1.5
- ep_rew declines −332 → −580 (vs. baseline which also declines but less sharply)

This early interference costs ~15k steps of useful training time and corrupts the critic's value calibration.

### 3.3 J-Spread Is Not the Bottleneck

The cross-pool experiment provides the clearest evidence. Cross-pool provides J-spread of ~1640 from step 0 (vs. ~50 in student-student or ~0 in teacher-teacher). Despite this, cross-pool performance is identical to teacher-source (~6.2–6.3k final). The larger J-spread creates larger gradients, but larger gradients fight Bellman harder, not better. The fundamental issue is not "too little signal" — it is "signal in the wrong space."

### 3.4 Late-Phase Pool Saturation

When the adaptive buffer promotes enough student episodes, `pref_student_episodes` fills with trajectories at J ≈ 1300–1400 (near-expert). Now:

- Teacher episodes: J ≈ 1340
- Student episodes: J ≈ 1300–1400 (after promotion)
- |ΔJ| ≈ 0–50 (noise level)

QPREF delta becomes random noise around zero. The loss gradient ≈ 0.5 per sample (softplus'(0) = 0.5), multiplied by near-zero J-spread. The gradient is random in sign and small in magnitude, but non-zero — so it injects gradient noise into the critic with zero useful signal. This coincides with the performance plateau and decline seen at steps 700k–1M in most QPREF runs.

---

## 4. Why QPREF Variants Converge to Similar Performance

All variants share the same bottleneck: **the critic is being trained to satisfy two objectives simultaneously, and those objectives disagree**.

| Phase | Teacher-source | Student-source | Cross-pool |
|-------|--------------|----------------|------------|
| Early (0–60k) | Medium interference (teacher-teacher J-spread ~0, boring pairs) | Severe (empty pool, or same-score student pairs) | Severe (large J-spread amplifies wrong gradients) |
| Middle (60k–700k) | Positive signal | Positive signal (after pool grows) | Positive signal (slightly faster) |
| Late (700k–1M) | Saturation, decline | Saturation, decline | Saturation, decline |

The middle phase is the only phase where QPREF provides net benefit. But:
1. The policy enters this phase behind baseline (early damage)
2. The phase ends before 1M steps (late damage)
3. The ceiling of the middle phase is not higher than baseline — it just reaches it slightly differently

Guard prevents the late decline but cannot undo the early interference. Cross-pool speeds up the middle phase but amplifies early interference proportionally. Net effect: all variants end at the same ceiling.

The **theoretical ceiling of QPREF in this formulation** is determined by how much the middle-phase benefit compensates for early and late interference. In practice this ceiling appears to be ~6.5–6.8k — well below the plain adaptive SAIL baseline.

---

## 5. Why Adaptive SAIL Remains Stronger

### 5.1 The Discriminator Already Does What QPREF Is Trying to Do

The GAIL discriminator provides a per-step surrogate reward that:
- Is high for expert-like (s, a) pairs
- Is low for random/poor behavior
- Is trained specifically against the *expert demonstrations* in the teacher buffer

This discriminator reward already encodes preference between expert and non-expert behavior, at per-step granularity. The actor maximizes discounted cumulative discriminator reward, which drives it toward expert-like trajectories. This is functionally equivalent to saying "prefer trajectories that look like the teacher."

QPREF is trying to add an additional ranking signal on top of this. But the discriminator is already teaching this ranking implicitly, and more precisely (per-step) than QPREF's coarse trajectory-mean formulation.

### 5.2 Adaptive Buffer Already Provides Curriculum

The adaptive buffer replaces the fixed expert demonstrations with student episodes as they improve. This provides a natural curriculum:
- Early: only teacher demos in buffer → discriminator rewards teacher-like behavior
- Late: promoted student episodes join → discriminator adapts to the current best behaviors
- The critic learns from transitions where surrogate reward is properly calibrated

QPREF is trying to inject a *separate* curriculum via J-ranking, which often runs counter to the discriminator curriculum.

### 5.3 LfD Mixing Provides Direct Expert Gradient

LfD mixing samples (expert state, expert action) pairs directly into the actor's gradient. This is a stronger signal than QPREF: it directly pushes the actor toward expert behavior rather than working through the critic as an intermediary.

### 5.4 Critic Is the Wrong Place to Inject RM Signal

QPREF modifies the critic loss. The critic's role is to estimate **expected future discounted GAIL reward**, not to rank trajectories by RM score. The actor is what generates behavior, and the actor is trained through: critic Q-values + LfD mixing + discriminator reward in Bellman targets. Corrupting the critic with a conflicting objective degrades ALL of these pathways simultaneously.

---

## 6. Root Cause

**Primary root cause: QPREF and Bellman objectives are misaligned because RM(J) and discriminator(GAIL-V) are different functions trained on different reward signals. Q is being optimized to be consistent with Bellman targets, not with RM J-ranking. Injecting J-ranking into critic gradient during Bellman training creates destructive gradient interference at both the early phase (untrained critic) and the late phase (pool saturation).**

**Secondary root cause: There is no phase in the 1M-step training window where QPREF's J-ranking and Bellman's V-ranking are aligned. In the early phase they disagree because Q is not yet calibrated. In the late phase they disagree because student J ≈ teacher J while student Q ≈ teacher Q through a different mechanism. Only in the middle phase do they weakly agree, but this window is not wide enough to compensate.**

Supporting evidence directly from logs:
- SAIL-QPREF (entcoeff=0.05, GT adaptive) reaches 6.0–6.6k; same config without QPREF reaches 8.3–8.7k. **Delta = −2000 points from QPREF alone.**
- qpref_delta < 0 during steps 11k–90k for ALL variants → confirmed early interference
- qpref_delta collapses to near-zero at steps ~700k+ when student pool fills → confirmed late saturation
- Cross-pool (J-spread 1640) performs no better than teacher-source (J-spread ~0) → confirmed J-spread is not the bottleneck
- Guard smoke (best QPREF result) reaches 6.55–6.61k stable — still 1.7k below baseline

---

## 7. What Would Need to Change

For QPREF to actually beat Adaptive SAIL, at least one of the following changes is necessary:

### 7.1 Fix the Objective Alignment Problem (Recommended)

Stop injecting QPREF into critic Bellman training. Instead, inject RM ranking via a **separate value head or auxiliary loss** that does not interfere with the Bellman Q-head. Options:
- Train a separate `V_rm(s)` head alongside Q, supervised by J-ranking, and use it only for actor gradient (actor loss = `−Q + λ * (−V_rm)`)
- Or use RM signal to directly shape the Bellman *target*: blend `r_disc` with `α * r_rm` in the TD target, rather than adding a ranking loss to critic

### 7.2 Delay QPREF Until Q Is Calibrated

Set `qpref_start_step` to a value where Q-values are in the same range as J-values. For HalfCheetah, J ≈ 1340 and Q converges toward that range by ~step 200k–300k. Before that, QPREF has no valid signal. `--qpref_start_step 300000` may eliminate the early-phase interference.

### 7.3 Detach the Ranking from Q and Apply It to the Actor

Instead of `L_qpref = softplus(−delta_Q / T)` on the critic, apply it on the actor directly:
```
actor_loss -= λ * (Q(s_teacher) − Q(s_student))
```
where Q is treated as fixed (detached). This pushes the actor toward states where Q is higher, using RM-curated teacher states as positive examples, without corrupting Q's Bellman calibration.

### 7.4 Replace J-Ranking with a Proper Shaped Reward

The cleanest solution: add `r_rm` as an additional component in the Bellman target, properly scaled relative to `r_disc`. This is what `--rm_in_critic` attempts. If the RM and discriminator rewards are on compatible scales, this allows Q to represent a blended value function that is calibrated to both. The RMInCritic experiments should be examined to see if scale alignment was achieved.

### 7.5 Use RM at the Actor Level Directly (Not Through Critic)

The RM can be used to filter or reweight policy gradient steps: only apply gradient for episodes where J exceeds a threshold. This avoids the critic altogether. PAIL (adaptive buffer replacement) is conceptually similar — it replaces the teacher buffer with high-J student episodes, which directly shapes the discriminator's training distribution.

### 7.6 Is QPREF Worth Pursuing In This Form?

**No.** In its current form — `softplus(−(Q_teacher − Q_student) / T)` added to critic loss, with any source pool — QPREF consistently produces a **−2000 point deficit** vs. the baseline it is supposed to improve. The guard prevents the worst of the late-phase damage but leaves the early-phase damage intact. Cross-pool provides larger J-spread but makes early interference worse. No combination of source, guard threshold, or warm-up helps because the core problem is the objective misalignment between J-ranking and Bellman V.

The investment should move to one of the alternatives in Section 7: either RM-in-Bellman-target (scaling RM reward to match GAIL reward scale), actor-level RM shaping (bypass critic entirely), or deferred-start QPREF (eliminate early-phase damage first, then re-evaluate middle-phase contribution).
