# Ant-v2 Run Analysis — sail_sb3 (Offline RM)

**Job IDs:** SAIL=48350960, SAIL-PREF-D=48350978, SAIL-TAC-D=48350966, SAIL-QPREF=48350972, PAIL=48350999, PAIL-PREF-D=48351018, PAIL-TAC-D=48351006, PAIL-QPREF=48351007
**Date analyzed:** 2026-04-23
**Log directory:** `sail_sb3/Ant_1200_sbatch/logs/`
**Total runs:** 40 (8 variants × 5 seeds)
**Demonstrator mean return:** 1215.9

---

## Executive Summary

Four of eight variants succeeded across all seeds: SAIL, SAIL-PREF-D, SAIL-QPREF, and SAIL-TAC-D seed 1. All four PAIL-family variants failed catastrophically (0/20 seeds), and SAIL-TAC-D failed 4/5 seeds. The failure mode for the PAIL family and SAIL-TAC-D failures is identical: the offline BPref reward model (`ant_pebble_1M`) assigns **higher scores to the early badly-performing policy than to the expert episodes**, causing an inverted learning signal whenever RM-based adaptive promotion is used. Variants using GT-based promotion (`adaptive_score_source: gt`) are unaffected.

---

## 1. SAIL (Baseline)

**Job:** 48350960 | Seeds 1–5 | `adaptive_score_source: gt`

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | **2761.75** | Steady improvement; initial dip to ~-630, then monotonic recovery |
| 2    | 1563.12     | Noisy growth; plateau ~1500 in final third |
| 3    | 2096.83     | Steady positive trajectory |
| 4    | 2580.36     | Steady improvement |
| 5    | 1874.09     | Noisy but positive overall |

**Summary statistics:**
- **Mean ± Std:** 2175.23 ± 468.4
- **Best observed:** 2761.75 (seed 1)
- **Demonstrator exceeded?** Yes — seeds 1, 4 exceed 1215.9; seeds 2, 3, 5 do not reach it
- **Completion:** 5/5 ✓
- **Variance:** Moderate

**Discriminator diagnostics (seed 1, final):**
- `disc_prob_exp_mean` = 0.677, `disc_prob_pol_mean` = 0.322 — healthy separation
- `surrogate_reward_mean` = 0.467 — in normal range
- `lfd_active` = 0 at end (LfD disabled, adaptive promotion active)

**Assessment:** Solid, consistent baseline. All seeds learn. High spread (1563–2761) is expected at this teacher quality. Shape is healthy: initial negative phase (Ant falls over at random init), then steady recovery as discriminator signal converges.

---

## 2. SAIL + PREFRANK

**Job:** 48350978 | Seeds 1–5 | `adaptive_score_source: gt` | BT ranking loss, weight=0.1

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 2331.58     | Steady positive |
| 2    | 1712.03     | Slower growth, settles below 2000 |
| 3    | 3163.64     | Strong learning, plateau ~3100 |
| 4    | 2617.56     | Steady positive |
| 5    | **3988.80** | Strongest seed; sustained improvement |

**Summary statistics:**
- **Mean ± Std:** 2762.72 ± 842.0
- **Best observed:** 3988.80 (seed 5)
- **Demonstrator exceeded?** Yes — seeds 3, 4, 5 substantially exceed 1215.9
- **Completion:** 5/5 ✓
- **Variance:** High (1712–3988 spread)

**Assessment:** PREFRANK improves mean performance over baseline SAIL (+587 mean). The BT ranking auxiliary loss adds a useful discrimination signal. High variance is primarily due to seed 5 being an outlier at 3988 — without it, mean would be ~2531. Still, the ranking loss does not hurt and appears weakly beneficial.

---

## 3. SAIL + TAC

**Job:** 48350966 | Seeds 1–5 | `adaptive_score_source: rm` | soft_tac: weight=0.5, temp=1.0

| Seed | Final Return | Curve Shape | Status |
|------|-------------|-------------|--------|
| 1    | **2122.05** | Positive improvement (only survivor) | ✓ |
| 2    | -1200.26    | Collapse — never recovers from initial negative phase | ✗ |
| 3    | -976.60     | Collapse | ✗ |
| 4    | -889.49     | Collapse | ✗ |
| 5    | -993.04     | Collapse | ✗ |

**Summary statistics:**
- **Mean ± Std:** -347.47 ± 1267.4
- **Completion:** 1/5 ✓ (seed 1 only)
- **Variance:** Extremely high (catastrophic bimodal — one survivor, four failures)

**TAC diagnostic (seed 2, final):**
- `soft_tac_loss` = 0.482, `tac_alignment` = 0.035 (near zero = policy failing to track teacher)
- Earlier in training: `tac_alignment` = -0.279 (negative = policy moving *away* from teacher)

**Assessment:** TAC with RM-based promotion fails 4/5 seeds. Seed 1 only survives due to stochastic timing — the first bad promotion in seed 1 occurs later (line ~9814) than in seed 3 (line ~5470), giving SAC just enough time to start learning before the TAC pool is corrupted. See the PAIL failure analysis below for root cause.

---

## 4. SAIL + QPREF

**Job:** 48350972 | Seeds 1–5 | `adaptive_score_source: gt` | qpref: source=student, weight=0.05, grad_interval=50

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 919.54      | Noisy, settles 900–1000 |
| 2    | **1700.99** | Noisy growth |
| 3    | 1085.20     | Moderate improvement |
| 4    | 1877.14     | Strongest in group |
| 5    | 1081.80     | Moderate |

**Summary statistics:**
- **Mean ± Std:** 1332.93 ± 413.0
- **Best observed:** 1877.14 (seed 4)
- **Demonstrator exceeded?** No — all seeds below 1215.9 except seeds 2 and 4
- **Completion:** 5/5 ✓
- **Variance:** Moderate

**Assessment:** QPREF completes reliably (GT-based promotion preserves stability) but underperforms vanilla SAIL (1333 vs 2175 mean). The Q-preference loss with weight=0.05 appears to weakly regularize the policy in a way that limits peak performance. The guard mechanism (`threshold=-0.3, confirm=3`) may be too conservative, suppressing the QPREF signal when it could help.

---

## 5. PAIL

**Job:** 48350999 | Seeds 1–5 | `adaptive_score_source: rm` | Boltzmann-weighted expert BCE, beta=5.0

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | -2845.35    | Monotonic collapse: -38 → -115 → -355 → -1070 → -2845 |
| 2    | -2217.97    | Monotonic collapse |
| 3    | -1967.75    | Monotonic collapse |
| 4    | -2373.26    | Monotonic collapse |
| 5    | **-1502.91**| Monotonic collapse (least severe) |

**Summary statistics:**
- **Mean ± Std:** -2181.45 ± 517.7
- **Best ever observed:** Near 0 (early random initialization only)
- **Completion:** 0/5 ✓ (all runs finish without crash, but all fail to learn)
- **Variance:** Moderate (consistent failure mode)

**Root cause:** See dedicated section below.

---

## 6. PAIL + PREFRANK

**Job:** 48351018 | Seeds 1–5 | `adaptive_score_source: rm` | PAIL + BT ranking loss, weight=0.1

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | -2538.40    | Monotonic collapse |
| 2    | -3001.40    | Deepest failure in any run |
| 3    | -3000.55    | Monotonic collapse |
| 4    | -2970.94    | Monotonic collapse |
| 5    | -2564.51    | Monotonic collapse |

**Summary statistics:**
- **Mean ± Std:** -2815.16 ± 206.2
- **Completion:** 0/5 ✓ (no crashes, but uniform failure)
- **Variance:** Low (consistently catastrophic)

**Assessment:** Adding PREFRANK to PAIL makes things worse, not better. Mean drops from -2181 to -2815. The ranking loss is being applied to a corrupted pool (bad student episodes treated as high-quality teachers), reinforcing the inverted signal.

---

## 7. PAIL + SOFT TAC

**Job:** 48351006 | Seeds 1–5 | `adaptive_score_source: rm` | PAIL + soft_tac: weight=0.5

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | -1845.89    | Monotonic collapse |
| 2    | -3001.58    | Deepest failure in any single run |
| 3    | -1216.07    | Collapse (least severe) |
| 4    | -1777.79    | Collapse |
| 5    | -2498.20    | Collapse |

**Summary statistics:**
- **Mean ± Std:** -2067.91 ± 857.4
- **Completion:** 0/5 ✓
- **Variance:** High (857 std — the compound failure modes produce wider spread)

**Assessment:** Combining PAIL (corrupted reweighting) with TAC (corrupted pool promotion) produces two interacting failure modes. Slightly better mean than PAIL-PREF-D (-2068 vs -2815) but higher variance.

---

## 8. PAIL + QPREF

**Job:** 48351007 | Seeds 1–5 | `adaptive_score_source: rm` | PAIL + qpref: source=student, weight=0.05

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | -2075.29    | Monotonic collapse |
| 2    | -2296.41    | Collapse |
| 3    | -2289.47    | Collapse |
| 4    | **-1961.09**| Collapse (least severe) |
| 5    | -2240.07    | Collapse |

**Summary statistics:**
- **Mean ± Std:** -2172.47 ± 126.1 (lowest variance of all PAIL variants)
- **Completion:** 0/5 ✓
- **Variance:** Low (remarkably consistent failure)

**Assessment:** QPREF's guard mechanism may slightly buffer the collapse (lowest mean among PAIL variants at -2172 vs PAIL-PREF-D at -2815), but cannot overcome the fundamental RM corruption. The low variance suggests the QPREF guard is activating consistently and damping the collapse slightly.

---

## Why the PAIL runs failed in sail_sb3

### Shared failure mode across all PAIL variants

All 20 PAIL-family runs (PAIL, PAIL-PREF-D, PAIL-TAC-D, PAIL-QPREF) fail with an identical root mechanism. The SAIL-TAC-D failures (seeds 2–5) share the same underlying cause.

### Evidence from logs

**Log header (PAIL seed 1):**
```
adaptive_score_source: rm (RM-based promotion, no GT)
pref_reweight_teacher: Boltzmann-weighted expert BCE
pref_beta=5.0
pref_expect_obs_dim=111 (Ant RM trained on native 111-dim obs)
[TeacherBuffer] Pref pool RM scores: mean=53.1 min=14.1 max=68.1 spread=54.0
[train_sail] RM adaptive: rm_expert_scores[0] (threshold) = 62.2
```

**Expert RM scores:** All 4 expert episodes scored 14.1–68.1 by the offline BPref RM (mean=53.1, threshold=62.2).

**First promotion (PAIL seed 1, line 823):**
```
[SAIL-Adaptive] Student rm_score=103.4 > threshold=62.2. Promoting.
| student_rm_score | 103 |
| ep_rew_mean      | -216 |
```

**Second promotion (line 889):**
```
[SAIL-Adaptive] Student rm_score=105.9 > threshold=62.2. Promoting.
| ep_rew_mean | -264 |
```

**Effect on Boltzmann reweighting (pool_size=4 → 5):**
```
Before promotion:  entropy=0.997  weight_max=0.443  (near-uniform weights)
After promotion:   entropy=0.016  weight_max=0.998  (single episode dominates)
                   j_max=103  (bad student episode now highest-scoring in pool)
```

The entropy of the preference weights collapsed from 0.997 (balanced) to **0.016** (one episode getting 99.8% of the weight). That episode is the promoted bad policy episode with env return -216.

**The cascade:**
1. Policy at step ~40,000 achieves env return ≈ -216 (falling, thrashing)
2. Offline BPref RM assigns this episode RM score = 103.4
3. All 4 expert episodes have RM scores 14.1–68.1 (max 68.1)
4. Since 103.4 > threshold (62.2), the bad episode is promoted into the expert pool
5. Boltzmann reweighting with β=5.0 concentrates 99.8% weight on this episode
6. The discriminator now trains overwhelmingly to classify the **bad episode** as "expert"
7. The policy learns to produce behavior matching the bad episode → ep_rew_mean decreases further
8. More bad promotions occur (rm_scores 105.9, 100.8, 88.3...) enriching the pool with bad episodes
9. Catastrophic monotonic collapse to -2845

### Root cause: offline RM is out-of-distribution for SAIL's early policy

The offline RM (`ant_pebble_1M/reward_model_1000000_0.pt`) is a BPref reward model trained on diverse rollouts from a BPref/SAC agent trained for 1M steps. This model was trained to distinguish quality differences within the BPref agent's experience — which likely never included very-negative-return trajectories like the ones SAIL produces in the first 50k–200k training steps.

When SAIL's early policy produces env returns of -200 to -300 (Ant falling backward or collapsing its legs), the offline RM is evaluating states entirely outside its training distribution. The RM's out-of-distribution extrapolation yields inflated scores (100–106) for these trajectories, which are nonsensically higher than the expert episodes (14–68).

**Obs dimension handling:** The env gives 112-dim obs (111 native + 1 time feature). The RM expects 111-dim (set via `pref_expect_obs_dim=111`). The `_fixdim` method in `pref_rm_eval.py` silently truncates: `x = x[:, :D]` (drops time feature). This truncation is consistent for both expert and student episodes, so it is **not** the direct cause of the inflated scores. The corruption is purely from OOD extrapolation.

**Why SAIL-TAC-D has the same failure mode:**
SAIL-TAC-D also uses `adaptive_score_source: rm`. Bad student episodes get promoted into the soft-TAC teacher pool. The TAC loss then pulls the policy toward these corrupted teacher episodes, producing `tac_alignment ≈ -0.279` (policy diverging from the bad "teacher"). Seed 1 survives only because stochasticity delays the first promotion to line ~9814 vs ~5470 in seed 3 — by seed 1's first promotion, SAC had already started learning something.

**Why SAIL, SAIL-PREF-D, SAIL-QPREF are unaffected:**
These three variants use `adaptive_score_source: gt`. The promotion threshold (1215.9) is based on the **actual episode return**, which no failing episode can exceed. The offline RM is never used for promotion decisions. (The RM is still loaded and used in SAIL-PREF-D and SAIL-QPREF for diagnostics/scoring, but does not gate promotions.)

### Failure classification

| Failure type | Assessment |
|---|---|
| Launch/config failure | **No** — runs start cleanly, all 20 runs complete (EXIT CODE: 0) |
| Hard crash / exception | **No** — no Python errors or tracebacks |
| NaN / inf / overflow | **No** — metrics remain finite throughout |
| Discriminator instability | **Secondary** — discriminator training after pool corruption is working correctly, just trained on the wrong data |
| Reward model miscalibration | **Primary root cause** — offline BPref RM assigns inflated OOD scores to early bad-policy episodes |
| Boltzmann weight collapse | **Proximate mechanism** — β=5.0 amplifies even a single high-scoring OOD episode to near 100% weight |
| Data/path issue | **No** — RM checkpoint loads correctly |
| Hyperparameter mismatch | **Contributing** — β=5.0 (vs β=1.0 for HalfCheetah) makes the weight collapse more extreme |

### Ranked causes of failure

1. **(Most likely / confirmed)** Offline RM is OOD for SAIL's early policy; assigns inflated scores to negative-return trajectories
2. **(Contributing)** `pref_beta=5.0` amplifies the OOD miscalibration — even a slightly inflated score monopolizes the Boltzmann weights
3. **(Possible contributing)** RM trained on BPref/SAC rollouts (which may not include very-negative-return Ant episodes) has a distribution gap with SAIL's initialization

---

## Overall Comparison

| Variant | Seeds OK | Mean Final | Best Final | Std | Assessment |
|---------|----------|-----------|-----------|-----|------------|
| SAIL | 5/5 | 2175 | 2762 | 468 | Strong baseline |
| SAIL-PREF-D | 5/5 | **2763** | **3989** | 842 | Best SAIL variant |
| SAIL-TAC-D | 1/5 | -347 | 2122 | 1267 | Broken (RM-based) |
| SAIL-QPREF | 5/5 | 1333 | 1877 | 413 | Reliable but weak |
| PAIL | 0/5 | -2181 | -1503 | 518 | Complete failure |
| PAIL-PREF-D | 0/5 | -2815 | -2538 | 206 | Complete failure |
| PAIL-TAC-D | 0/5 | -2068 | -1216 | 857 | Complete failure |
| PAIL-QPREF | 0/5 | -2172 | -1961 | 126 | Complete failure |

**Which variant looks best overall:** SAIL-PREF-D (mean 2763, one seed at 3989). With GT-based promotion intact, the BT ranking auxiliary loss adds a small consistent benefit.

**Did PREFRANK help?** Yes, mildly — SAIL-PREF-D mean (+587 vs SAIL) with high variance. Not transformative.

**Did TAC help?** No — 4/5 seeds collapsed with RM-based promotion. The one surviving seed (seed 1, +2122) cannot be attributed to TAC helping since vanilla SAIL seed 2 also reached 1563 without TAC.

**Did QPREF help?** No — SAIL-QPREF (1333) substantially underperforms SAIL (2175). The Q-preference loss with weight=0.05 appears to regularize in a harmful direction for this environment and teacher quality.

**Did PAIL behave differently from SAIL?** Yes, catastrophically so — all PAIL variants fail completely. The PAIL mechanism requires a calibrated RM for adaptive promotion, which the offline BPref RM does not provide.

**Key failure modes observed:**
1. OOD offline RM causing Boltzmann weight collapse in all PAIL variants
2. OOD offline RM causing corrupt TAC pool in SAIL-TAC-D
3. QPREF Q-preference loss weakly suppressing performance

---

## Conclusions

### Best-performing variants
1. SAIL-PREF-D — mean 2763, peak 3989
2. SAIL (baseline) — mean 2175, peak 2762

### Worst-performing variants
1. PAIL-PREF-D — mean -2815, uniformly catastrophic
2. PAIL — mean -2181
3. PAIL-QPREF — mean -2172

### Most stable variants
1. PAIL-QPREF — paradoxically, the most consistent (albeit in catastrophic failure, std=126)
2. SAIL-QPREF — most stable among the successful variants (std=413)

### Most unstable variants
1. SAIL-TAC-D — extremely bimodal (1 success, 4 catastrophic failures, std=1267)
2. SAIL-PREF-D — highest spread among successful variants (std=842)

### Key failure modes
1. **OOD offline RM → inflated student scores → Boltzmann collapse** — affects all PAIL variants and SAIL-TAC-D
2. **RM-based promotion with β=5.0 is brittle** — the combination is too sensitive to any miscalibration

### Concrete next debugging steps
1. **Replace offline RM with GT-based promotion for all PAIL variants** — test whether PAIL with GT threshold = 1215.9 recovers (this is exactly what sail_sb3_online's online RM does)
2. **Diagnose offline RM scores on held-out bad trajectories** — evaluate the BPref RM on synthetic episodes with env return -200 and confirm it assigns inflated scores
3. **Reduce pref_beta for Ant** — try β=1.0 (same as HalfCheetah) to make Boltzmann weighting less sensitive to a single high-scoring episode
4. **Switch SAIL-TAC-D to GT-based promotion** — matches the online setup and should fix the 4/5 failure rate
5. **Investigate QPREF weight** — try weight=0.1–0.5; current weight=0.05 may be too conservative to add meaningful signal

---
*Report generated 2026-04-23 from log analysis of 40 runs in `sail_sb3/Ant_1200_sbatch/logs/`.*
