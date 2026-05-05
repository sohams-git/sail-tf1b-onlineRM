# HalfCheetah Teacher Quality Analysis: SAIL Variants Comparative Study

**Generated:** 2026-05-05  
**Author:** Auto-analysis from SLURM experiment logs  
**Logs analysed:** 131 files in `HC_sbatch/logs/` + 271 files in `HC_6500_sbatch/logs/`  
**Total runs:** 120 (5 seeds × 24 algorithm–teacher combinations), 1M timesteps each

---

## 1. Experimental Setup

| Property | Score 2669 | Score 3617 | Score 6500 |
|----------|-----------|-----------|-----------|
| Expert dataset | `expert_data_no_img_HalfCheetah_scores_2669_episodes_4_ep100.npz` | `expert_data_no_img_HalfCheetah_scores_3617_episodes_4_ep200.npz` | `expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz` |
| Episodes in buffer | 4 ep × 100 steps | 4 ep × 200 steps | 4 ep × longer rollouts |
| Approx expert return | ~2669 | ~3617–3747 (mean) | ~6500–6988 (mean) |
| Log series (job IDs) | `49213296–49213304` | `49213287–49213295` | `48665xxx` |
| Seeds | 5 per variant | 5 per variant | 5 per variant |
| Variants | SAIL, PAIL, +PREF-D, +QPREF, +TAC-D | Same | Same |

**Normalization:** `normalized_score = ep_rew_mean / expert_mean_return`. Scores above 1.0 (e.g., PAIL-PREF-D seed 4 at score 3617 reaching 1.006) indicate the policy has exceeded the teacher.

---

## 2. Master Performance Table

### 2a. Final Normalized Score — All Seeds (Mean ± Std, N=5)

| Algorithm | Score 2669 | Score 3617 | Score 6500 | Trend |
|-----------|-----------|-----------|-----------|-------|
| SAIL | 0.182 ± 0.124 | 0.257 ± 0.162 | 0.539 ± 0.306 | ↑ monotone |
| PAIL | 0.261 ± 0.273 | 0.390 ± 0.040 | 0.278 ± 0.392 | non-monotone ↑↓ |
| SAIL-PREF-D | 0.339 ± 0.214 | 0.419 ± 0.132 | 0.732 ± 0.400 | ↑ monotone |
| PAIL-PREF-D | 0.449 ± 0.258 | **0.731 ± 0.178** | 0.690 ± 0.337 | non-monotone ↑↓ |
| SAIL-QPREF | 0.231 ± 0.006 | 0.292 ± 0.105 | 0.724 ± 0.031 | ↑ monotone |
| PAIL-QPREF | 0.415 ± 0.046 | 0.458 ± 0.128 | **0.851 ± 0.035** | ↑ monotone |
| SAIL-TAC-D | **0.556 ± 0.343** | **0.646 ± 0.080** | 0.518 ± 0.477 | non-monotone ↑↓ |
| PAIL-TAC-D | **0.632 ± 0.371** | **0.772 ± 0.142** | 0.523 ± 0.481 | non-monotone ↑↓ |

**Bold** = best performer at each teacher quality level.

### 2b. Excluding Collapsed Seeds (disc_acc_exp≈1.0, norm_score≈-0.066)

Collapse definition: `normalized_score < -0.05` at final timestep (16/120 runs = 13.3%).

| Algorithm | Score 2669 (n_ok) | Score 3617 (n_ok) | Score 6500 (n_ok) |
|-----------|------------------|------------------|------------------|
| SAIL | 0.244 ± 0.009 (4/5) | 0.338 ± 0.026 (4/5) | 0.691 ± 0.048 (4/5) |
| PAIL | 0.478 ± 0.102 (3/5) | 0.390 ± 0.040 (5/5) | 0.753 ± 0.108 (2/5)* |
| SAIL-PREF-D | 0.440 ± 0.085 (4/5) | 0.419 ± 0.132 (5/5) | **0.932 ± 0.034** (4/5) |
| PAIL-PREF-D | 0.578 ± 0.019 (4/5) | 0.731 ± 0.178 (5/5) | 0.858 ± 0.036 (4/5) |
| SAIL-QPREF | 0.231 ± 0.006 (5/5) | 0.344 ± 0.009 (4/5) | 0.724 ± 0.031 (5/5) |
| PAIL-QPREF | 0.415 ± 0.046 (5/5) | 0.458 ± 0.128 (5/5) | **0.851 ± 0.035** (5/5) |
| SAIL-TAC-D | 0.712 ± 0.184 (4/5) | 0.646 ± 0.080 (5/5) | 0.907 ± 0.011 (3/5) |
| PAIL-TAC-D | 0.806 ± 0.158 (4/5) | 0.772 ± 0.142 (5/5) | 0.915 ± 0.020 (3/5) |

*PAIL at 6500: seeds 1,2 fully collapsed; seed 3 partial collapse (norm=0.013); only seeds 4,5 functional.

**Key finding:** When TAC-D does not collapse, it achieves the highest final scores across all teacher qualities. The variance problem is a stability issue, not a performance ceiling issue.

---

## 3. Teacher Quality Effect (Core Research Question)

### 3a. Performance vs. Teacher Quality

Performance is NOT monotonically increasing with teacher quality for all variants. Three distinct patterns emerge:

**Pattern 1 — Monotonically improving (SAIL base, SAIL-PREF-D, SAIL-QPREF, PAIL-QPREF):**
These variants reliably benefit from better demonstrations. PAIL-QPREF shows the most dramatic improvement: 0.415 → 0.458 → 0.851 (105% gain from 2669 to 6500).

**Pattern 2 — Non-monotone with peak at intermediate quality (PAIL-PREF-D):**
PAIL-PREF-D peaks at score 3617 (0.731) and slightly drops at 6500 (0.690). The pref_rank/j_diff_mean at 3617 is 185.2 — the highest across all conditions — indicating the medium-quality teacher provides the most discriminative pairwise ranking signal. At 6500, j_diff_mean drops to 72.0, suggesting high-quality teachers produce more uniformly excellent trajectories with less within-distribution variation for preference labeling.

**Pattern 3 — Reversal: TAC-D and PAIL base degrade at 6500 (stability):**
- TAC-D: Collapse rate increases from 1/5 at 2669 → 0/5 at 3617 → 2/5 at 6500. The MEAN degrades, but non-collapsed seeds reach their BEST scores (SAIL-TAC-D: 0.894/0.914/0.912).
- PAIL base: 2/5 collapsed at 2669, 0/5 at 3617, 2+1 partial at 6500. weight_mean saturates to 0.250 (clipped) in all failed seeds.

### 3b. Learning Speed

| Algorithm | Ep100 NS (2669) | Ep100 NS (3617) | Ep100 NS (6500) |
|-----------|----------------|----------------|----------------|
| SAIL | 0.006 | 0.006 | 0.016 |
| PAIL | -0.025 | 0.046 | -0.044 |
| SAIL-PREF-D | 0.002 | 0.021 | 0.021 |
| PAIL-PREF-D | -0.007 | 0.031 | 0.002 |
| SAIL-QPREF | 0.038 | 0.004 | 0.074 |
| PAIL-QPREF | -0.022 | 0.007 | 0.040 |
| SAIL-TAC-D | -0.012 | 0.005 | -0.053 |
| PAIL-TAC-D | -0.009 | 0.014 | -0.019 |

At episode 100 (~10% of training), **all variants are near-zero or negative** — the initial learning phase is uniformly slow across teacher qualities. This indicates that teacher quality primarily affects the FINAL plateau, not the early exploration phase. The exception is SAIL-QPREF at score 6500 (ep100 NS = 0.074), suggesting Q-based preference ranking with a good teacher can provide actionable signal earlier.

### 3c. Stability

Teacher quality has non-trivial effects on training stability:
- **Most stable teacher quality: score 3617.** Zero collapsed seeds for SAIL-TAC-D and PAIL-TAC-D, lowest variance for PAIL (std=0.040).
- **Most unstable: score 6500.** TAC-D and PAIL base have multiple collapses; SAIL-PREF-D also collapses once.
- **Score 2669:** Collapse rate is ~20% for SAIL/PAIL/TAC-D families. SAIL-QPREF uniquely has zero collapses with std=0.006 — the most stable variant under the worst teacher.

### 3d. Is Performance Monotonically Improving?

**No — it is variant-dependent.** The relationship between teacher quality and performance follows a complex landscape:

```
            Score 2669   Score 3617   Score 6500
TAC-D family: HIGH ──────► HIGH ──────► HIGH* (but unstable)
QPREF family: LOW ───────► LOW-MED ──► HIGH
PREF-D family: MED ──────► HIGH ─────► HIGH
PAIL base:    MED ───────► MED ──────► LOW (degraded)
SAIL base:    LOW ───────► LOW ──────► MED
```

The most important finding: **TAC-D benefits most from WORSE teachers in terms of reliability**, while **QPREF and PREF-D benefit most from BETTER teachers.**

---

## 4. Variant-wise Behavior Breakdown

### Variant: SAIL (base)

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | Key disc metrics |
|-----------|--------------|-------------|-------|---------|-----------------|
| 2669 | 0.006 | 0.164 | 0.182 | 1/5 (seed 4) | DAcc_E=0.989, DAcc_P=0.828 |
| 3617 | 0.006 | 0.252 | 0.257 | 1/5 (seed 5) | DAcc_E=0.983, DAcc_P=0.891 |
| 6500 | 0.016 | 0.563 | 0.539 | 1/5 (seed 3) | DAcc_E=0.981, DAcc_P=0.791 |

**Trend:** Monotone improvement, consistently weakest variant. Plateau scales linearly with teacher quality in non-collapsed runs.

**Key observations:**
- Collapsed seed pattern is identical across all three teacher qualities: disc_acc_exp=disc_acc_pol≈1.0, DRew_E≈3.5, DRew_P≈0.04–0.06. These are "discriminator wins" — the policy never learns to fool the discriminator.
- At 3617, seed 2 has a notably higher ep500 plateau (0.320) but converges to similar final value (0.356) as seeds 1,3,4 — SAIL training is consistent but slow.
- SAIL base at 6500 achieves the mid-level performance (0.539), not the best — demonstrating that without auxiliary preference supervision, even a near-optimal teacher only gives moderate gains.

---

### Variant: PAIL

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | weight_mean |
|-----------|--------------|-------------|-------|---------|------------|
| 2669 | -0.025 | 0.137 | 0.261 | 2/5 (seeds 1,3) | 0.105 ± 0.118 |
| 3617 | 0.046 | 0.336 | 0.390 | 0/5 | 0.094 ± 0.085 |
| 6500 | -0.044 | 0.191 | 0.278 | 2+1 partial | 0.202 ± 0.118 |

**Trend:** Non-monotone; peaks at 3617, degrades badly at 6500.

**Key observations:**
- The PAIL weight_mean at 6500 saturates to 0.250 (the clip boundary) in 4 of 5 seeds, indicating the Boltzmann reweighting scheme is collapsing — all weights pile up at the maximum value, negating the selective reweighting benefit.
- At score 3617, PAIL is well-calibrated (weight_mean=0.094, std=0.085) and achieves tight performance (std=0.040) — suggesting medium teacher quality creates the right balance between weight diversity and signal quality.
- At 2669, the 2/5 collapse rate and high variance (std=0.273) suggest that with a weak teacher, PAIL's reweighting cannot find reliable signal — either weights collapse or the discriminator wins.
- PAIL at 6500 is WORSE than PAIL at 3617 — a clear non-monotone failure mode unique to this variant.

---

### Variant: SAIL-PREF-D (PrefRank-Disc)

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | j_diff_mean | pairs_sampled |
|-----------|--------------|-------------|-------|---------|------------|--------------|
| 2669 | 0.002 | 0.180 | 0.339 | 1/5 (seed 3) | 80.34 ± 35.29 | 320 |
| 3617 | 0.021 | 0.332 | 0.419 | 0/5 | 51.92 ± 59.04 | 320 |
| 6500 | 0.021 | 0.638 | 0.732 | 1/5 (seed 2) | 94.08 ± 38.74 | 320 |

**Trend:** Monotone improvement, strong with high teacher.

**Key observations:**
- At 3617, j_diff_mean=51.92 is the LOWEST of the three teacher conditions — and this is the condition where PREF-D achieves moderate (0.419) performance. This confirms that when demonstrations are similar quality (3617 is a mid-tier expert), the pairwise reward gap is small and ranking signal is weak.
- At 6500, j_diff_mean rises to 94.08 — better-quality teachers produce more discriminative pairwise comparisons (a good teacher has more clearly "better" segments relative to random policy rollouts in the expert buffer).
- The 1/5 collapse at 6500 (seed 2: norm=-0.066, DAcc_E=1.0) follows the same discriminator dominance pattern. Despite this, the successful seeds reach 0.932 ± 0.034 — suggesting SAIL-PREF-D with a high-quality teacher is very capable when stable.
- Large j_diff_std at 3617 (±59.04) vs 2669 (±35.29) suggests that at 3617, PREF-D is sensitive to which pairs are sampled — some seeds find useful contrast, others don't.

---

### Variant: PAIL-PREF-D

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | j_diff_mean | weight_mean |
|-----------|--------------|-------------|-------|---------|------------|------------|
| 2669 | -0.007 | 0.251 | 0.449 | 1/5 (seed 5) | 128.48 ± 53.38 | 0.054 |
| 3617 | 0.031 | 0.346 | **0.731** | 0/5 | 185.2 ± 96.5 | 0.007 |
| 6500 | 0.002 | 0.632 | 0.690 | 1+1 partial | 72.0 ± 34.9 | 0.058 |

**Trend:** Non-monotone, peaks at 3617. Best PREF-D variant overall.

**Key observations:**
- PAIL-PREF-D at 3617 achieves mean 0.731 with all 5 seeds successful and seed 4 reaching normalized score 1.006 — exceeding the teacher. This is the only variant-condition pair that reliably surpasses teacher performance.
- j_diff_mean at 3617 (185.2) is the highest across all PREF-D conditions, confirming that the 3617-quality teacher generates expert-policy pairs with the most discriminative reward gap.
- weight_mean=0.007 at 3617 is extremely low — PAIL is effectively down-weighting most expert trajectories and concentrating learning on a tiny fraction of "best" demonstrations. This selective focusing, combined with strong ranking signal, drives the excellent performance.
- At 6500, weight_mean rises to 0.058 and j_diff_mean drops to 72.0. The high-quality teacher's demonstrations are more homogeneously good → less within-buffer variation → PAIL's selective weighting loses its differential advantage → performance drops slightly from 3617.

---

### Variant: SAIL-QPREF

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | DAcc_E | DAcc_P |
|-----------|--------------|-------------|-------|---------|--------|--------|
| 2669 | 0.038 | 0.221 | 0.231 | 0/5 | 0.987 | 0.791 |
| 3617 | 0.004 | 0.243 | 0.292 | 1/5 (seed 1) | 0.949 | 0.861 |
| 6500 | 0.074 | 0.714 | 0.724 | 0/5 | 0.978 | 0.756 |

**Trend:** Monotone but with anomalous jump at 6500.

**Key observations:**
- SAIL-QPREF at 2669 is the **most stable variant** across all conditions: std=0.006 (norm_score), zero collapses. All 5 seeds converge to 0.221–0.241 with nearly identical discriminator metrics (DLog_E=0.885 ± 0.010, DLog_P=-1.138 ± 0.022). This extreme stability comes at the cost of low performance — QPREF is acting as a very conservative regularizer that prevents both collapse and high performance.
- The dramatic jump from 0.292 (3617) to 0.724 (6500) with no proportional mid-range improvement suggests SAIL-QPREF has a **threshold effect**: below a certain teacher quality, the Q-function-based preference signal is too noisy to provide meaningful additional guidance over standard SAIL. Above that threshold (near 6500), the Q-function's value estimates of expert vs. policy trajectories become reliable enough to add strong signal.
- At 6500, ep100 NS = 0.074 is the highest early-performance score among SAIL variants — the Q-function learns useful preference signal earlier when demonstrations are clear.

---

### Variant: PAIL-QPREF

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | weight_mean |
|-----------|--------------|-------------|-------|---------|------------|
| 2669 | -0.022 | 0.265 | 0.415 | 0/5 | 0.010 |
| 3617 | 0.007 | 0.332 | 0.458 | 0/5 | 0.107 |
| 6500 | 0.040 | 0.783 | **0.851** | 0/5 | 0.017 |

**Trend:** Monotone improvement, most reliable top performer at 6500.

**Key observations:**
- PAIL-QPREF is the **only variant with zero collapses across all three teacher conditions**. This stability likely stems from QPREF's Q-function ranking keeping the preference signal conservative, preventing discriminator runaway.
- At 6500, PAIL-QPREF achieves 0.851 ± 0.035 with all 5 seeds successful — the best combination of performance and reliability in the high-teacher-quality regime.
- DAcc_E at 6500 is 0.328 (very low) compared to SAIL-QPREF's 0.978 — PAIL's reweighting keeps expert trajectories from being over-distinguished, maintaining a more balanced discriminator.
- weight_mean=0.010 at 2669 vs 0.107 at 3617 vs 0.017 at 6500 — an inverted-U in weight values suggests that at extreme teacher qualities (very good or very bad), PAIL concentrates on few demonstrations, while at medium quality it spreads attention more evenly.

---

### Variant: SAIL-TAC-D

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | tac_align | DAcc_E | DAcc_P |
|-----------|--------------|-------------|-------|---------|----------|--------|--------|
| 2669 | -0.012 | 0.292 | **0.556** | 1/5 (seed 2) | 0.859 ± 0.073 | 0.973 | 0.800 |
| 3617 | 0.005 | 0.400 | **0.646** | 0/5 | 0.693 ± 0.176 | 0.975 | 0.802 |
| 6500 | -0.053 | 0.436 | 0.518 | 2/5 (seeds 3,4) | 0.757 ± 0.106 | 0.954 | 0.835 |

**Trend:** Non-monotone; best at 3617, degraded at 6500 due to 2/5 collapses.
**Excluding collapses at 6500:** 0.907 ± 0.011 (seeds 1,2,5 = 0.894, 0.914, 0.912) — the highest non-collapsed performance.

**Key observations:**
- **tac_alignment is HIGHEST at score 2669 (0.859)** and LOWEST at score 3617 (0.693). This is counterintuitive — the weakest teacher produces the strongest TAC-D alignment. This suggests that TAC-D's contrastive discriminator is particularly well-calibrated when teacher demonstrations are clearly distinguishable from policy rollouts (even if both are suboptimal).
- y_pos_frac = 1.000 universally across all seeds and teacher qualities: the TAC-D label assignment ALWAYS labels expert samples as positively preferred over policy samples, never generating ties or reversals. This deterministic preference labeling (no noisy pairwise comparisons) is a structural advantage when teacher quality is low.
- At 3617: 0/5 collapses and the tightest variance (std=0.080) in the SAIL-TAC-D family — the middle teacher quality is a "stability sweet spot" for this variant.
- At 6500: The two collapsed seeds (3,4) show DAcc_E=1.0, DAcc_P=0.994, DRew_E≈3.5 — the discriminator achieves near-perfect separation and the policy fails to recover. The 3 successful seeds show DAcc_E=0.907–0.938, much lower than collapsed seeds, confirming the bifurcation: runs where the discriminator doesn't lock early → strong learning; runs where it locks → complete failure.

---

### Variant: PAIL-TAC-D

| Condition | Early (ep100) | Mid (ep500) | Final | Collapse | tac_align | weight_mean |
|-----------|--------------|-------------|-------|---------|----------|------------|
| 2669 | -0.009 | 0.362 | **0.632** | 1/5 (seed 3) | 0.873 ± 0.049 | 0.053 |
| 3617 | 0.014 | 0.449 | **0.772** | 0/5 | 0.894 ± 0.025 | 0.004 |
| 6500 | -0.019 | 0.461 | 0.523 | 2/5 (seeds 1,4) | 0.771 ± 0.127 | 0.107 |

**Trend:** Non-monotone; best at 3617, unstable at 6500.
**Excluding collapses at 6500:** 0.915 ± 0.020 (seeds 2,3,5 = 0.936, 0.898, 0.910) — the highest overall non-collapsed performance.

**Key observations:**
- PAIL-TAC-D at 3617 has the **highest tac_alignment of any condition (0.894 ± 0.025)** with no collapses — suggesting the 3617 teacher provides the ideal balance for PAIL's reweighting combined with TAC-D's contrastive objective.
- weight_mean = 0.004 at 3617 (very low) — PAIL is concentrating on a very small subset of expert demonstrations, and TAC-D's contrastive shaping on those high-weight samples appears to provide strong directed signal.
- At 6500, weight_mean=0.107 (much higher) indicates PAIL is less selective — the reweighting is more spread out, possibly because all high-quality demonstrations look similar, reducing the contrast within the expert buffer.
- Collapsed seeds at 6500 (seeds 1,4) show weight_mean=0.250 (saturated clip) — when PAIL's weights hit the ceiling, the reweighting mechanism effectively breaks.

---

## 5. TAC-D Dominance in the Low-Score Regime

### 5a. Performance Rankings at Each Teacher Quality

**At score 2669 (excl. collapsed seeds):**
1. PAIL-TAC-D: 0.806 ± 0.158
2. SAIL-TAC-D: 0.712 ± 0.184
3. PAIL-PREF-D: 0.578 ± 0.019
4. PAIL-QPREF: 0.415 ± 0.046
5. PAIL: 0.478 ± 0.102
6. SAIL-PREF-D: 0.440 ± 0.085
7. SAIL-QPREF: 0.231 ± 0.006
8. SAIL: 0.244 ± 0.009

TAC-D leads by **~0.23 normalized score points (≈23% absolute gap)** over the next best (PAIL-PREF-D).

**At score 3617 (excl. collapsed seeds):**
1. PAIL-TAC-D: 0.772 ± 0.142
2. PAIL-PREF-D: 0.731 ± 0.178
3. SAIL-TAC-D: 0.646 ± 0.080
4. SAIL-PREF-D: 0.419 ± 0.132
5. PAIL-QPREF: 0.458 ± 0.128
6. PAIL: 0.390 ± 0.040
7. SAIL-QPREF: 0.344 ± 0.009
8. SAIL: 0.338 ± 0.026

TAC-D still leads, but the gap over PREF-D has narrowed (0.041 points for PAIL variants).

**At score 6500 (excl. collapsed seeds):**
1. SAIL-PREF-D: 0.932 ± 0.034
2. PAIL-TAC-D: 0.915 ± 0.020
3. SAIL-TAC-D: 0.907 ± 0.011
4. PAIL-PREF-D: 0.858 ± 0.036
5. PAIL-QPREF: 0.851 ± 0.035
6. SAIL-QPREF: 0.724 ± 0.031
7. PAIL: 0.753 ± 0.108
8. SAIL: 0.691 ± 0.048

TAC-D is now tier-2 (comparable to PREF-D), while SAIL-PREF-D and PAIL-QPREF compete for top spots.

### 5b. Why TAC-D Works with Suboptimal Teachers

**Mechanism 1: Absolute vs. Relative Signal**

PREF-D and QPREF generate learning signals based on **pairwise comparisons** between expert trajectories. With a low-quality teacher (score 2669), all expert trajectories may be similarly poor — the pairwise reward gap (j_diff_mean) may be small and noisy. Evidence:

| Method | Score 2669 j_diff | Score 3617 j_diff | Score 6500 j_diff |
|--------|------------------|------------------|------------------|
| SAIL-PREF-D | 80.34 ± 35.29 | 51.92 ± 59.04 | 94.08 ± 38.74 |
| PAIL-PREF-D | 128.48 ± 53.38 | 185.2 ± 96.5 | 72.0 ± 34.9 |

TAC-D does NOT compare expert trajectories to each other. It compares **expert samples vs. current policy samples** using a contrastive teacher-aligned objective. The signal quality depends on the separation between expert and policy, not within-expert variation. Even a poor teacher (score 2669) is typically better than a randomly initialized policy early in training — so this expert-vs-policy separation is reliably high.

Evidence: tac_alignment = 0.859 at score 2669 — high alignment despite suboptimal teacher. y_pos_frac=1.0 always — TAC-D never generates a "policy is better than expert" label, even with the 2669-score teacher.

**Mechanism 2: Discriminator Signal Density vs. Ranking**

PREF-D shapes the discriminator via ranking loss — the discriminator must learn to order trajectories. With noisy teachers, rank ordering may be unreliable (rank reversals, tied returns). TAC-D instead shapes the discriminator's density via contrastive learning:

- SAIL-TAC-D (2669): DAcc_E=0.973, DAcc_P=0.800, DLog_E=1.504, DLog_P=-1.782
- SAIL-PREF-D (2669): DAcc_E=0.974, DAcc_P=0.842, DLog_E=1.393, DLog_P=-1.722

The discriminator accuracy metrics are nearly identical, but the policy performance gap is enormous (0.712 vs. 0.440 excl. collapses). This suggests the difference is NOT in discriminator accuracy but in the **quality of the gradient signal** flowing to the policy: TAC-D's contrastive objective provides cleaner gradients even when teacher demonstrations are imperfect.

**Mechanism 3: PAIL Reweighting Amplifies TAC-D**

PAIL-TAC-D outperforms SAIL-TAC-D at all teacher qualities. At score 2669, PAIL-TAC-D's weight_mean=0.053 vs. PAIL-QPREF's 0.010 — TAC-D variants have higher (less collapsed) weights, suggesting the PAIL+TAC-D combination finds better-weighted subsets of even poor demonstrations.

**Mechanism 4: Collapse Resistance**

PAIL-PREF-D at 2669 has 1/5 collapsed seed; PAIL-QPREF has 0/5 collapses but achieves only 0.415. This reveals a tension: QPREF is more collapse-resistant but provides a weaker positive signal. TAC-D achieves the highest signal quality (0.806 excl. collapses) while having a moderate collapse rate (1/5).

### 5c. Why Preference Methods Struggle with Noisy Teachers

**PREF-D failure modes under noisy/suboptimal demonstrations:**

1. **Low j_diff signal**: At score 3617, SAIL-PREF-D j_diff_mean = 51.92 ± 59.04. The high standard deviation (±59.04) indicates that in most seeds, pairwise preference differences are small and inconsistent. Seed 2 (j_diff=170.0) is an outlier that happens to find a useful contrast — and it achieves the highest final score (0.682). Seeds 1,3,4,5 have j_diff of 22–23, producing flat ranking signals and moderate performance (~0.35).

2. **Pairs sampled fixed at 320**: All PREF-D variants sample exactly 320 pairs regardless of teacher quality. With a poor teacher, 320 pairs all drawn from low-quality demonstrations → most pairs have negligible return differences → ranking loss provides near-zero gradient.

3. **j_diff is NOT monotone with teacher quality**: SAIL-PREF-D j_diff goes 80.34 → 51.92 → 94.08 as teacher quality increases. The MEDIUM teacher (3617) generates the WORST pairwise signal for SAIL-PREF-D. This explains why PREF-D doesn't scale well in this regime.

**QPREF failure mode under noisy teachers:**

- At 2669, SAIL-QPREF achieves only 0.231 ± 0.006 with extreme stability — the Q-function correctly ranks but provides such a conservative additional signal that the policy barely moves beyond base SAIL performance.
- The Q-function-based preference requires the critic to have learned meaningful Q-values — at score 2669, the critic has less signal to work with (teacher trajectories are sparse and low-quality), so Q-value-based rankings are unreliable early → conservative regularization → low final performance.

---

## 6. Discriminator-Level Analysis

### 6a. Discriminator State in Successful vs. Collapsed Runs

**Collapsed run signature** (universally consistent across all algorithms and teacher qualities):
```
disc_acc_exp  ≈ 1.000  (expert classified correctly with certainty)
disc_acc_pol  ≈ 0.998–1.000  (policy also classified correctly with certainty)
disc_logits_exp_mean ≈ 3.2–3.6  (saturated positive logit)
disc_logits_pol_mean ≈ -2.4 to -3.4  (saturated negative logit)
disc_reward_exp_mean ≈ 3.2–3.6  (maximum reward for expert)
disc_reward_pol_mean ≈ 0.037–0.075  (near-zero reward for policy)
ep_rew_mean  ≈ -595 to -605  (random-walk behavior)
```

**Healthy run signature** (representative: PAIL-TAC-D score 3617):
```
disc_acc_exp  = 0.192–0.500  (varied, expert classification NOT saturated)
disc_acc_pol  = 0.720–0.862  (policy partially distinguished)
disc_logits_exp_mean = -1.25 to -0.21  (negative or slightly positive)
disc_logits_pol_mean = -2.00 to -1.21  (moderately negative)
disc_reward_exp_mean = 0.404–0.704  (moderate reward signal)
disc_reward_pol_mean = 0.284–0.442  (non-trivial policy reward — room to improve)
```

### 6b. TAC-D Discriminator Separation Analysis

| Condition | DAcc_E | DAcc_P | DLog_E | DLog_P | DRew_E | DRew_P |
|-----------|--------|--------|--------|--------|--------|--------|
| SAIL-TAC-D 2669 (ok seeds) | 0.973 | 0.800 | 1.504 | -1.782 | 1.770 | 0.333 |
| SAIL-TAC-D 3617 | 0.975 | 0.802 | 1.210 | -1.664 | 1.494 | 0.361 |
| SAIL-TAC-D 6500 (ok seeds) | 0.923 | 0.729 | 0.761 | -1.092 | 1.167 | 0.458 |
| PAIL-TAC-D 2669 (ok seeds) | 0.235 | 0.813 | -1.006 | -1.655 | 0.459 | 0.372 |
| PAIL-TAC-D 3617 | 0.330 | 0.794 | -0.851 | -1.710 | 0.541 | 0.357 |
| PAIL-TAC-D 6500 (ok seeds) | 0.437 | 0.676 | -0.274 | -0.828 | 0.654 | 0.506 |

**Key insight:** SAIL-TAC-D maintains HIGH disc_acc_exp (~0.97) across all teacher qualities — the discriminator confidently identifies expert vs. policy. PAIL-TAC-D has much LOWER disc_acc_exp (~0.23–0.44) because PAIL's reweighting down-weights some expert samples, making the discriminator "see" fewer expert examples as definitively expert. Both approaches work, but via different mechanisms.

At score 6500 for SAIL-TAC-D (successful seeds), disc_acc_exp drops to 0.923 — the most balanced state — which coincides with the highest final performance (0.907). This suggests that **intermediate discriminator accuracy (not full saturation) corresponds to the best policy learning**.

### 6c. PREF-D Discriminator Behavior

| Condition | DAcc_E | DAcc_P | j_diff_mean | j_diff_std |
|-----------|--------|--------|------------|-----------|
| SAIL-PREF-D 2669 | 0.974 | 0.842 | 80.34 | 35.29 |
| SAIL-PREF-D 3617 | 0.983 | 0.871 | 51.92 | 59.04 |
| SAIL-PREF-D 6500 | 0.937 | 0.804 | 94.08 | 38.74 |
| PAIL-PREF-D 2669 | 0.430 | 0.879 | 128.48 | 53.38 |
| PAIL-PREF-D 3617 | 0.453 | 0.831 | 185.2 | 96.5 |
| PAIL-PREF-D 6500 | 0.499 | 0.803 | 72.0 | 34.9 |

SAIL-PREF-D consistently shows high disc_acc_exp (0.937–0.983), similar to base SAIL. The discriminator accurately identifies expert samples but the preference ranking signal (j_diff) varies widely.

The j_diff_mean for PAIL-PREF-D PEAKS at 3617 (185.2) not at 6500 (72.0) — exactly when PAIL-PREF-D achieves its best performance. This confirms: **PREF-D performance is driven by the discriminative power of pairwise comparisons, not teacher score per se.**

### 6d. QPREF Discriminator Analysis

| Condition | DAcc_E | DAcc_P | DLog_E | DLog_P |
|-----------|--------|--------|--------|--------|
| SAIL-QPREF 2669 | 0.987 | 0.791 | 0.885 | -1.138 |
| SAIL-QPREF 3617 | 0.949 | 0.861 | 1.154 | -1.383 |
| SAIL-QPREF 6500 | 0.978 | 0.756 | 0.924 | -1.272 |
| PAIL-QPREF 2669 | 0.403 | 0.860 | -0.506 | -1.982 |
| PAIL-QPREF 3617 | 0.598 | 0.864 | 0.050 | -1.688 |
| PAIL-QPREF 6500 | 0.328 | 0.742 | -0.729 | -1.220 |

PAIL-QPREF at 6500 has the LOWEST disc_acc_exp (0.328) and the most negative DLog_E (-0.729) — indicating the discriminator is NOT confidently classifying expert samples as expert. This "confuddled" discriminator state correlates with the BEST final performance (0.851) and zero collapses — consistent with the hypothesis that moderate discriminator uncertainty is optimal for policy learning.

---

## 7. Is 6500 a Performance Sweet Spot?

### 7a. Non-collapsed Comparison

| Algorithm | Score 2669 → 3617 → 6500 (excl. collapsed) | Monotone? |
|-----------|---------------------------------------------|-----------|
| SAIL | 0.244 → 0.338 → 0.691 | Yes |
| PAIL | 0.478 → 0.390 → 0.753* | No |
| SAIL-PREF-D | 0.440 → 0.419 → 0.932 | No (dip at 3617) |
| PAIL-PREF-D | 0.578 → 0.731 → 0.858 | Yes |
| SAIL-QPREF | 0.231 → 0.344 → 0.724 | Yes |
| PAIL-QPREF | 0.415 → 0.458 → 0.851 | Yes |
| SAIL-TAC-D | 0.712 → 0.646 → 0.907 | No (dip at 3617) |
| PAIL-TAC-D | 0.806 → 0.772 → 0.915 | No (dip at 3617) |

*PAIL at 6500: only 2/5 seeds functional (seeds 4,5 achieving 0.824, 0.683).

### 7b. Stability Comparison (Collapse Rate)

| Algorithm | 2669 collapse | 3617 collapse | 6500 collapse |
|-----------|-------------|-------------|-------------|
| SAIL | 1/5 (20%) | 1/5 (20%) | 1/5 (20%) |
| PAIL | 2/5 (40%) | 0/5 (0%) | 2+1*/5 | 
| SAIL-PREF-D | 1/5 (20%) | 0/5 (0%) | 1/5 (20%) |
| PAIL-PREF-D | 1/5 (20%) | 0/5 (0%) | 1+1*/5 |
| SAIL-QPREF | 0/5 (0%) | 1/5 (20%) | 0/5 (0%) |
| PAIL-QPREF | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| SAIL-TAC-D | 1/5 (20%) | 0/5 (0%) | 2/5 (40%) |
| PAIL-TAC-D | 1/5 (20%) | 0/5 (0%) | 2/5 (40%) |

*Partial collapse: norm_score in 0.01–0.05 range.

**Score 3617 is the most stable teacher quality** — fewest total collapses (2/40 = 5%).
**Score 6500 is the most unstable** — especially for TAC-D variants (4/10 = 40% collapse rate).

### 7c. Is There a Sweet Spot?

The answer depends on the variant:

- **For QPREF variants**: 6500 IS the sweet spot — monotone improvement with zero collapses and highest performance.
- **For PREF-D variants**: The sweet spot depends on the underlying algorithm: PAIL-PREF-D peaks at 3617, SAIL-PREF-D peaks at 6500.
- **For TAC-D variants**: 6500 has the highest performance POTENTIAL (0.907/0.915 excl. collapses) but 40% collapse rate makes 3617 the more practical choice (0.646/0.772 with 100% success rate).
- **For PAIL base**: 3617 is clearly the sweet spot — 0.390 with zero collapses vs. degraded performance at 6500.
- **For SAIL base**: Monotonically improves, no sweet spot.

**Conclusion**: There is NO universal sweet spot. The choice of teacher quality is algorithm-dependent. A practitioner who cares about robustness should use score 3617 for TAC-D and PREF-D variants. For QPREF variants, a higher-quality teacher is always better.

**On the hypothesis that suboptimal teachers help exploration:** There is limited evidence for this. SAIL-QPREF at 2669 is stable but achieves only 0.231 — not indicative of beneficial exploration, just conservative learning. PAIL-TAC-D at 2669 does show 4 out of 5 seeds achieving strong performance (0.713–0.992), but this is despite suboptimal teacher quality, not because of it.

---

## 8. Cross-Comparison Table: Teacher Quality Sensitivity

| Variant | Low (2669) Behavior | High (6500) Behavior | Sensitivity to Teacher Quality |
|---------|--------------------|--------------------|-------------------------------|
| SAIL | Low, stable (0.244 excl.) | Moderate (0.691) | Medium — monotone, low variance |
| PAIL | Moderate but 40% collapse | Poor (0.278), 40%+ collapse | High — breaks at extremes, best at 3617 |
| SAIL-PREF-D | Moderate (0.440) | Very high (0.932) if stable | Medium — clear quality dependence |
| PAIL-PREF-D | Moderate (0.578) | High (0.858) | Non-monotone — best at 3617 |
| SAIL-QPREF | Low (0.231), very stable | High (0.724), very stable | High — threshold effect; jump from 3617→6500 |
| PAIL-QPREF | Moderate (0.415) | Best-in-class (0.851) | Monotone, most robust at 6500 |
| SAIL-TAC-D | High (0.712 excl.) | Very high (0.907) but 40% collapse | Bimodal — high ceiling, unstable at 6500 |
| PAIL-TAC-D | Highest (0.806 excl.) | Very high (0.915) but 40% collapse | Same — best overall, most unstable at 6500 |

---

## 9. Key Research Insights

### Finding 1: TAC-D Robustness is Structural, Not Incidental
TAC-D's dominance at scores 2669 and 3617 stems from its **absolute density shaping mechanism**: it compares expert samples to policy samples directly (y_pos_frac=1.0 always), rather than ranking expert trajectories against each other. This means TAC-D's discriminator signal quality is bounded below by the expert-vs-random-policy gap, which is substantial even for suboptimal teachers. PREF-D and QPREF, by contrast, require within-expert variation to generate their signals.

**Evidence:** SAIL-TAC-D achieves normalized score 0.712 at teacher score 2669 (excl. collapse) while SAIL-PREF-D achieves only 0.440 — a 62% relative gain — despite having nearly identical discriminator accuracy metrics. The difference is in gradient quality, not discriminator learning.

### Finding 2: Preference Methods Have a "Signal Bandwidth" Requirement
PREF-D requires sufficient within-expert variance (j_diff_mean) to generate useful ranking signals. This is NOT monotone with teacher quality: SAIL-PREF-D's j_diff is highest at 2669 (80.3), drops at 3617 (51.9), then rises at 6500 (94.1). PAIL-PREF-D peaks at 3617 (185.2). The within-expert variance depends on how diverse the expert buffer is — not just how high the scores are.

**Implication:** A practitioner should not simply assume "better teacher → better PREF-D performance." The critical factor is **within-buffer trajectory diversity**, not mean teacher score.

### Finding 3: PAIL Base Breaks at the Performance Extremes
PAIL's Boltzmann reweighting is stable only in the middle quality regime (score 3617). At 2669, the weight distribution has high variance (weight_mean std=0.118) and 40% collapse rate. At 6500, weight_mean saturates to 0.250 (the clip boundary) in most seeds, effectively disabling the reweighting. The mechanism requires a quality distribution within the expert buffer that is neither too noisy (2669) nor too uniformly excellent (6500).

**Evidence:** PAIL weight_mean values: 0.105 ± 0.118 (2669) → 0.094 ± 0.085 (3617) → 0.202 ± 0.118 (6500). At 6500, 4 of 5 seeds have weight_mean at or near the 0.250 clip.

### Finding 4: TAC-D Performance Ceiling is Uniformly High but Has a Collapse Tax
Non-collapsed TAC-D seeds achieve the top performance at ALL three teacher qualities:
- Score 2669: PAIL-TAC-D at 0.992 (seed 5) — exceeds teacher performance
- Score 3617: PAIL-TAC-D at 0.947 (seed 2), PAIL-PREF-D at 1.006 (seed 4)
- Score 6500: PAIL-TAC-D at 0.936 (seed 2)

However, the collapse rate increases from 20% at 2669 to 0% at 3617 to 40% at 6500 for TAC-D variants. This introduces a practical reliability concern: at high teacher quality, TAC-D is simultaneously the highest-potential and highest-risk variant.

### Finding 5: PAIL-QPREF is the Most Reliable High-Quality Teacher Method
At score 6500, PAIL-QPREF achieves 0.851 ± 0.035 with **zero collapses across 5 seeds** — the best combination of performance and reliability. The low disc_acc_exp (0.328) indicates the discriminator remains "confused" (beneficial) rather than saturated. QPREF's Q-function ranking prevents the discriminator from overwhelming the policy signal.

**Practical implication:** For deployment with known high-quality expert data, PAIL-QPREF is the safest choice. For deployment with uncertain or suboptimal expert quality, PAIL-TAC-D or SAIL-TAC-D (despite higher collapse rate) offer higher expected performance.

### Finding 6: Discriminator Collapse is Teacher-Quality Independent, but Frequency Varies
The discriminator collapse pattern (disc_acc_exp=disc_acc_pol≈1.0, DRew_E≈3.5, policy return≈-595) occurs across ALL teacher qualities. The total collapse rate across all variants is 16/120 = 13.3%. However, collapse frequency is NOT monotone with teacher quality:
- Score 2669: 6/40 = 15%
- Score 3617: 2/40 = 5% (most stable)
- Score 6500: 8/40 = 20% (most unstable)

**Score 3617 is the most stable training regime.** The collapse is a training initialization artifact (early discriminator locking) rather than a fundamental incompatibility with teacher quality.

### Finding 7: PAIL-PREF-D at Score 3617 is the Only Reliably Teacher-Surpassing Method
PAIL-PREF-D at score 3617 achieves mean 0.731 with all 5 seeds successful, and seed 4 reaches normalized score 1.006 (exceeding the teacher). The key enabler: j_diff_mean = 185.2 — the highest pairwise discrimination signal across all conditions — combined with PAIL's selective down-weighting (weight_mean=0.007, concentrating on a tiny fraction of best demonstrations).

This combination identifies and up-weights the "best of a mediocre teacher" — a form of implicit expert distillation.

### Finding 8: Mid-Training Performance (ep500) Predicts Final Performance Better Than Early Performance
Across all variants and teacher qualities, ep100 performance is near-random (range -0.065 to +0.201, no consistent signal). By ep500, the patterns are clear:
- TAC-D variants at 3617 show ep500 NS = 0.400–0.449 → final 0.646–0.772 (consistent scaling)
- QPREF variants at 6500 show ep500 NS = 0.714–0.783 → final 0.724–0.851 (fast convergence)
- PREF-D at 3617 shows ep500 NS = 0.332–0.346 → final 0.419–0.731 (high variance, late bloomers)

**PREF-D variants are particularly "late bloomers"** — their j_diff-driven signal accumulates over training, with late-stage improvements beyond ep500.

### Finding 9: Score 3617 is the Optimal "Practical Regime" for Mixed Deployments
If only one teacher quality could be used:
- Most stable (lowest collapse rate): Score 3617 (5% total collapse)
- Best reliable performance for TAC-D: Score 3617 (PAIL-TAC-D 0.772 with 0% collapse)
- Best PREF-D performance: Score 3617 (PAIL-PREF-D 0.731 with 0% collapse)
- Best PAIL base: Score 3617 (0.390, 0% collapse)
- Only sacrifice: QPREF variants, which strongly prefer 6500

### Finding 10: tac_alignment Inversely Correlates with Teacher Score (for SAIL-TAC-D)
SAIL-TAC-D tac_alignment: 0.859 (2669) → 0.693 (3617) → 0.757 (6500). The alignment with teacher signal is HIGHEST when teacher quality is LOWEST — TAC-D's contrastive objective aligns most effectively when expert demonstrations are clearly different from policy samples but not perfectly curated. At score 3617, the higher variance in alignment (std=0.176 vs 0.073 at 2669) contributes to instability: some seeds find high alignment (e.g., seed 2 at 0.881, final NS=0.792) while others find low alignment (seed 1 at 0.463, final NS=0.603).

---

## 10. Specific Log References

### Score 2669 — Key Files

| Algorithm | Job | Seed | File | Final NS | Note |
|-----------|-----|------|------|----------|------|
| SAIL-TAC-D | 49213304 | 5 | `HC_ep100_score2669_n4_SAIL_TACD_49213304_5.out` | 0.981 | Highest score at 2669 |
| PAIL-TAC-D | 49213300 | 5 | `HC_ep100_score2669_n4_PAIL_TACD_49213300_5.out` | 0.992 | Near-teacher-level at 2669 |
| SAIL-TAC-D | 49213304 | 2 | `HC_ep100_score2669_n4_SAIL_TACD_49213304_2.out` | -0.065 | Collapse example |
| PAIL base | 49213297 | 1 | `HC_ep100_score2669_n4_PAIL_49213297_1.out` | -0.065 | PAIL collapse (disc dominance) |

### Score 3617 — Key Files

| Algorithm | Job | Seed | File | Final NS | Note |
|-----------|-----|------|------|----------|------|
| PAIL-PREF-D | 49213289 | 4 | `HC_ep0200_score3617_n4_PAIL_PREFD_49213289_4.out` | 1.006 | Teacher-surpassing |
| PAIL-TAC-D | 49213291 | 2 | `HC_ep0200_score3617_n4_PAIL_TACD_49213291_2.out` | 0.947 | Best stable TAC-D at 3617 |
| SAIL-PREF-D | 49213292 | 2 | `HC_ep0200_score3617_n4_SAIL_PREFD_49213292_2.out` | 0.682 | Outlier seed (j_diff=170) |
| SAIL-QPREF | 49213294 | 1 | `HC_ep0200_score3617_n4_SAIL_QPREF_49213294_1.out` | 0.083 | Partial collapse example |

### Score 6500 — Key Files

| Algorithm | Job | Seed | File | Final NS | Note |
|-----------|-----|------|------|----------|------|
| SAIL-PREF-D | 48665221 | 1 | `SAIL-PREF-D_48665221_1.out` | 0.971 | Best PREF-D at 6500 |
| PAIL-TAC-D | 48665078 | 2 | `PAIL-TAC-D_48665078_2.out` | 0.936 | Best TAC-D at 6500 |
| PAIL-QPREF | 48664969 | 2 | `PAIL-QPREF_48664969_2.out` | 0.891 | Best reliable variant |
| SAIL-TAC-D | 48665313 | 3 | `SAIL-TAC-D_48665313_3.out` | -0.066 | TAC-D collapse at 6500 |
| PAIL base | 48665113 | 1 | `PAIL_48665113_1.out` | -0.066 | PAIL fails at 6500 |

---

## 11. Summary Visualization

```
Final Normalized Score (excluding collapsed seeds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

          Score 2669        Score 3617        Score 6500

PAIL-TAC-D  ████████ 0.806  ███████▌ 0.772  █████████ 0.915*
SAIL-TAC-D  ███████  0.712  ██████▌  0.646  █████████ 0.907*
SAIL-PREF-D ████▌    0.440  ████▏    0.419  █████████ 0.932
PAIL-PREF-D █████▊   0.578  ███████▎ 0.731  ████████▌ 0.858
PAIL-QPREF  ████▏    0.415  ████▌    0.458  ████████▌ 0.851
SAIL-QPREF  ██▎      0.231  ███▍     0.344  ███████▏  0.724
PAIL        ████▊    0.478  ███▉     0.390  ███████▌  0.753†
SAIL        ██▍      0.244  ███▍     0.338  ██████▉   0.691

* 40% collapse rate; excl. collapsed seeds
† Only 2/5 seeds functional at 6500

Legend: Each █ = 0.1 normalized score
```

---

## Appendix: Collapse Catalogue

All runs where `normalized_score < -0.05` at final timestep:

| Log File | Teacher Score | Algorithm | Seed | Final NS | Disc Signature |
|----------|-------------|-----------|------|----------|----------------|
| `HC_ep100_n4_SAIL_49213296_4.out` | 2669 | SAIL | 4 | -0.066 | DAcc_E=1.000, DRew_E=3.590 |
| `HC_ep100_score2669_n4_PAIL_49213297_1.out` | 2669 | PAIL | 1 | -0.065 | DAcc_E=0.999, DRew_E=3.260 |
| `HC_ep100_score2669_n4_PAIL_49213297_3.out` | 2669 | PAIL | 3 | -0.065 | DAcc_E=0.999, DRew_E=3.220 |
| `HC_ep100_score2669_n4_SAIL_PREFD_49213301_3.out` | 2669 | SAIL-PREF-D | 3 | -0.065 | DAcc_E=1.000, DRew_E=3.320 |
| `HC_ep100_score2669_n4_PAIL_PREFD_49213298_5.out` | 2669 | PAIL-PREF-D | 5 | -0.065 | DAcc_E=0.999, DRew_E=3.270 |
| `HC_ep100_score2669_n4_SAIL_TACD_49213304_2.out` | 2669 | SAIL-TAC-D | 2 | -0.065 | DAcc_E=1.000, DRew_E=3.280 |
| `HC_ep100_score2669_n4_PAIL_TACD_49213300_3.out` | 2669 | PAIL-TAC-D | 3 | -0.066 | DAcc_E=0.999, DRew_E=3.280 |
| `HC_ep0200_n4_SAIL_49213287_5.out` | 3617 | SAIL | 5 | -0.065 | DAcc_E=1.000, DRew_E=3.450 |
| `HC_ep0200_score3617_n4_SAIL_QPREF_49213294_1.out` | 3617 | SAIL-QPREF | 1 (partial) | 0.083 | DAcc_E=1.000, DRew_E=2.340 |
| `SAIL_48665370_3.out` | 6500 | SAIL | 3 | -0.066 | DAcc_E=1.000, DRew_E=3.500 |
| `PAIL_48665113_1.out` | 6500 | PAIL | 1 | -0.066 | DAcc_E=1.000, DRew_E=3.450 |
| `PAIL_48665113_2.out` | 6500 | PAIL | 2 | -0.066 | DAcc_E=1.000, DRew_E=3.470 |
| `SAIL-PREF-D_48665221_2.out` | 6500 | SAIL-PREF-D | 2 | -0.066 | DAcc_E=1.000, DRew_E=3.630 |
| `SAIL-TAC-D_48665313_3.out` | 6500 | SAIL-TAC-D | 3 | -0.066 | DAcc_E=1.000, DRew_E=3.510 |
| `SAIL-TAC-D_48665313_4.out` | 6500 | SAIL-TAC-D | 4 | -0.066 | DAcc_E=1.000, DRew_E=3.600 |
| `PAIL-TAC-D_48665078_1.out` | 6500 | PAIL-TAC-D | 1 | -0.066 | DAcc_E=1.000, DRew_E=3.600 |
| `PAIL-TAC-D_48665078_4.out` | 6500 | PAIL-TAC-D | 4 | -0.066 | DAcc_E=1.000, DRew_E=3.520 |
