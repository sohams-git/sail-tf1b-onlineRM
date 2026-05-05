# Ant-v2 Run Analysis — sail_sb3_online (Online RM)

**Job IDs:** SAIL-PREF-D=48350731, SAIL-TAC-D=48350784, SAIL-QPREF=48350758, PAIL=48350723, PAIL-PREF-D=48350689, PAIL-TAC-D=48350701, PAIL-QPREF=48350695
**Date analyzed:** 2026-04-23
**Log directory:** `sail_sb3_online/Ant_1200_sbatch/logs/`
**Total runs:** 35 (7 variants × 5 seeds)
**Demonstrator mean return:** 1215.9
**Note:** Plain SAIL (no preference augmentation) is absent from this directory — 7 variants present, not 8.

---

## Executive Summary

All 35 runs completed successfully. Every variant, including all PAIL-family variants, succeeded — in stark contrast to sail_sb3 (offline RM), where all 20 PAIL runs failed catastrophically. The online reward model (trained during the run via human-preference simulation with held-out accuracy ≥ 0.60) provides correctly-calibrated episode scores, resolving the OOD miscalibration that destroyed the offline PAIL family. The best-performing variant is **PAIL + PREFRANK** (mean 4182, all seeds above 3400). SAIL + TAC and PAIL + TAC are also strong. SAIL + QPREF and PAIL + QPREF underperform relative to the other variants.

---

## 1. SAIL Runs

**Plain SAIL is not present in this directory.** No log files with prefix `SAIL_` exist. Only SAIL with preference augmentation (PREF-D, TAC-D, QPREF) is available. This section is therefore omitted — see sail_sb3 for baseline SAIL performance.

---

## 2. SAIL + PREFRANK

**Job:** 48350731 | Seeds 1–5 | `adaptive_score_source: gt` | BT ranking loss, weight=0.1 | Online RM active

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 2617.79     | Steady positive |
| 2    | **3249.69** | Strong sustained growth |
| 3    | 3170.47     | Strong growth, plateau ~3100 |
| 4    | 2703.79     | Moderate growth |
| 5    | 3104.69     | Strong growth |

**Summary statistics:**
- **Mean ± Std:** 2969.29 ± 293.0
- **Best observed:** 3249.69 (seed 2)
- **Demonstrator exceeded?** Yes — all 5 seeds exceed 1215.9; seeds 2, 3, 5 exceed 3000
- **Completion:** 5/5 ✓
- **Variance:** Low-moderate

**Online RM behavior:** GT-based promotion used for adaptive scoring. Online RM trained but not used for promotion gating (it contributes to the discriminator preference ranking auxiliary signal). This variant is stable by construction.

**Comparison to offline:** Offline SAIL-PREF-D mean=2763 ± 842. Online SAIL-PREF-D mean=2969 ± 293. Online version achieves comparable mean with substantially lower variance — the online RM provides tighter ranking supervision.

**Assessment:** Consistent, reliable learner. The BT ranking loss helps, and the lower variance vs the offline version suggests the online RM adds a stabilizing discriminator signal. All seeds exceed the demonstrator return.

---

## 3. SAIL + TAC

**Job:** 48350784 | Seeds 1–5 | `adaptive_score_source: gt` | soft_tac: weight=0.5, temp=1.0 | Online RM active

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 3387.53     | Strong positive trajectory |
| 2    | **4060.95** | Best SAIL variant; sustained high returns |
| 3    | 2057.42     | Weakest seed; modest growth |
| 4    | 3234.85     | Steady improvement |
| 5    | 3979.05     | Strong; near seed 2 level |

**Summary statistics:**
- **Mean ± Std:** 3344.16 ± 765.4
- **Best observed:** 4060.95 (seed 2)
- **Demonstrator exceeded?** Yes — all seeds exceed 1215.9; seeds 1, 2, 4, 5 exceed 3000
- **Completion:** 5/5 ✓
- **Variance:** Moderate-high (seed 3 outlier at 2057)

**Key difference from offline SAIL-TAC-D:** Online version uses `adaptive_score_source: gt` (GT-based promotion). Offline SAIL-TAC-D used `adaptive_score_source: rm` and failed 4/5 seeds. Switching to GT-based scoring completely resolves the failure mode.

**Assessment:** Best-performing SAIL variant in this directory. The TAC soft alignment loss works well when the teacher pool is populated with correctly-promoted episodes. Seed 3 is an outlier at 2057 (moderate success, not failure), which keeps variance elevated. Mean 3344 substantially exceeds vanilla offline SAIL (2175).

---

## 4. SAIL + QPREF

**Job:** 48350758 | Seeds 1–5 | `adaptive_score_source: gt` | qpref: source=student, weight=0.05, grad_interval=50 | Online RM active

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 1671.48     | Modest improvement |
| 2    | 1653.49     | Modest improvement |
| 3    | 1190.84     | Near demonstrator level; no further growth |
| 4    | **2700.55** | Outlier — substantially better than peers |
| 5    | 1294.22     | Modest improvement |

**Summary statistics:**
- **Mean ± Std:** 1702.12 ± 588.5
- **Best observed:** 2700.55 (seed 4 outlier)
- **Demonstrator exceeded?** Seeds 1, 2, 4 marginally exceed 1215.9; seed 3 just under; seed 5 just above
- **Completion:** 5/5 ✓
- **Variance:** Moderate (primarily driven by seed 4 outlier)

**Comparison to offline SAIL-QPREF:** Offline mean=1333 ± 413. Online mean=1702 ± 589. Slight improvement in mean, slightly higher variance. Online RM does not substantially help QPREF — both versions plateau in the 1200–1900 range (excluding seed 4 outlier).

**Assessment:** QPREF consistently underperforms other SAIL variants. The Q-preference loss with weight=0.05 appears insufficient to add meaningful signal beyond the discriminator. Seed 4 at 2700 suggests the mechanism *can* work but does not reliably trigger. The pattern matches the offline version: stable but weak.

---

## 5. PAIL

**Job:** 48350723 | Seeds 1–5 | `adaptive_score_source: gt` (GT adaptive) + Online RM for Boltzmann reweighting | pref_beta=5.0

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 2756.64     | Steady positive; online RM activates early and provides progressive improvement |
| 2    | **3070.85** | Strong sustained growth |
| 3    | 2925.17     | Steady |
| 4    | 2931.22     | Steady |
| 5    | 1700.84     | Weakest seed; moderate growth |

**Summary statistics:**
- **Mean ± Std:** 2676.94 ± 497.5
- **Best observed:** 3070.85 (seed 2)
- **Demonstrator exceeded?** Yes — all seeds exceed 1215.9; seeds 1–4 exceed 2700
- **Completion:** 5/5 ✓
- **Variance:** Moderate

**Online RM activation (seed 1):**
```
[OnlineRMManager] RM ACTIVATED: segments=594 updates=50 held_out_acc=0.750
[OnlineRMManager] Rescore #1: 4 episodes updated, mean |ΔJ|=964.51
[OnlineRMManager] Rescore #2: 4 episodes updated, mean |ΔJ|=64.70
...
[OnlineRMManager] Rescore #20: 91 episodes updated, mean |ΔJ|=95.82
```

The online RM activates after accumulating sufficient preference segments (≥500) with held-out accuracy ≥ 0.60. It then rescores the pref pool every 20,000 env steps. The large initial |ΔJ|=964.5 confirms the GT scores and online RM scores converge, correcting the initial uniform weights.

**Comparison to offline PAIL:** Offline mean=-2181. Online mean=+2677. **Delta: +4858.** This is the most dramatic improvement in the experiment. The online RM completely reverses PAIL's failure mode.

**Assessment:** Online PAIL succeeds cleanly. The online RM correctly ranks episode quality so the Boltzmann reweighting assigns high weight to genuinely good episodes. Seed 5 at 1700 is the only underperformer; all others are 2700+.

---

## 6. PAIL + PREFRANK

**Job:** 48350689 | Seeds 1–5 | Online RM + PAIL + BT ranking loss, weight=0.1

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 4156.55     | Strong sustained growth |
| 2    | 4456.94     | Strong |
| 3    | 4283.53     | Strong |
| 4    | **4565.33** | Highest single-seed return in entire experiment |
| 5    | 3449.03     | Slightly weaker, but still strong |

**Summary statistics:**
- **Mean ± Std:** 4182.28 ± 504.0
- **Best observed:** 4565.33 (seed 4)
- **Demonstrator exceeded?** Yes — all seeds exceed 1215.9 by a factor of 2.8–3.7×
- **Completion:** 5/5 ✓
- **Variance:** Moderate (driven mainly by seed 5 outlier at 3449)

**Assessment:** **Best-performing variant in the entire sail_sb3_online experiment.** The combination of online PAIL (correctly-calibrated Boltzmann reweighting) and BT preference ranking (discriminator auxiliary loss) is synergistic. Online PAIL establishes a good preference ordering; PREFRANK reinforces this ordering through the discriminator. Mean 4182 is the highest across all 75 runs analyzed. In stark contrast, offline PAIL-PREF-D achieved mean -2815 (the worst in the offline experiment).

---

## 7. PAIL + SOFT TAC

**Job:** 48350701 | Seeds 1–5 | Online RM + PAIL + soft_tac: weight=0.5

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 3449.58     | Strong growth |
| 2    | 4106.33     | Strong growth |
| 3    | 3296.69     | Steady moderate growth |
| 4    | **4654.26** | Highest single-seed return across all PAIL-TAC variants |
| 5    | 1441.71     | Weakest seed; modest but positive growth |

**Summary statistics:**
- **Mean ± Std:** 3389.71 ± 1431.3
- **Best observed:** 4654.26 (seed 4)
- **Demonstrator exceeded?** Yes — all seeds exceed 1215.9
- **Completion:** 5/5 ✓
- **Variance:** High (seed 5 at 1441 pulls mean down; seed 4 at 4654 pulls it up)

**Comparison to offline PAIL-TAC-D:** Offline mean=-2068. Online mean=+3390. **Delta: +5458.** All 5 seeds recover.

**Assessment:** Online PAIL-TAC-D succeeds but with high variance. Seed 4 is exceptional (4654), seed 5 is weak (1441). The TAC mechanism with an online-RM-validated pool is effective when the pool is clean, but the timing of the online RM activation vs TAC pool construction appears to create occasional variance. Second-best mean PAIL variant (3390 vs PAIL-PREF-D at 4182), but notably higher variance.

---

## 8. PAIL + QPREF

**Job:** 48350695 | Seeds 1–5 | Online RM + PAIL + qpref: source=student, weight=0.05

| Seed | Final Return | Curve Shape |
|------|-------------|-------------|
| 1    | 2419.83     | Moderate positive growth |
| 2    | 2034.16     | Modest growth |
| 3    | **3449.94** | Best seed — outlier performance |
| 4    | 3173.26     | Strong growth |
| 5    | 2159.34     | Modest growth |

**Summary statistics:**
- **Mean ± Std:** 2647.11 ± 631.3
- **Best observed:** 3449.94 (seed 3)
- **Demonstrator exceeded?** Yes — all seeds exceed 1215.9
- **Completion:** 5/5 ✓
- **Variance:** Moderate

**Comparison to offline PAIL-QPREF:** Offline mean=-2172. Online mean=+2647. **Delta: +4819.** All 5 seeds recover.

**Assessment:** Online PAIL-QPREF succeeds and is stable, but underperforms PAIL base (2647 vs 2677 — essentially the same). QPREF with weight=0.05 adds minimal signal beyond the base PAIL mechanism. Pattern consistent with the offline SAIL-QPREF finding: QPREF at this weight is neutral-to-slightly-negative on performance.

---

## Overall Comparison

### Summary table (sail_sb3_online only)

| Variant | Seeds OK | Mean Final | Best Final | Std | vs. Offline Mean |
|---------|----------|-----------|-----------|-----|-----------------|
| SAIL | N/A | N/A | N/A | N/A | (not present) |
| SAIL-PREF-D | 5/5 | 2969 | 3250 | 293 | +207 (↑7%) |
| SAIL-TAC-D | 5/5 | **3344** | 4061 | 765 | **+3691 (↑1062%)** |
| SAIL-QPREF | 5/5 | 1702 | 2701 | 589 | +369 (↑28%) |
| PAIL | 5/5 | 2677 | 3071 | 498 | **+4858 (↑∞)** |
| PAIL-PREF-D | 5/5 | **4182** | 4565 | 504 | **+6997 (↑∞)** |
| PAIL-TAC-D | 5/5 | 3390 | 4654 | 1431 | **+5458 (↑∞)** |
| PAIL-QPREF | 5/5 | 2647 | 3450 | 631 | **+4819 (↑∞)** |

### Which variant looks best overall

**Online PAIL-PREF-D** — mean 4182, peak 4565, all 5 seeds above 3400. Combines the correctly-calibrated online RM Boltzmann reweighting with BT preference ranking on the discriminator. Synergistic.

### Effect of preference augmentations

**PREFRANK (BT ranking):**
- SAIL: +206 (2763 → 2969), variance reduced (+). Mildly helpful.
- PAIL: +1505 (2677 → 4182), variance roughly maintained. Strongly helpful.
- **PREFRANK is beneficial in the online RM setting, especially for PAIL.**

**TAC (soft teacher alignment):**
- SAIL: +375 (2969 → 3344), variance increases. Mildly helpful, less stable.
- PAIL: +713 (2677 → 3390), variance increases substantially (std=1431). Helpful but introduces seed 5-type failures.
- **TAC helps but increases variance in both cases.**

**QPREF (Q-function preference loss):**
- SAIL: -1267 (2969 → 1702). Hurtful.
- PAIL: -30 (2677 → 2647). Neutral.
- **QPREF at weight=0.05 consistently underperforms. Does not benefit from online RM.**

### Does online RM change the relative ranking of methods?

Yes, dramatically so:

| Rank | Offline (sail_sb3) | Online (sail_sb3_online) |
|------|-------------------|------------------------|
| 1    | SAIL-PREF-D (2763) | PAIL-PREF-D (4182) |
| 2    | SAIL (2175) | PAIL-TAC-D (3390) |
| 3    | SAIL-TAC-D (−347 avg, 1 seed OK) | SAIL-TAC-D (3344) |
| 4    | SAIL-QPREF (1333) | SAIL-PREF-D (2969) |
| 5–8  | All PAIL variants (all negative) | PAIL (2677), PAIL-QPREF (2647) |

The online RM completely inverts the PAIL family from worst to best. In the offline setting, PAIL variants are uniformly catastrophic. In the online setting, they are either the best (PAIL-PREF-D) or competitive (PAIL, PAIL-QPREF). The online RM is not a minor tuning — it is a **prerequisite for PAIL to function at all** on Ant-v2 with this teacher quality.

### PAIL vs SAIL in the online setting

With online RM: PAIL (2677) vs SAIL-PREF-D (2969) — PAIL is competitive. PAIL-PREF-D (4182) clearly outperforms all SAIL variants. This suggests PAIL's preference reweighting mechanism is intrinsically beneficial when the RM is reliable.

---

## Conclusions

### Best-performing variants
1. PAIL-PREF-D — mean 4182, peak 4565, lowest variance of high-performing variants
2. PAIL-TAC-D — mean 3390, peak 4654 (but high variance, seed 5 at 1441)
3. SAIL-TAC-D — mean 3344, peak 4061

### Worst-performing variants
1. SAIL-QPREF — mean 1702 (well below demonstrator level for most seeds)
2. PAIL-QPREF — mean 2647 (competitive but weakest PAIL variant)

### Most stable variants
1. SAIL-PREF-D — std=293 (by far the most consistent)
2. PAIL-PREF-D — std=504 (consistent despite high mean)
3. PAIL — std=498

### Most unstable variants
1. PAIL-TAC-D — std=1431 (seed 4 at 4654 vs seed 5 at 1441)
2. SAIL-TAC-D — std=765 (seed 3 at 2057 is an outlier)
3. SAIL-QPREF — std=589 (seed 4 outlier at 2700)

### Key failure modes observed

No catastrophic failures in this directory. The online RM resolves all PAIL failures. Residual issues:
1. **High variance in TAC variants** — soft teacher alignment is effective but sensitive to timing of online RM activation vs pool construction; occasional weak seeds (seed 5 in PAIL-TAC-D at 1441)
2. **QPREF consistently underperforms** — weight=0.05 too small; or the Q-preference loss mechanism needs a stronger or better-tuned signal even with a reliable RM
3. **SAIL-QPREF seed 3 at 1190** — nearly at demonstrator level but no further improvement; possible plateau from overly conservative QPREF guard

### Concrete next debugging steps
1. **Establish plain SAIL baseline in sail_sb3_online** — missing from this directory; needed to isolate the contribution of online RM alone
2. **Increase QPREF weight** — test weight=0.1–0.5 in both settings; current 0.05 appears too conservative across all conditions
3. **Investigate PAIL-TAC-D seed 5 failure** — determine whether the weak seed is due to online RM activation timing or TAC pool corruption; compare RM activation line numbers across seeds
4. **Run PAIL-PREF-D longer** — at mean 4182, some seeds are still growing (not plateaued); the 1200-ep budget may be limiting peak performance
5. **Port successful online PAIL + PREFRANK to HalfCheetah** — confirm the online RM fix is environment-general, not Ant-specific

---
*Report generated 2026-04-23 from log analysis of 35 runs in `sail_sb3_online/Ant_1200_sbatch/logs/`.*
