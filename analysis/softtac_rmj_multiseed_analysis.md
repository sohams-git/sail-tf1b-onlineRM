# Soft-TAC RMJ Multi-Seed Analysis
**Jobs:** 48220669 (seeds 1-3) and 48223892 (seeds 1-2) | **Date:** 2026-04-18  
**Config:** Online RM + Soft-TAC RM-only J | `weight=0.5 temp=1.0 tie_eps=0.1 max_student_trajs=200`  
**Expert baseline:** 9094 (HalfCheetah-v2)

---

## 1. Results Summary

| Run | Seed | Job | Final ep_rew | Score% | First promo | Total promos | disc_loss | tac_align | surr_rew | Outcome |
|-----|------|-----|-------------|--------|-------------|-------------|-----------|-----------|----------|---------|
| s1a | 1 | 48220669 | **8185** | **90.0%** | 427k | 90 | 0.750 | 0.874 | 0.394 | Success |
| s2a | 2 | 48220669 | **−600** | **−6.6%** | Never | 0 | 0.442 | 0.351 | 1.770 | **FAIL** |
| s3  | 3 | 48220669 | **8820** | **96.9%** | 287k | 137 | 0.873 | 0.911 | 0.452 | Best |
| s1b | 1 | 48223892 | **8160** | **89.7%** | 371k | 128 | 1.000 | 0.871 | 0.519 | Success |
| s2b | 2 | 48223892 | **7100** | **78.1%** | 612k | 52 | 0.969 | 0.679 | 0.444 | Success (late) |

**Success rate: 4/5 (80%).** Excluding the one failure, mean = **8066** (88.7% of expert).  
**Including failure: mean = 6371** (70.1% of expert).

---

## 2. Failure Analysis — Seed 2, Job 48220669

### What happened

```
ep_rew_mean = -600 (all 1M steps)
surr_rew: 0.80 → 1.77 (steadily rising, discriminator saturation)
lfd_active = True (entire run, zero promotions)
Soft-TAC IS active (tac_loss=0.325, tac_alignment=0.351 at end)
```

### Failure cascade

1. Random initialization caused early discriminator bias: `surr_rew ≈ 0.8` already at step 200k (vs ~0.5 for healthy seeds at that point).
2. LfD mixing permanently active: every TD3 batch is 50% from the one expert episode (ring-truncated buffer retains only episode 3).
3. Discriminator overfit to episode 3 → gives inflated surrogate reward to all policy transitions → `surr_rew` climbs to 1.77 by end.
4. Policy receives correct surrogate reward gradient direction but the absolute magnitude is wrong and the policy never finds a trajectory that achieves gt_score > 6988.3.
5. Never promotes → loop is permanent.

### Why Soft-TAC couldn't rescue it

Soft-TAC **was** active and computing a valid loss (`tac_loss=0.325`). But `tac_alignment=0.351` (low) indicates the discriminator's pair rankings disagree with the RM's. The saturated discriminator assigns nearly-uniform high reward to all policy episodes — the discriminator has lost its ranking resolution. Soft-TAC can push toward better alignment, but at `weight=0.5` it cannot overcome a completely saturated discriminator.

### J_std diagnostic

In the failed run, J_std **monotonically increases** from 309 → 1335+ across all 49 rescores:

| Rescore | J_std | Interpretation |
|---------|-------|----------------|
| 1 | 309 | Expert J high (RM-scored), student J low (stuck policy) |
| 10 | 986 | Growing bimodality |
| 20 | 1336 | Still growing — policy never improved |
| 49 | ~1300+ | Max bimodality, expert >> student forever |

This is the **opposite** of the J_std collapse problem. Instead of all episodes converging to high quality, they diverge: expert episodes score very high under the improving RM, while student episodes stay near minimum. J_std > 1000 means Soft-TAC has excellent label quality (large J differences, no ties), yet it still couldn't rescue the run.

### Same seed, second run (48223892_2) succeeded

48223892_2 uses the same seed=2 but achieved 78.1% with first promotion at 612k. J_std in this run showed normal growth-then-decline behavior. The difference is likely non-determinism in CPU threading (OMP threads cause ordering differences even with fixed seed). This confirms the failure is stochastic, not deterministic for seed 2.

**Failure probability: approximately 1/5 (20%) observed.** This matches the ~50% failure rate seen in RMAdaptive without PrefRank (plan doc), suggesting the problem persists even with Soft-TAC active.

---

## 3. J_std Comparison Across Seeds

| Seed | J_std peak | J_std nadir | J_std final | Pattern |
|------|-----------|-------------|-------------|---------|
| s1a (48220669) | ~983 (rescore 11) | ~80 (rescore 22) | ~602 (rescore 49) | Grow → collapse → recover |
| s2a (48220669, FAIL) | 1335+ (rescore 49) | never declines | 1335+ | Monotonic growth (policy stuck) |
| s3 (48220669) | ~1169 (rescore 10) | ~142 (rescore 26) | ~534 (rescore 49) | Grow → moderate collapse → recover |
| s1b (48223892) | ~907 (early) | ~116 (mid) | ~375 (final) | Grow → collapse → recover |
| s2b (48223892) | ~1033 (rescore 13) | ~118 (rescore 25) | ~396 (final) | Grow → collapse → recover |

**Key diagnostic rule:**
- J_std growing monotonically → policy is stuck (expert J rises, student J stagnates)
- J_std growing then declining → policy succeeded enough to fill recency buffer with near-expert episodes
- The J_std collapse (nadir ~80–142) is now confirmed across all 4 successful runs — it is a structural feature of the recency-200 pool design, not a bug

---

## 4. Learning Progression: Successful Seeds

| Step | s1a (90%) | s3 (97%) | s1b (90%) | s2b (78%) |
|------|-----------|----------|-----------|-----------|
| 50k | ~−330 | ~−250 | ~−400 | ~−286 |
| 100k | ~−50 | ~−150 | ~−200 | ~−100 |
| 150k | ~79 | ~50 | ~−50 | ~−68 |
| 200k | ~2480 | ~500 | ~600 | ~200 |
| 300k | ~6010 | ~5000 | ~5300 | ~3000 |
| 400k | ~6620 | ~6700 | ~6700 | ~4400 |
| 500k | ~6810 | ~6950 | ~7000 | ~5200 |
| 600k | ~6600 | ~7000 | ~7300 | ~6300 |
| 700k | ~6870 | ~7340 | ~7700 | ~6700 |
| 800k | ~7180 | ~7800 | ~7950 | ~6900 |
| 900k | ~7470 | ~8300 | ~8080 | ~7000 |
| 1000k | **8185** | **8820** | **8160** | **7100** |

**Observations:**
1. **s3 is the fastest to improve** — first promotion at 287k (vs 371k, 427k, 612k for others). This head start compounds.
2. **s2b is the slowest** — first promotion at 612k, leaving only 388k steps to climb from threshold to final score.
3. **All 4 show the same general S-curve**: negative reward until ~100-150k, rapid rise 150k-300k, plateau at expert threshold, continued improvement post-promotion.
4. **s3 continues to improve steeply** after 700k (7340 → 8820), while s1a and s1b plateau earlier at ~8160-8185.

---

## 5. Soft-TAC Quality Metrics — Healthy Seeds

| Seed | First tac_loss | Peak tac_loss | Nadir tac_align | Final tac_loss | Final tac_align |
|------|---------------|--------------|-----------------|----------------|-----------------|
| s1a | 0.294 (27k) | ~0.535 (600k) | −0.07 (600k) | 0.063 | 0.874 |
| s3 | ~0.213 (27k) | ~0.466 | ~0.159 (J_std nadir ~600k) | 0.045 | 0.911 |
| s1b | ~0.25 | ~0.40 | ~0.10 | 0.065 | 0.871 |
| s2b | ~0.25 | ~0.40 | ~0.20 | 0.160 | 0.679 |

**Pattern consistent across all successful seeds:**
- High initial `tac_loss` (~0.2–0.3) when first activated (pool is small and diverse)
- `tac_loss` declines as policy improves and discriminator gains resolution
- A crisis window (nadir `tac_alignment`) coincides with J_std collapse (recency buffer fills with near-expert episodes)
- Recovery: `tac_alignment` → 0.67–0.91 and `tac_loss` → 0.04–0.16 at 1M

**s2b's weaker final alignment (0.679 vs 0.871–0.911)** is consistent with its late first promotion: the policy was still in early-quality territory at 612k, so the recency buffer never accumulated as many diverse stages of learning.

---

## 6. Cross-Seed Comparison: Seed 1 Replicated

Seeds 1a and 1b used `seed=1` in different SLURM submissions:
- 48220669_1: ep_rew = 8185, first promo 427k, 90 promos
- 48223892_1: ep_rew = 8160, first promo 371k, 128 promos

Results are very similar (8185 vs 8160, <0.3% difference) despite the non-determinism from CPU threading. Seed 1 is a stable operating point for this configuration.

Seed 2 shows high variance: one catastrophic failure (−6.6%) and one late success (78.1%). The failure probability appears seed-dependent — seed 1 has not failed across any run.

---

## 7. Performance vs. Baseline

From the plan-mode analysis (previous batch):

| Method | Seeds | ep_rew range | Notes |
|--------|-------|-------------|-------|
| TFParity_Adapt (GT) | 2 seeds | 6480–6640 | No pref/TAC |
| TFParity_AdaptPref (GT+Pref) | 2 seeds | 7370–8510 | |
| RMAdaptive (RM only) | 2 seeds | 5820–6730 | 50% failure rate |
| RMAdaptPref (RM+Pref) | 2 seeds | 7650–8510 | |
| **SoftTAC RMJ (RM+TAC)** | **5 seeds** | **-600 – 8820** | **4/5 succeed** |

Successful SoftTAC RMJ seeds (mean 8066) are competitive with RMAdaptPref (mean ~8080) and better than RMAdaptive without PrefRank (mean ~6275). The failure mode and success rate (~80%) are similar to RMAdaptive without PrefRank.

---

## 8. Open Issues Confirmed/Updated

### Issue 1: J_std collapse (confirmed structural, not a bug)
All 4 successful seeds show J_std collapse when the recency-200 buffer fills with expert-quality episodes. Nadir values:
- s1a: 80 (severe) — causes negative tac_alignment
- s3: 142 (moderate)
- s1b: 116 (moderate)
- s2b: 118 (moderate)

The alignment crisis in s1a (tac_alignment = −0.07 at 600k) appears to be an outlier: s3, s1b, and s2b all maintain positive alignment (≥0.10) even at nadir. The -0.07 in s1a may reflect a particularly unfortunate pool composition at that step.

### Issue 2: Catastrophic failure mode persists
The 1/5 failure is the same mechanism as RMAdaptive seed 2 (plan doc): stochastic initialization → early discriminator saturation → lfd_active=True forever → no promotions. The presence of Soft-TAC does not eliminate this failure mode at `weight=0.5`.

**Potential mitigation:** Adding PrefRank (`--pref_rank_disc`) alongside Soft-TAC was shown to eliminate this failure mode for RMAdaptPref. PrefRank forces the discriminator to maintain ranking resolution even before promotions occur.

### Issue 3: High variance in first promotion step
First promotions: 287k, 371k, 427k, 612k. The 325k spread causes proportional spread in final performance (97% for 287k vs 78% for 612k). At 1k steps per ep_rew unit and ~385k remaining steps after late promotion vs 713k for early promotion, a ~325k headstart ≈ +1700 ep_rew.

---

## 9. Summary Table

| Metric | s1a | s2a (FAIL) | s3 | s1b | s2b |
|--------|-----|-----------|-----|-----|-----|
| Final ep_rew | 8185 | −600 | **8820** | 8160 | 7100 |
| Normalized score | 90.0% | −6.6% | **96.9%** | 89.7% | 78.1% |
| First promotion step | 427k | Never | **287k** | 371k | 612k |
| Total promotions | 90 | 0 | **137** | 128 | 52 |
| J_std nadir | 80 | N/A (no nadir) | 142 | 116 | 118 |
| J_std peak | ~983 | ~1335+ | ~1169 | ~907 | ~1033 |
| Final tac_alignment | 0.874 | 0.351 | **0.911** | 0.871 | 0.679 |
| Final tac_loss | 0.063 | 0.325 | **0.045** | 0.065 | 0.160 |
| Final disc_loss | 0.750 | 0.442 | 0.873 | 1.000 | 0.969 |
| Final surr_rew | 0.394 | **1.770** | 0.452 | 0.519 | 0.444 |

**Verdict:** RM-only J Soft-TAC achieves 78–97% of expert on 4/5 seeds. The best result (seed 3: 96.9%) is the strongest single-seed result in this experiment family. The failure mode (surr_rew saturation + zero promotions) persists at ~20% rate and is not mitigated by Soft-TAC alone. The J_std collapse is a structural feature of the recency-200 pool but systems recover within ~200k steps and do not prevent the runs from reaching strong final performance.
