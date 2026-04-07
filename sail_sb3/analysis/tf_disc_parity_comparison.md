# TF Disc Parity (ec=0.01) — PyTorch Run Analysis

**Environment:** HalfCheetah-v2  
**Date:** 2026-04-05  
**Purpose:** Evaluate effect of matching TF's entcoeff=0.01 in PyTorch, with and without PrefRank.  
**Jobs:** `HC_Adaptive_LfD_TFDisc_46766558` (seeds 1–2), `HC_AdaptPref_LfD_TFDisc_46766592` (seeds 1–2)

---

## Run Configurations

Both sbatch files set `--entcoeff 0.01` (matching TF's hardcoded value at `sail.py:224`). All other hyperparameters are identical to the previous `ec=0.05` runs.

| Config | Value |
|--------|-------|
| entcoeff | **0.01** (TF-matched) |
| disc_train_freq | 500 (same as TF) |
| disc_gradient_steps | 10 (same as TF) |
| disc_batch_size | 256 |
| gradcoeff | 10.0 |
| lfd_mixing | Yes |
| adaptive | Yes |
| pref_rank_weight | 0.0 (Adaptive) / 0.1 (AdaptPref) |

---

## Final Results

### Adaptive SAIL (no PrefRank) — ec=0.01

| Seed | Final ep_rew @1M | Best ep_rew | disc_loss @1M | surrogate @1M | Promotions | Teacher buf |
|------|------------------|-------------|---------------|---------------|------------|-------------|
| 1 | **8,460** | 8,460 | 1.25 | 0.67 | 201 | 205,000 |
| 2 | **8,830** | 8,830 | 1.29 | 0.67 | 195 | 199,000 |

First promotions: seed=1 at 256k steps (score 6,848), seed=2 at 201k steps (score 6,789).

### Adaptive + PrefRank — ec=0.01 (incomplete)

| Seed | Last ep_rew | Steps reached | disc_loss | pref_loss | Promotions | Teacher buf |
|------|-------------|---------------|-----------|-----------|------------|-------------|
| 1 | **7,760** | ~920k | 1.04 | 0.08 | 104 | 108,000 |
| 2 | **8,090** | ~739k | 1.27 | 0.03 | ~100 | ~100,000 |

First promotion seed=1: **575k steps** (score 6,775). Note: incomplete — both runs hit the 24-hour SLURM limit before reaching 1M.

---

## Discriminator Behavior — ec=0.01 vs ec=0.05

### Adaptive SAIL seed=1 milestone comparison

| Step | ec=0.01 disc_loss | ec=0.01 surrogate | ec=0.05 disc_loss | ec=0.05 surrogate |
|------|-------------------|-------------------|--------------------|-------------------|
| 20k | 0.604 | 0.836 | 0.624 | 0.813 |
| 50k | 0.318 | 1.15 | 0.443 | 0.963 |
| 100k | 0.341 | 1.15 | 0.389 | 1.09 |
| 200k | 0.661 | 0.939 | 0.857 | 0.821 |
| 300k | 0.937 | 0.512 | 1.05 | 0.562 |
| 500k | 1.13 | 0.599 | 1.15 | 0.609 |
| 1M | 1.25 | 0.67 | 1.26 | 0.67 |

**Key observation:** At 50–100k steps, ec=0.01 has a lower disc_loss and higher surrogate reward (discriminator is more confident/saturated). But by 300k steps the two trajectories converge and by 1M they are **nearly identical** (disc_loss: 1.25 vs 1.26; surrogate: 0.67 vs 0.67).

---

## Q1: Does ec=0.01 help, hurt, or not matter for Adaptive SAIL?

**Answer: Doesn't matter (marginally helps in this run).**

- ec=0.01: seeds 1–2 achieve 8,460–8,830 at 1M
- ec=0.05: seeds 1–2 achieve 8,290–8,290 at 1M

The difference (170–540 reward) is within normal seed variance for HalfCheetah-v2 (±200–400). There is no statistically reliable advantage. The previous hypothesis that ec=0.05 prevents saturation while ec=0.01 causes it is **not supported** by the final results — both produce healthy discriminators by 200–300k steps.

**Why the early saturation (disc_loss~0.32 at 50k for ec=0.01) doesn't hurt:**
The disc saturates because the early policy is terrible (-603 ep_rew at 50k) — easily distinguished from expert (6741+). Saturation here is correct behavior: the expert IS better than the policy by a wide margin. The surrogate reward stays high (1.15) and correctly shapes the policy to improve. As the policy improves and promotions happen, the disc unsaturates naturally. The entropy coefficient doesn't need to force unsaturation — the policy's improvement does it.

---

## Q2: Is discriminator cadence (2× vs 1×) important?

**Context:** The sbatch comment says "disc fires every 500 env steps (independent of policy, 2x per policy update)" referring to the ratio — policy updates every 1000 steps, disc updates every 500. Both TF and PyTorch have this same 2:1 disc:policy update ratio.

**Answer: This ratio is preserved in PyTorch.** No change was made to disc_train_freq. The "TFDisc" name refers only to the entcoeff change. The cadence is already identical between TF and PyTorch.

---

## Q3: Is TF-style entcoeff=0.01 better, worse, or equivalent?

**Answer: Equivalent, with minor seed-level variance.**

Both ec values achieve stable discriminators and comparable final performance. The primary concern documented in CLAUDE.md — that ec=0.01 causes permanent saturation — is **not confirmed at 1M steps with LfD mixing enabled**. LfD mixing provides diverse signal early (expert+policy mixed in batch), which helps prevent permanent saturation even at low entcoeff.

**CLAUDE.md note about saturation:** The documented saturation with ec=0.01 may have occurred in earlier runs **without LfD mixing** (vanilla SAIL). With `--lfd_mixing`, the early mixed batches provide enough expert signal to keep the discriminator learning even at ec=0.01.

---

## Q4: Does TF Disc Parity (ec=0.01) affect AdaptPref?

**Answer: Yes — it delays promotion more severely than ec=0.05.**

| Config | First promotion | ep_rew at 920k |
|--------|-----------------|----------------|
| AdaptPref ec=0.05 seed=1 | 432k | 7,540 (at 1M) |
| AdaptPref ec=0.01 seed=1 | **575k** | 7,760 |

With ec=0.01, the disc saturates more strongly early (disc_loss=0.198–0.265 at 100–200k). The pref_loss is near-zero during saturation (0.000771 at 100k → 6.94e-05 at 500k — dropping to zero as saturation increases). This means neither GAIL nor pref signals are useful for 500k steps. The policy still learns (because LfD mixing provides some signal), but first promotion is delayed to 575k.

After the first promotion, the teacher buffer diversifies and pref_loss rises to meaningful levels (0.006 at 600k → 0.08 at 900k). By 920k, seed=1 reaches 7,760 — slightly above ec=0.05's 7,540 at 1M, suggesting the trajectory is trending up and may exceed ec=0.05 if run to 1M.

---

## pref_loss Behavior — PyTorch AdaptPref ec=0.01 vs TF PrefRank

| Step | PyTorch pref_loss (ec=0.01, s=1) | TF pref_loss (ec=0.01, s=0) |
|------|-----------------------------------|-----------------------------|
| 50k | 0.00276 | ~4e-05 |
| 100k | 0.000771 | ~5e-05 |
| 200k | 0.000681 | **0.013** (post 1st promo) |
| 300k | 0.000434 | 0.006–0.018 |
| 500k | 6.94e-05 | 0.01–0.06 |
| 600k+ | 0.006–0.12 (post promotion at 575k) | 0.01–0.11 |

**Key difference:** In TF, the first promotion at 201k causes a step-change in pref_loss from ~0 to 0.013. In PyTorch, the same happens but at 575k. Before their respective first promotions, pref_loss is near-zero in both frameworks.

The reason TF PrefRank has **slightly earlier** pref_loss activation is the earlier first promotion (201k in TF vs 575k in PyTorch ec=0.01). This earlier activation is seed-driven, not PrefRank-driven (same threshold bug 6932, TF seed=0 may be luckier).

---

## Practical Takeaways

1. **entcoeff=0.01 is fine for Adaptive SAIL with LfD mixing.** No need to use 0.05 for the pure adaptive setting. Both achieve 8,290–8,830 at 1M.

2. **entcoeff=0.01 delays AdaptPref promotion by ~143k steps vs ec=0.05** (575k vs 432k). If using PrefRank, ec=0.05 is slightly less harmful.

3. **The TF/PyTorch gap is NOT resolved by matching entcoeff.** Even at ec=0.01, PyTorch achieves 8,460–8,830 vs TF's 6,548. The gap is driven by TF's 1,000-transition ring buffer vs PyTorch's unlimited teacher buffer.

4. **Do NOT revert CLAUDE.md recommendation to ec=0.01 prematurely.** While ec=0.01 performs well in these runs, ec=0.05 was the empirically confirmed stable default. The marginal performance difference does not justify removing the safety margin against saturation.
