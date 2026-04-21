# PrefRank vs Soft-TAC vs QPREF — Log-Based Analysis
*Date: 2026-04-10 | Runs analyzed: 3 seeds across 3 loss variants*

---

## Executive Summary

**PrefRank wins by a large margin (~1200 ep_rew)** because it generates meaningful discriminator gradient throughout training. Soft-TAC silences itself within 1000 steps (tac_alignment=1.0, loss≈0) due to expert-pool homogeneity (3% RM spread across 4 expert episodes). QPREF starts with a useful positive delta (+0.55 at 35k) but inverts to strongly negative values after 400k steps as Q-value Bellman noise swamps the tiny RM score differences between expert episodes.

| Metric at 1M steps | PrefRank (RM+Pref) | Soft-TAC (Adaptive) | QPREF (Adaptive) |
|---|---|---|---|
| ep_rew_mean | **7810** | 6620 | 6550 |
| disc_loss | 1.30 | 0.97 | ~0.85 |
| Pref/TAC/QPREF loss | 4.08 (pref) | 0.25 (tac) | 2.47 (qpref) |
| Alignment signal active | Most of run | ~13% of run | Inverted after 400k |
| Effective loss contribution | Load-bearing | Near-zero until 500k | Counterproductive after 400k |

---

## Log 1: PrefRank (HC_RMAdaptPref_LfD_46818417_2)

**Config:** RM-adaptive, `--pref_rank_disc`, entcoeff=0.01, no soft_tac, no qpref.
Old-format run: starts with 1 expert episode in ring buffer (not SepPool).

| Step | ep_rew | disc_loss | pref_loss | surr_rew | n_pref_eps | n_promotions |
|------|--------|-----------|-----------|----------|------------|--------------|
| 10k | −447 | N/A | N/A | N/A | 1 | 0 |
| 50k | −387 | 0.34 | N/A | 1.13 | 1 | 0 |
| 100k | 310 | 0.38 | N/A | 1.13 | 1 | 0 |
| 200k | 4930 | 0.81 | N/A | 0.87 | 1 | 0 |
| 300k | 6560 | 1.01 | N/A | 0.77 | 1 | 0 |
| 306k | — | — | **FIRST** | — | 2 | 1 |
| 400k | 6980 | 1.09 | 0.14 | 0.57 | ~10 | ~10 |
| 500k | 7180 | 1.12 | 0.23 | 0.55 | ~50 | ~50 |
| 600k | 7500 | 1.11 | 0.69 | 0.51 | ~80 | ~80 |
| 700k | 7840 | 0.90 | 0.24 | 0.46 | ~100 | ~100 |
| 800k | 7800 | 0.93 | 0.53 | 0.40 | ~108 | ~108 |
| 900k | 7850 | 0.81 | 0.96 | 0.37 | ~111 | ~111 |
| 1000k | 7810 | 1.30 | **4.08** | 0.35 | ~111 | ~111 |

**Key observations:**
- pref_loss inactive for first 300k steps (only 1 episode in ring buffer → ValueError in `sample_pref_pairs`)
- First promotion at 306k → N=2 → pref_loss first appears (~0.14)
- Meaningful pref_loss (0.14–0.69) active from ~400k–700k. This is the phase where ep_rew grows from 6980→7840.
- ep_rew continues improving past the no-pref plateau of 6600–6700 precisely because pref_loss provides ongoing discriminator signal.
- High late pref_loss (4.08 at 1M) reflects large, diverse pool of promoted student episodes — discriminator must rank across a wide quality spectrum.

---

## Log 2: Soft-TAC (HC_SoftTAC_Adaptive_47250637_2)

**Config:** RM-adaptive (`--adaptive_score_source rm`), `--soft_tac`, soft_tac_weight=0.5, soft_tac_temp=1.0, tac_tie_eps=0.0.
SepPool: ALL 4 expert episodes loaded before ring truncation.
Expert RM scores: [1333.0, 1340.8, 1357.4, 1374.9] → mean=1353.5, spread=41.8 (**3.1% variation**).

| Step | ep_rew | disc_loss | soft_tac_loss | tac_alignment | surr_rew | Notes |
|------|--------|-----------|---------------|---------------|----------|-------|
| 12k | −390 | 2.57 | 0.057 | 0.887 | N/A | Warmup — disc still learning |
| 13k | −386 | 2.31 | **2.3e−7** | **1.000** | N/A | **SATURATED — loss dead** |
| 50k | −532 | 0.45 | 3.0e−7 | 1.000 | 1.13 | |
| 100k | −490 | 0.21 | 3.0e−5 | 1.000 | 1.12 | Disc learns to assign high reward to all 4 |
| 200k | 2390 | 0.50 | ~0 | 1.000 | 1.01 | |
| 300k | 5900 | 0.78 | ~0 | 1.000 | 0.93 | |
| 400k | 6690 | 0.78 | ~0 | 1.000 | 0.82 | Plateau — soft_tac still dead |
| 500k | 6740 | 0.79 | 0.041 | 0.919 | 0.74 | First promotions → pool diversifying |
| 600k | 6810 | 0.88 | 0.222 | 0.556 | 0.65 | Promoted students in pool |
| 700k | 6810 | 0.93 | 0.23 | 0.535 | 0.60 | |
| 800k | 6800 | 0.95 | 0.31 | 0.380 | 0.56 | |
| 900k | 6750 | 0.96 | 0.29 | 0.421 | 0.53 | |
| 1000k | 6620 | 0.97 | 0.25 | 0.500 | 0.52 | Alignment oscillating, no improvement |

**Key observations:**
- tac_alignment hits 1.0 at step 13k and stays there until ~480k. This means `1 - mean(y * tanh(delta_J / T)) ≈ 1 - 1 = 0`.
- Soft_tac_loss ≈ 0 for the first ~87% of training (steps 13k–480k out of 1M).
- When loss eventually activates (500k+), ep_rew is already plateaued at 6740. The late soft_tac signal (alignment 0.5–0.9) cannot lift the policy further.
- Final ep_rew=6620 is **1190 lower than PrefRank**.

---

## Log 3: QPREF (HC_QPREF_Adapt_LfD_47225856_2)

**Config:** GT-adaptive (`--adaptive_score_source gt` — not RM), `--qpref`, qpref_weight=0.1, qpref_temp=1.0, qpref_batch=4, qpref_grad_interval=10.
4 expert episodes (same 3% RM spread as Soft-TAC run).

| Step | ep_rew | disc_loss | qpref_loss | qpref_delta | Q_pos | Q_neg | Notes |
|------|--------|-----------|------------|-------------|-------|-------|-------|
| 12k | −421 | 2.73 | 0.694 | −0.003 | 1.5 | 1.5 | Near-random Q values |
| 20k | −290 | 2.45 | 0.528 | **+0.374** | 30.1 | 29.7 | Q rising, delta positive |
| 35k | −360 | 2.41 | 0.456 | **+0.550** | 55.2 | 54.7 | Peak positive delta |
| 100k | −298 | 0.315 | 0.297 | +1.20 | 82.8 | 81.6 | Delta positive, Q ≈ 82 |
| 200k | 4700 | 0.598 | 0.314 | +1.13 | — | — | Still healthy |
| 300k | 6580 | 0.755 | 0.711 | +0.105 | — | — | Delta collapsing |
| 400k | 6590 | 0.800 | **1.24** | **−0.728** | — | — | **INVERTED — delta negative** |
| 500k | 6600 | 0.798 | 1.41 | −0.728 | — | — | Plateau |
| 600k | 6580 | 0.815 | 1.51 | −0.783 | — | — | |
| 700k | 6580 | 0.810 | 1.50 | −0.684 | — | — | |
| 900k | 6550 | 0.847 | 1.75 | −1.05 | — | — | Delta worsening |
| 1000k | 6550 | 0.841 | **2.47** | **−1.95** | — | — | Signal fully inverted |

**Key observations:**
- QPREF starts positive and meaningful (+0.374 at 20k, +0.55 at 35k). Expert episodes with higher RM scores have slightly higher Q_pos than Q_neg.
- After 300k steps, Q values for expert episodes converge as the critic learns from surrogate rewards. The 3% RM spread (41.8 mean Q units out of 1353) becomes negligible compared to Bellman update noise.
- Delta sign-flips at ~350–400k: Q_pos < Q_neg for sampled pairs. The loss now has softplus(−(Q_pos−Q_neg)) = softplus(positive), pushing the critic to DECREASE Q_pos relative to Q_neg — the opposite of the intended preference ranking.
- qpref_loss grows monotonically (0.694→2.47) as delta worsens. The critic is actively receiving counterproductive gradient.
- ep_rew plateaus at ~6590 from 400k onward — no improvement in the final 600k steps.

---

## Mechanistic Analysis: Why PrefRank Works and the Others Don't

### 3.1 Pool Composition — The Root Difference

| Loss | Pool at training start | Pool at 500k | Pool diversity |
|------|----------------------|--------------|----------------|
| PrefRank | 1 expert episode | ~50 (1 expert + ~49 promoted students) | HIGH — student quality spans 0→7000+ |
| Soft-TAC | 4 expert episodes | 4 expert + promoted students (late) | LOW early, grows late |
| QPREF | 4 expert episodes | 4 expert only (or + early promotions) | LOW — 3% RM spread |

**PrefRank** in the old-format run has exactly 1 expert episode before first promotion. After the first promotion (306k), it immediately gets a (expert, promoted_student) pair spanning a wide quality gap. The softplus BT loss `mean(softplus(-(J_disc_pos - J_disc_neg)))` on a (6988-return expert, first-promoted student) pair provides strong gradient.

**Soft-TAC** starts with 4 expert episodes, all scoring within 3% of each other on the RM. The disc quickly assigns high `J_disc` to all 4 episodes (they all look "expert" to the disc). When sampling pairs: J_disc_pos ≈ J_disc_neg → delta_J ≈ 0 → tanh(0/1.0) = 0 → y*tanh ≈ 0 for tied pairs, 1 for labeled pairs. But since ALL pairs have near-identical RM scores, they're all labeled +1 (y=+1 because diff > 0.0 threshold for one direction) or 0. After 13k steps, disc has learned that ALL pairs have J_disc_pos > J_disc_neg (it's right on expert-expert pairs due to random variation), so tanh → 1 → y*tanh = y = 1 → alignment = 1 → loss = 0.

### 3.2 Loss Gradient Geometry

**PrefRank (softplus BT):**
```
gradient = sigmoid(-(J_pos - J_neg)) * (-1)
```
When J_pos < J_neg (disc disagrees with preference), sigmoid ≈ 1 → gradient ≈ -1 (strong correction).
When J_pos > J_neg (disc agrees), sigmoid → 0 → gradient → 0 (gentle).
This asymmetric gradient keeps the disc continuously corrected on hard pairs (where it's currently wrong).

**Soft-TAC (tanh):**
```
gradient ∝ y * (1 - tanh²(delta_J / T)) * (1/T)
```
When |delta_J| is large (disc already correctly ranks the pair), `tanh(delta_J) ≈ ±1` → `(1 - tanh²) ≈ 0` → **near-zero gradient**. This is the designed saturation for "easy" pairs. But with a homogeneous expert pool where ALL pairs are easy (disc correctly ranks all 4 expert episodes), ALL pairs contribute near-zero gradient. Tanh specifically kills gradient on solved pairs. With a 3% quality spread, pairs are "solved" after 13k steps.

**QPREF (softplus on Q-values):**
```
gradient = sigmoid(-(Q_pos - Q_neg)/T) * (critic gradient)
```
Structurally similar to PrefRank but on Q-values instead of disc returns. The problem is that Q-values are not directly trained to reflect RM scores — they're trained via TD Bellman updates with surrogate rewards. The RM score differences (41.8 units) are below the noise floor of Bellman bootstrap errors once Q-values reach 80+.

### 3.3 Why Soft-TAC Is Designed for a Different Setting

Soft-TAC in the TF codebase was designed with the assumption that the preference pool contains trajectories spanning a wide quality range (student rollouts from early training mixed with expert). In that setting, pairs like (good_expert, bad_student) have large delta_J, and the tanh steers the disc without overconfident BT. In the current PyTorch run, the SepPool design pre-loads ALL expert episodes BEFORE ring truncation — so the pool at step 0 contains only 4 near-identical expert episodes. Soft-TAC is not designed to handle a homogeneous pool.

### 3.4 Timeline Comparison

```
Steps:     0        100k     200k     300k     400k     500k     600k    700k    1M
           |--------|--------|--------|--------|--------|--------|--------|-------|
PrefRank:  [dead:0 episodes]         [active:diverse pool grows from 306k]  →7810
Soft-TAC:  [dead:aligned=1.0 from 13k]                   [weak:0.5 align] → 6620
QPREF:     [OK:delta+]    [positive delta]   [INVERTED delta: loss hurts]  → 6550
```

---

## Diagnosis: Ranked Root Causes

### Soft-TAC

**Rank 1 — Expert pool too homogeneous (fatal, active from step 13k):**
- RM scores of 4 expert episodes span only 41.8 units (3.1% of mean 1353.5)
- Disc needs only 13k steps to correctly rank all 4 expert episodes
- tac_alignment = 1.0 → soft_tac_loss ≈ 0 for 87% of training
- No gradient contribution to discriminator during the critical 50k–400k learning phase

**Rank 2 — Pool diversity arrives too late (structural):**
- Soft-TAC with SepPool (4 expert episodes) has no student diversity until first promotions (~480k in this run)
- By the time promoted students enter the pool and diversify it, ep_rew is already plateaued at 6740
- The 500k–1M activation of soft_tac_loss finds a disc already locked into a suboptimal equilibrium

**Rank 3 — Tanh gradient vanishes on solved pairs (by design, harmful here):**
- Tanh's zero-gradient-on-easy-pairs property works correctly in TF's mixed pool (skip the easy pairs, focus on hard ones)
- In a homogeneous expert pool, ALL pairs are "easy" after 13k steps → tanh kills all gradient

**Rank 4 — tac_tie_eps=0.0 causes over-labeling (minor):**
- All expert-expert pairs with any RM difference receive y=+1 or y=-1
- Near-zero differences get hard labels → disc "solves" them trivially → contributes to premature saturation

### QPREF

**Rank 1 — Q-value noise swamps RM score differences after 300k (fatal, causes counterproductive gradient):**
- Expert RM score differences: ~41.8 units
- Q values reach 80+ by 100k steps. Bellman bootstrap noise on a batch of 256 transitions with surrogate rewards can easily span ±5–10 Q units.
- The mean-Q difference for the 4 expert episodes (sampled with batch=4) loses the RM signal to noise after ~350k steps
- delta flips negative at 400k → qpref_loss is actively pushing Q_pos DOWN relative to Q_neg → **counterproductive**

**Rank 2 — Expert pool too homogeneous (same as Soft-TAC):**
- Using only 4 expert episodes with 3% spread for the entire 1M run provides insufficient quality signal
- No promoted student episodes join the QPREF pool (only `--qpref_source teacher` is active)

**Rank 3 — GT-adaptive promotion mask (independent issue):**
- This run uses `--adaptive_score_source gt`, not rm. GT thresholds are different calibration.
- Not the root cause of the delta inversion but may affect promotion timing.

**Rank 4 — qpref_grad_interval=10 fires too frequently (minor):**
- Every 10 critic steps means ~100 QPREF updates per 1000 disc steps
- High frequency means noisy delta estimates have more opportunities to accumulate counterproductive gradient

---

## Concrete Improvement Plans

### Soft-TAC Fixes

#### Fix A: Mixed pool at initialization (high impact, requires `--include_student_pairs` flag)
Instead of loading only expert episodes at SepPool initialization, add the **first N=10 random rollout episodes** from the replay buffer to the pool at `learning_starts`. These will span ep_rew ≈ -600 to -200 (HalfCheetah random policy), providing quality contrast to expert episodes (RM score ~1353 vs ~−150).

**Expected effect:** Pool diversity from step 0 → tanh active from step 0 → disc learns discriminative rewards earlier.

**Implementation:** In `_build_pref_episodes()`, after loading expert episodes, if `soft_tac=True`, register a callback to add the first 10 random rollout episodes when they become available (after `learning_starts`).

#### Fix B: Lower tac_tie_eps to reduce over-labeling (quick, hyperparameter-only)
Increase tie zone: `--tac_tie_eps 20.0` (units: RM score units). This labels as y=0 (tied) all pairs where RM difference < 20 RM units, keeping y=+1/−1 only for clearly distinct pairs (>20 RM units apart).

**Expected effect:** Reduces the "trivially solved" labeled pairs. Forces the tanh to only fire on pairs with actual quality contrast. Would reduce the false y=+1 labels on near-identical expert episodes.

**Tradeoff:** With 4 expert episodes spanning 41.8 units, ALL pairs would be tied (difference < 20). The loss would receive y=0 everywhere → zero gradient. This is actually correct behavior (don't pretend you can rank near-identical episodes) but means Soft-TAC provides zero guidance until diverse promotions appear.

**Verdict:** Correct but insufficient on its own. Needs Fix A to pair with it.

#### Fix C: Use `pref_promote_quantile` promoted students in Soft-TAC pool (medium impact)
Add promoted student episodes to the Soft-TAC pool (currently only expert episodes and their RM scores are in the pool). When a student is promoted, add it to `pref_episodes` with its RM score. This creates (expert, promoted_student) pairs with large RM score differences.

**Implementation:** Already done for PrefRank (both use `pref_episodes`). The Soft-TAC loss already samples from `pref_episodes`. The issue is timing — this fix is already structurally in place, but promotions arrive late (~480k). Fix A would make this earlier.

#### Fix D: Soft-TAC temperature annealing (experimental)
Start with `--soft_tac_temp 0.1` (hard tanh, steep gradient). Anneal to `soft_tac_temp=1.0` over 300k steps. Steep tanh means tanh(delta_J/0.1) = ±1 for even tiny delta_J → stronger gradient on easy pairs during warmup. Reduces dead loss period.

**Risk:** May overcorrect on noise. Needs careful tuning.

### QPREF Fixes

#### Fix A: Add `--qpref_source mixed` with promoted student episodes (high impact)
The student episodes with RM scores should be added to the QPREF pool as they're promoted. A (high_rm_expert, recent_promoted_student) pair has a **large Q_pos − Q_neg** because the student episode was recently trained with surrogate rewards and has higher Q estimates. This reverses the noise problem.

**Implementation:** Add a `qpref_source` option that includes promoted student episodes alongside expert episodes. The pool grows over time, providing increasingly diverse pairs.

#### Fix B: Stop QPREF when delta is consistently negative (safety guard, quick)
Add a running average of `qpref_delta` over the last 100 QPREF updates. If `running_avg_delta < −0.5`, disable QPREF updates until it recovers. Log a warning.

**Implementation:** Add `self._qpref_delta_ema: float = 0.0` updated as `0.9 * ema + 0.1 * delta`. Skip QPREF loss if `ema < self.qpref_disable_threshold` (default −0.5).

**Expected effect:** Prevents the counterproductive gradient after 400k. The run effectively becomes "QPREF active early, disabled late" — similar to learning-rate warmup/cooldown.

#### Fix C: Increase `--qpref_grad_interval` to 50 (hyperparameter-only, quick)
Reduce QPREF firing rate from every 10 steps to every 50. This reduces the frequency of counterproductive gradient accumulation and also reduces the noise sensitivity (each QPREF update is now ~5x more "considered").

**Expected effect:** Delta sign flip delayed or attenuated. The positive delta phase (0–300k) has fewer QPREF updates total, so the noise accumulates slower.

#### Fix D: Use `--qpref_source rm` scores directly (architecture change, removes Q dependency)
Instead of using TD3 Q-values as the preference signal, use the RM scalar rewards directly in a QPREF-style loss: `mean(softplus(-(rm_pos - rm_neg) / T))`. This removes the Q-value noise problem entirely.

**Note:** This is essentially the `pref_rm_loss` in TF SAIL (a third loss distinct from pref_rank_disc and soft_tac). It was not implemented in PyTorch. This would be a separate loss, not QPREF. Consider implementing as `--pref_rm_rank` flag.

---

## Recommended Next Experiment

**Best immediate experiment: Soft-TAC + PrefRank together (not instead of)**

Both losses are structurally compatible and complementary:
- PrefRank provides continuous BT signal on diverse (expert, student) pairs → strong during 300k–1M
- Soft-TAC with the current implementation would add near-zero loss during 0–480k but could add alignment signal in 500k+

**Proposed sbatch flags:**
```bash
--soft_tac
--soft_tac_weight 0.3         # lower weight (PrefRank is the primary loss)
--soft_tac_temp 1.0
--tac_tie_eps 15.0             # exclude near-tied expert-expert pairs
--pref_rank_disc               # PrefRank is the primary disc alignment
--pref_rank_weight 1.0
--pref_promote_quantile 0.5
--adaptive_score_source rm
```

**Alternatively, if the goal is to make Soft-TAC work standalone:**

Run Soft-TAC with a **wider expert pool** using a different expert dataset with higher quality spread (>20% RM variation instead of 3%). The current HalfCheetah expert episodes are too similar. Or: use the `--soft_tac` flag with `--pref_reweight_teacher` which will add promoted student episodes to the pool at first promotion, providing diversity similar to what PrefRank gets naturally.

**For QPREF:** The most actionable fix is Fix B (disable when delta < −0.5) combined with Fix C (grad_interval=50). These can be done in a single 2-line code change and would likely yield a cleaner signal in the 0–300k window without the 400k–1M counterproductive phase.

---

## Summary Table

| Issue | Soft-TAC | QPREF |
|-------|----------|-------|
| Root cause | Pool homogeneity (3% RM spread → tac_align=1.0 from 13k) | Q-value noise > RM score differences after 300k |
| Loss dead? | 87% of training (13k–480k) | After 400k (inverts sign) |
| Recoverable? | Yes — with pool diversity (Fix A or mixed pool) | Partially — disable guard + lower freq |
| Currently helping? | Only 500k–1M, weakly | Hurting after 400k |
| Performance gap vs PrefRank | −1190 ep_rew at 1M | −1260 ep_rew at 1M |
| Best 1-line fix | `--tac_tie_eps 15.0` (reduces false easy-pair labeling) | `--qpref_grad_interval 50` + delta guard |
| Best structural fix | Mixed pool at init or PrefRank combination | Add promoted students to QPREF pool |
