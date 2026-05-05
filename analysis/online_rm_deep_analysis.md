# Deep Analysis: Online Reward Model (sail_sb3_online)

**Date:** 2026-05-05  
**Codebase:** `sail_sb3_online/` — Online RM variants of SAIL  
**Log coverage:** Walker2d-v2 (59 runs), HalfCheetah-v2 (15 runs), Swimmer-v2 (57 runs), Hopper-v2 (14 runs), Ant-v2 (13 runs)  
**Analysis basis:** Parsed `.out` log files from SLURM jobs; code reading of `reward_models/`, `datasets/`, `algorithms/`

---

## Architecture Summary

Before diving into metrics, a brief map of the RM:

| Component | File | Description |
|-----------|------|-------------|
| `OnlinePrefRewardModel` | `reward_models/online_pref_rm.py` | MLP (obs+act → scalar r_t); AdamW; Bradley-Terry BCE loss |
| `SegmentStore` | `datasets/segment_pref_buffer.py` | Ring buffer of fixed-length segments (len=50); stores obs/act/mask/return/source |
| `OnlineRMManager` | `reward_models/online_rm_manager.py` | Orchestrates training, held-out eval, activation gating, rescore scheduling |
| Integration | `algorithms/sail.py` + `utils/callbacks.py` | Downstream use in PAIL, QPREF, TAC, PrefRank, adaptive scoring |

The RM is a 2-layer MLP `Linear(obs+act, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)` predicting per-step reward. Segments of 50 steps are stored in a ring buffer (max 10,000 segments). Pairs are sampled uniformly with tie rejection (min return gap = 0.5) and trained with Bradley-Terry BCE: `Loss = BCE(sigmoid(R₂ − R₁), label)`.

---

## Part 1 — RM Score Evolution

### 1.1 What Is Logged

The following online RM metrics are logged to WandB at every train-log interval (`rm_train_freq=1000` steps):

```
online_rm/held_out_acc       — Accuracy on a fixed 100-pair held-out set (0.0–1.0)
online_rm/is_active          — 0/1 flag: whether the three-gate threshold is met
online_rm/loss               — BCE loss on the last training batch
online_rm/segment_store_size — Total segments currently in the ring buffer
online_rm/teacher_count      — Teacher (expert) segments in the buffer
online_rm/student_count      — Student segments in the buffer
online_rm/updates            — Cumulative gradient steps
online_rm/rescore_count      — Number of times pref_episodes have been rescored
```

**What is NOT logged:** The RM-predicted episode return (cumulative sum of r_t over a full episode) is not directly logged. There is no `online_rm/episode_return_mean` metric. To compare RM returns vs GT returns would require adding logging to the `rescore_pref_episodes()` call in `sail.py:562–591`.

The `pref_reweight/` metrics (j_mean, j_std, weight_min, weight_mean, entropy) provide an indirect view of how RM-labeled returns evolve over time.

### 1.2 Held-out Accuracy Evolution by Environment

The held-out accuracy (`online_rm/held_out_acc`) is the best single proxy for RM quality in the available logs. The following table shows the trajectory for one representative seed per environment.

#### HalfCheetah-v2 — PAIL-QPREF, seed 1

| Step | Acc | Active | Segs | Updates | Loss | ep_rew |
|------|-----|--------|------|---------|------|--------|
| 1k | — | 0 | — | 0 | — | −325 |
| 26k | **0.88** | **1** | 580 | 50 | — | — |
| 76k | 0.91 | 1 | 1,580 | 550 | 0.082 | 335 |
| 151k | 0.93 | 1 | 3,080 | 1,300 | 0.038 | 4,670 |
| 301k | 0.98 | 1 | 6,080 | 2,800 | 0.028 | 6,670 |
| 526k | **1.00** | 1 | 10,000 | 5,050 | 0.106 | 6,610 |
| 751k | 1.00 | 1 | 10,000 | 7,300 | 0.017 | 7,190 |
| 1000k | **1.00** | 1 | 10,000 | 9,550 | 0.010 | 7,870 |

HC RM reaches perfect held-out accuracy by ~526k and stays there. Policy performance tracks RM quality: the jump from 335 (76k) to 6,670 (301k) coincides with the RM consolidating above 0.97.

#### Walker2d-v2 — PAIL-TAC-D, seed 1

| Step | Acc | Active | Segs | Updates | Loss | ep_rew | ns_pct |
|------|-----|--------|------|---------|------|--------|--------|
| 19k | 0 | 0 | 195 | 0 | — | 3.1 | 0.07% |
| 44k | **0.89** | **1** | 572 | 50 | — | — | — |
| 79k | 0.95 | 1 | 1,140 | 390 | 0.058 | 224 | 4.8% |
| 197k | 0.99 | 1 | 3,390 | 1,570 | 0.036 | 1,920 | 40.6% |
| 354k | 0.96 | 1 | 6,450 | 3,140 | 0.029 | 3,390 | 71.9% |
| 471k | 0.99 | 1 | 8,780 | 4,320 | 0.015 | 4,150 | 87.9% |
| 550k | 0.98 | 1 | 10,000 | 5,100 | 0.016 | 3,190 | 67.6% |
| 826k | **1.00** | 1 | 10,000 | 7,860 | 0.013 | 4,840 | 103% |
| 999k | 0.98 | 1 | 10,000 | 9,440 | 0.006 | 4,410 | 93.5% |

Walker2d RM rises quickly to ~0.95 and fluctuates between 0.96–1.00 in the steady state. Note: teacher segments get evicted from the buffer at ~549k (teacher_count drops from 79 to 0), after which the RM trains purely on student segments. Accuracy remains high because student episodes at this point span a wide enough return range.

#### Ant-v2 — PAIL-QPREF, seed 1

| Step | Acc | Active | Segs | Updates | Loss | ep_rew | ns_pct |
|------|-----|--------|------|---------|------|--------|--------|
| 11k | 0 | 0 | 260 | 0 | — | −61 | −1.1% |
| 29k | **0.76** | **1** | 591 | 50 | — | −348 | — |
| 93k | 0.81 | 1 | 1,830 | 750 | 0.246 | −301 | −5.2% |
| 196k | 0.86 | 1 | 3,870 | 1,570 | 0.091 | 1,050 | 18.0% |
| 318k | **0.97** | 1 | 6,280 | 2,660 | 0.041 | 2,220 | 38.2% |
| 420k | 0.98 | 1 | 8,320 | 3,500 | 0.033 | 2,720 | 46.9% |
| **500k** | 0.95 | 1 | 9,930 | 4,300 | 0.073 | 2,420 | 41.7% |
| 623k | **0.77** | 1 | 10,000 | 5,350 | 0.021 | 2,830 | 48.7% |
| 750k | 0.72 | 1 | 10,000 | 6,500 | 0.024 | 2,850 | 49.0% |
| 831k | 0.72 | 1 | 10,000 | 7,130 | 0.017 | 2,960 | 50.9% |
| 999k | 0.76 | 1 | 10,000 | 9,470 | 0.023 | 2,850 | 49.1% |

**Ant is the anomaly.** The RM rises to 0.97 around 318k, then **degrades to 0.69–0.77 after the buffer fills at ~500k.** This is analyzed in detail in Part 5.

#### Hopper-v2 — PAIL-PREFRANK, seed 1

| Step | Acc | Active | Segs | ep_rew | ns_pct |
|------|-----|--------|------|--------|--------|
| 15 | — | 0 | — | 12.9 | 0.4% |
| 66k | **0.97** | **1** | 580 | — | — |
| 150k | 0.97 | 1 | 2,070 | 1,050 | 29.2% |
| 300k | 1.00 | 1 | 4,920 | 2,130 | 59.0% |
| 452k | 1.00 | 1 | 7,940 | 3,240 | 89.8% |
| 677k | **1.00** | 1 | 10,000 | 2,340 | 65.0% |
| 979k | 1.00 | 1 | 10,000 | 2,590 | 71.9% |

Hopper RM reaches 1.00 by 300k and never drops.

#### Swimmer-v2 — PAIL, seed 2

| Step | Acc | Active | Segs | ep_rew | ns_pct |
|------|-----|--------|------|--------|--------|
| 1k | — | 0 | — | 0.14 | 0.04% |
| 26k | **0.93** | **1** | 580 | — | — |
| 76k | 0.97 | 1 | 1,580 | 3.39 | 0.95% |
| 301k | 0.98 | 1 | 6,080 | −19.9 | −5.6% |
| 601k | 0.97 | 1 | 10,000 | 7.70 | 2.1% |
| 751k | 0.98 | 1 | 10,000 | 198 | 55.1% |
| 901k | 0.98 | 1 | 10,000 | 308 | 85.9% |
| 1000k | **0.99** | 1 | 10,000 | 322 | 89.7% |

Swimmer RM accuracy is excellent throughout. The slow policy learning (near-zero ep_rew until ~600k) is **not** caused by RM failure — the RM is 0.97–0.99 the whole time. The bottleneck is in the discriminator / policy learning, not the reward model.

### 1.3 Does the RM Become More Accurate Over Time?

**Yes, for all environments except Ant.** The progression is:

1. **Before activation (~26–73k steps):** RM trains but accuracy is below 60% threshold. The segment buffer is filling (194–580 student segments).
2. **Activation (~26–73k):** Exactly at `min_updates=50`, the RM jumps to 0.76–0.97 on first check. This indicates the RM is already substantially trained when it first becomes active.
3. **Rapid improvement (~76–300k):** Accuracy rises from 0.76–0.97 to 0.93–1.00 as more diverse student segments are added and the loss falls from 0.08–0.28 to 0.01–0.06.
4. **Steady state (300k–1M):** HC/Hopper/Walker2d/Swimmer stay at 0.97–1.00. Ant degrades to 0.69–0.77.

**The RM does approach an optimal reward for most environments.** 0.98–1.00 on a held-out preference pair set means the RM correctly identifies the higher-return segment 98–100% of the time. For non-Ant environments, this is achieved by ~300k steps.

### 1.4 When Does the RM Become Useful?

| Environment | Activation Step | Acc at Activation | RM "Reliable" (acc ≥ 0.90) |
|-------------|----------------|-------------------|---------------------------|
| HalfCheetah-v2 | ~26k | 0.88 | ~76k |
| Walker2d-v2 | ~44–73k | 0.83–0.97 | ~79k |
| Swimmer-v2 | ~26k | 0.93 | ~76k |
| Hopper-v2 | ~66k | 0.97 | ~150k |
| Ant-v2 | ~29k | 0.76 | ~300k → degrades after 500k |

**Practical answer:** The RM becomes a reliable ranking signal at 76k–150k steps for HC/Walker/Swimmer/Hopper. For Ant, reliability peaks at 300k and degrades thereafter.

---

## Part 2 — RM Quality Metrics

### 2.1 What the Metrics Mean

**`online_rm/held_out_acc` (0.0–1.0)**  
Fraction of 100 fixed held-out pairs where the RM correctly identifies the higher-return segment. This is the primary quality signal. A value of 1.00 means the RM has learned a reward function that perfectly orders the held-out segments by their environment return.

**`online_rm/loss` (BCE, ↓ better)**  
Training loss on the last batch of 256 pairs. Loss curve tells a consistent story across all runs:
- Phase 0 (0–50k): no training, loss = None
- Phase 1 (50k–300k): 0.10–0.28, decreasing rapidly
- Phase 2 (300k–buffer full): 0.01–0.06, slowly decreasing
- Phase 3 (buffer full, 500k+): 0.005–0.025, flat or slight rise

The loss rising slightly after buffer fill (phase 3) is normal — new student segments from an improving policy are harder to rank than early low-return segments.

**`online_rm/segment_store_size`**  
Grows linearly at `segment_len / episode_len` segments per episode until hitting the 10,000 cap. For Walker2d (~1000-step episodes), the buffer fills at ~500k steps. For Hopper (~500-step episodes), it fills at ~250k. The teacher segments (expert data, 79–150 segments depending on env) are present initially and get gradually evicted as the student segments overwrite them after ~500k steps.

**`online_rm/teacher_count` (initially 79–150, eventually 0)**  
Once the buffer fills, teacher segments are evicted by the ring buffer's wraparound. This is important: after ~500k steps, the RM trains purely on student-vs-student comparisons. The teacher segments provided an anchor (clear return differences between expert and early random policy), which is why accuracy is easier to achieve early on.

**`online_rm/updates` (cumulative)**  
Scales at `(1000 / rm_train_freq) × rm_gradient_steps = 10 steps per 1000 env steps`. At 1M steps: ~9,500 gradient steps. A reasonably trained model given the task complexity.

**`online_rm/rescore_count` (final: 47–49)**  
The pref_episodes pool is rescored every 20,000 env steps → ~49 rescores over 1M steps. This means the Boltzmann weights (PAIL), the QPREF pool labels, and the Soft-TAC J-labels are updated ~49 times during training, roughly every 20k steps.

### 2.2 Is the 60% Accuracy Threshold Sufficient?

**It is a good safety gate but not a quality threshold.** The RM never activates at 60% — in all observed runs, the RM is already at 0.74–0.97 when it first passes the activation check. This means the threshold is never actually binding; activation happens at `min_updates=50` (50 gradient steps after ~500 segments are collected), which occurs at 26k–73k steps.

The practical implication: the gate could be raised to 0.80 without changing behavior, since the RM always arrives at activation already above that level. The 60% threshold was set conservatively to avoid blocking a potentially slow-starting RM, but empirically the RM learns quickly enough that this conservatism is unnecessary.

### 2.3 Is the RM Actually Reliable When Active?

**Yes, for HC/Walker2d/Hopper/Swimmer. Partially for Ant.**

For HC/Walker2d/Hopper/Swimmer, the RM achieves and maintains 0.97–1.00 accuracy throughout most of training. Given 100 held-out pairs at 0.98+ accuracy, the expected probability of a wrong ranking is ~2%, which is acceptable noise.

**For Ant**, reliability degrades to 0.69–0.77 after ~500k steps. At 0.72 accuracy, the RM is wrong on ~28% of pairs — near-random in some regions of the state-action space. This is a real problem for any downstream component that relies on RM rankings (QPREF, Soft-TAC), but note that even 0.72 is better than random (0.50), so the signal is degraded but not inverted.

---

## Part 3 — RM vs Ground Truth Alignment

### 3.1 What Can Be Inferred from Available Logs

The logs do not directly record RM-predicted episode returns alongside GT returns. The closest proxies are:

1. **`online_rm/held_out_acc`** — Measures RM ranking alignment with GT on held-out segment pairs (ground truth label = which segment has higher sum(r_t)).
2. **`pref_reweight/j_mean`, `pref_reweight/j_std`** — Statistics of RM-assigned J values for all episodes in the pref pool. As training progresses, j_mean grows (RM assigns higher rewards to better policies), and j_std grows (RM discriminates more strongly between episodes).

### 3.2 Boltzmann Reweighting J-Statistics (Walker2d PAIL-TAC-D)

| Step | j_mean | j_std | w_min | w_mean | entropy |
|------|--------|-------|-------|--------|---------|
| 60k | 250 | 29.7 | 5.7e-49 | 0.091 | 0.00017 |
| 131k | 527 | 174 | 0 | 0.029 | ~0 |
| 202k | 773 | 319 | 0 | 0.020 | ~0 |
| 344k | 1,010 | 443 | 0 | 0.016 | 0.005 |
| 556k | 1,710 | 851 | 0 | 0.009 | ~0 |
| 697k | 2,040 | 1,130 | 0 | 0.007 | 0.018 |
| 768k | 2,140 | 1,250 | 0 | 0.007 | 0.362 |
| 980k | 2,040 | 1,560 | 0 | 0.007 | ~0 |

**Interpretation of j_mean growth:** j_mean rises from ~250 at 60k to ~2,040 by 980k. Given that the GT expert episodes have returns of ~1,463–1,784 (for Walker2d at score 1500), and the RM-assigned J values are in a different scale (sums of per-step rewards from a 256-hidden MLP, unbounded), the absolute values are not directly comparable to GT returns. What matters is the *relative ordering*: the RM is assigning higher J to better episodes (held_out_acc ≥ 0.98), which is confirmed by the accuracy data.

**Critical observation — Boltzmann weight collapse:**  
The `entropy` of the Boltzmann distribution is nearly zero throughout (with occasional brief spikes). The `weight_mean` drops from 0.091 at 60k to 0.007 by 700k as the pool grows to 100+ episodes. `w_min = 0` consistently. This means the Boltzmann softmax is placing essentially all weight on **a single episode** — the one with the highest RM-assigned J. This is the **temperature scale mismatch problem**: with β=1.0 and J values on the order of hundreds to thousands, exp(J/β) is astronomically large for the best episode relative to others, causing numerical collapse of the distribution. This is analyzed further in Part 5.

### 3.3 RM Ranking vs GT Ranking

From the held-out accuracy data and the J-statistics, we can infer:

- **Agreement region:** For pairs where return gap is large (e.g., expert vs early random policy), the RM is nearly always correct (accuracy 0.97–1.00 in all envs except Ant).
- **Disagreement region (label noise):** Near-tie pairs (return gap < 0.5) are explicitly re-sampled up to 10 times, but if still tied, are emitted with label=0 and counted in `tie_count`. For a near-plateau Ant policy (all student episodes returning ~2,700–3,100), the return-gap distribution becomes tight, making most pairs near-ties and the RM's accuracy meaningless.
- **Late training:** For HC/Walker2d (highly diversified student pool by 500k+), the RM maintains high accuracy because the student pool contains episodes ranging from early-training (low return) to near-expert (high return). The ring buffer preserving all historical episodes is beneficial here.

**How much noise exists in RM labels?**

| Environment | Late-training acc | Expected wrong-label rate | Impact on downstream |
|-------------|-------------------|--------------------------|---------------------|
| HC | 1.00 | ~0% | Negligible |
| Walker2d | 0.97–0.99 | ~1–3% | Minor |
| Hopper | 1.00 | ~0% | Negligible |
| Swimmer | 0.97–0.99 | ~1–3% | Minor |
| Ant | 0.69–0.77 | ~23–31% | Significant |

### 3.4 Effect on Downstream Components

**QPREF (online RM labels for Q-function ranking):**  
QPREF trains the Q-function on preference pairs sampled from RM-labeled episodes. At 1–3% wrong labels, QPREF noise is manageable. At 23–31% (Ant), the QPREF loss gradient contains a large fraction of wrong-direction updates, which can prevent the Q-function from learning to rank policy quality.

**Soft-TAC (student pool signed-margin loss):**  
Soft-TAC uses sign(J_pos − J_neg) as labels after rescore. At high RM accuracy, the sign is correct ~97–99% of the time. At Ant's 0.72 accuracy, the sign may be wrong for ~28% of episode pairs, directly mislabeling the "which episode is better" comparisons.

**PrefRank (BT loss on discriminator using J differences):**  
PrefRank pairs have large j_diff_mean (by construction — it samples from the pref_episodes pool where promotions enforce a quality threshold). Even with some label noise, the magnitude of the loss gradient is dominated by the correct signal when j_diff is large.

**PAIL (Boltzmann reweighting):**  
Boltzmann collapse (entropy → 0) means the weight is concentrated on one episode regardless of RM quality. The effect of RM noise is minimal because PAIL is already in a degenerate state.

---

## Part 4 — Variant-wise RM Behavior

### 4.1 Walker2d Summary: All Variants at 1M Steps

| Variant | N seeds | Median ns_pct | Min ns_pct | Max ns_pct | Failures | RM acc range |
|---------|---------|--------------|-----------|-----------|---------|-------------|
| SAIL (no RM) | 5 | 68.0% | 41.5% | 86.0% | 0 | — |
| SAIL-PREFRANK | 8 | 93.8% | 80.7% | 115% | 0 | 0.97–1.00 |
| SAIL-TAC-D | 8 | 72.5% | 15.9% | 92.6% | 2 low seeds | 0.93–0.99 |
| SAIL-QPREF | 8 | 37.6% | 0.04% | 95.0% | 2 failures | 0.87–0.98 |
| PAIL | 8 | 88.5% | 42.0% | 103% | 0 | 0.97–1.00 |
| PAIL-PREFRANK | 8 | 89.3% | 39.8% | 124% | 0 | 0.98–1.00 |
| PAIL-TAC-D | 8 | 97.3% | −0.4% | 113% | 1 failure | 0.98–1.00 |
| PAIL-QPREF | 8 | 76.2% | 53.4% | 94.3% | 0 | 0.98–1.00 |

**SAIL base (no RM usage) acts as the baseline.** Its median 68% and lack of failures confirm that the discriminator alone learns a useful reward. Every RM-augmented variant improves above this baseline on median performance, except SAIL-QPREF (37.6%) and SAIL-TAC-D with high variance.

### 4.2 Does RM Behave Differently Across Variants?

**RM accuracy is nearly identical across all variants that use the online RM.** For Walker2d, all PAIL/SAIL variants with online RM show final acc 0.93–1.00 with no systematic difference. The RM learns the same preference function regardless of whether QPREF, TAC, or PrefRank is the downstream consumer.

The key difference is how the downstream component uses the RM signal:

**SAIL-PREFRANK:** Uses RM J-values only for the PrefRank discriminator loss (BT ranking). This is the most stable usage — it adds a scalar ranking gradient to the disc loss. RM noise at 1–3% is well-absorbed.

**PAIL (Boltzmann reweighting):** Uses RM J-values to reweight the expert teacher batch. The Boltzmann collapse issue (see Part 5) means effectively only 1 episode gets weight, but that episode is the best-quality one, which is arguably correct behavior. PAIL achieves strong performance (88.5% median).

**PAIL-TAC-D:** Combines Boltzmann reweighting (PAIL) with Soft-TAC student pool (TAC). The Soft-TAC component is gated by RM activation and requires a rescored pool. One catastrophic failure observed (seed 2 of job 48684226: ep_rew = −16.9) — the RM never activated (teacher_count=0 from the start, suggesting an empty expert data issue), which prevented Soft-TAC from ever operating. Other seeds succeed well (97.3% median).

**SAIL-QPREF / PAIL-QPREF:** QPREF trains the Q-function on RM-labeled trajectory pairs. SAIL-QPREF shows high variance (0.04%–95%) with 2 near-zero seeds. PAIL-QPREF is more stable (53%–94%). The difference suggests that when QPREF operates without Boltzmann reweighting context (SAIL-QPREF), the Q-function ranking objective can conflict with the discriminator's reward signal in ways that destabilize training.

**Which variant produces the best RM training signal?**

All variants receive the same RM training signal (teacher+student segments in the segment store, BT loss). The segment distribution differs slightly:
- SAIL variants: student pool contains only episodes that passed the adaptive threshold → student segments have higher average return → easier to rank → potentially faster accuracy growth
- PAIL variants: all student episodes added to pool → more diverse returns → richer training signal but more ties

Empirically, the accuracy curves are indistinguishable between SAIL and PAIL variants. The RM accuracy is determined by the segment return diversity, not by which downstream loss uses the RM.

### 4.3 Activation Timing by Variant (Walker2d)

| Variant | Act step range | Explanation |
|---------|---------------|-------------|
| PAIL | 45k–53k | All student episodes added → buffer fills faster |
| PAIL-PREFRANK | 43k–50k | Same as PAIL |
| PAIL-TAC-D | 44k–54k | Same |
| PAIL-QPREF | 50k–56k | Same |
| SAIL-PREFRANK | 44k–52k | Only promoted episodes added → buffer fills slightly slower |
| SAIL-TAC-D | 43k–73k | Same |
| SAIL-QPREF | 50k–55k | Same |

Activation timing is uniform at ~44–73k across all variants. The one outlier (SAIL-TAC-D s3 at 73k) is a seed that had slower early policy learning and thus slower segment accumulation.

---

## Part 5 — Failure Modes

### 5.1 Failure Mode A: Ant RM Accuracy Collapse (Buffer Saturation)

**What happens:** Ant RM accuracy peaks at 0.97 around step 318k, then degrades monotonically to 0.69–0.77 after step 500k. The degradation tracks exactly with the ring buffer saturation (fills at ~500k for Ant's ~200-step episodes × 50-step segments = ~4 segs/ep × ~125k eps = 500k steps).

**Mechanism:**  
- **Phase 1 (0–300k):** Buffer contains teacher segments (expert data, returns 50–120/seg) + early student segments (returns 0–50). Large return gaps → easy to rank → accuracy ~0.80+.
- **Phase 2 (300k–500k):** Buffer fills. Teacher segments diluted but still present. Student segments span 0–120/seg. Accuracy peaks at 0.97.
- **Phase 3 (500k+):** Buffer full, ring overwrite active. Teacher segments evicted by ~510k. Student episodes plateau at ~2,700–3,100 GT return (ep_rew is flat in this region). At 50-step segments: most segments return ~130–155. Spread is narrow. The `tie_margin=0.5` threshold is much smaller than the actual return spread (~20–25/seg), so pairs are not rejected — but the BT loss gradient is small when preferences are not strong. The RM's logit differences are small → accuracy fluctuates around 0.70–0.78.
- **Root cause:** Ant learning plateaus. A plateau means all student episodes have similar return → all segments have similar quality → the RM cannot differentiate them → accuracy collapses to near-chance.

**Why this doesn't happen for HC/Walker2d/Hopper:** These environments continue learning past 500k. The student pool always contains episodes from different quality levels (early training at 1k ep_rew, late training at 7k ep_rew). Large return diversity → easy ranking → sustained high accuracy.

**Mitigation:** For environments that plateau, RM accuracy is not a reliable quality signal after buffer saturation. Adding `min_return_gap_for_segment` as a segment admission filter (only add segments where return is far from the current pool's mean) would help. Alternatively, stratified sampling from the buffer (ensure some teacher + some early student + recent student) would prevent eviction of the anchor segments.

### 5.2 Failure Mode B: Boltzmann Weight Collapse (PAIL variants)

**What happens:** In PAIL variants, the Boltzmann weights for the teacher buffer's pref_episodes pool collapse to effectively zero entropy — one episode captures all weight — very early in training (~60k steps for Walker2d).

**Mechanism:**  
The Boltzmann weight for episode i is `w_i = exp(J_i / β) / Σ_j exp(J_j / β)`. With β=1.0 and J values that grow from ~250 to ~2,000 over training, the softmax denominator is dominated by the maximum J. For a pool of N=100 episodes where J_max is just 200 units above J_mean:

`w_max / w_mean = N × exp((J_max − J_mean) / β) = 100 × exp(200) ≈ 10^89`

This is numerically zero for all episodes except the maximum. The `pref_reweight/entropy` ≈ 0 throughout training confirms this.

**Effect:** PAIL reduces to "always sample the single highest-RM-J episode from the expert buffer" as the discriminator's expert batch. This is not necessarily wrong (the best episode is a good teacher), but it defeats the purpose of diversity-seeking reweighting. Episodes that are second-best or near-expert-quality contribute nothing to discriminator training.

**Why PAIL still works:** The discriminator learns from a stable, high-quality expert signal (single best episode). It's essentially equivalent to offline SAIL with a 1-episode teacher buffer. The policy learns well anyway because the discriminator is still informative.

**Fix:** Use temperature scaling. The correct β for Boltzmann reweighting should be on the order of the J-value variance, e.g., `β = J_std / log(N)`. With J_std ~1,000 and N=100: `β ≈ 1000 / log(100) ≈ 217`. This would produce meaningful weight diversity.

### 5.3 Failure Mode C: SAIL-QPREF High Variance and Failures

**What happens:** SAIL-QPREF is the worst-performing variant (37.6% median ns_pct for Walker2d) with 2 near-total failures (ep_rew < 3, ns_pct < 0.05%).

**Mechanism:**  
QPREF adds a Q-function ranking loss: the Q-function should assign higher values to preferred (higher-J) trajectories. In SAIL-QPREF, this creates a conflict:

1. The discriminator provides surrogate reward to the Q-function (via critic TD updates).
2. QPREF provides a second training signal to the Q-function: preference rankings from RM-labeled episodes.

If the surrogate reward and the QPREF ranking are consistent, training is stable. If they conflict (e.g., discriminator assigns high reward to an episode that the RM assigns low J), the Q-function receives contradictory gradients. SAIL-QPREF does not have the Boltzmann reweighting that PAIL provides as an alignment mechanism — SAIL uses the raw discriminator reward without RM weighting. The result is higher variance and occasional catastrophic divergence.

**Why PAIL-QPREF is more stable:** With PAIL's reweighting, the discriminator's expert batch is dominated by the best-RM-J episode. This aligns the discriminator's reward signal with the RM's J-values, reducing the Q-function conflict.

### 5.4 Failure Mode D: RM Non-Activation (PAIL-TAC-D Walker2d seed 2)

**What happens:** `PAIL-TAC-D_48684226_2.out` shows `act_step=None` (RM never activated) and `ep_rew = −16.9` at 1M steps.

**Mechanism:** The log shows `teacher_count=0` from the first logged block, suggesting the teacher data was not loaded (possibly an I/O race on the NFS path, or the expert_data file was temporarily unavailable when the job started). Without teacher segments, the RM trains on student-vs-student pairs from a random policy — all returns near zero → near-random accuracy → never crosses the 60% threshold → RM never activates → LfD never turns off → policy trains against a single-episode teacher buffer → discriminator saturates → collapse.

This is not a systematic RM failure but a data-loading failure. Other seeds of the same job succeed normally.

---

## Part 6 — Final Conclusions

### 6.1 Is the Online RM Actually Learning a Good Reward?

**Yes, for 4 of 5 environments. Partially for Ant.**

- **HC, Walker2d, Hopper, Swimmer:** The RM achieves 0.97–1.00 held-out accuracy from ~150k steps onwards, indicating it has learned a reward function consistent with environment returns. It correctly orders segments by quality 97–100% of the time on a fixed held-out set.
- **Ant:** The RM learns well initially (0.97 at 300k) but degrades to 0.69–0.77 after the buffer fills at ~500k. The degradation is caused by Ant's policy plateauing — once all student episodes return similar values, the RM cannot differentiate them.

### 6.2 When Is the RM Useful vs Harmful?

**Useful:** From activation (~26–73k steps) through the end of training, except for Ant after 500k.

**Harmful/Degraded:**
- Ant post-500k: RM accuracy 0.69–0.77. Any component that relies on RM rankings (QPREF, Soft-TAC signs) is receiving noisy labels. The effect on final policy performance is moderate: Ant PAIL-QPREF achieves 49% ns_pct — decent but below the theoretical maximum.
- Boltzmann collapse (all PAIL variants): RM is not harmful but its potential information content is wasted. The Boltzmann weighting degenerates to single-episode sampling, losing diversity benefits.
- SAIL-QPREF conflict: When QPREF ranking loss conflicts with discriminator reward, the RM provides accurate labels to an unstable training objective.

### 6.3 Why RM-Based Methods May Fail or Succeed

**Success factors:**
1. **Sufficient return diversity in the segment buffer.** When the policy is still improving (wide spread of episode qualities), the RM easily learns to rank segments, achieving high accuracy.
2. **PAIL or PREFRANK as downstream consumer.** These add the RM signal softly (as a weight or ranking loss additive), rather than overriding the discriminator's reward.
3. **RM activation before policy learning begins.** Activation at 26–73k, well before the main learning phase (100k–500k), means the RM is ready when the policy needs it.

**Failure factors:**
1. **Policy plateau (Ant).** When the policy stops improving, segment diversity collapses and RM accuracy degrades. The RM outlives its usefulness but stays active (hysteresis threshold of 0.55 still not hit at 0.70 accuracy).
2. **β scale mismatch (PAIL).** Boltzmann reweighting with β=1.0 collapses to a point mass, wasting the RM's ability to provide a graded expert signal.
3. **Q-function conflict (SAIL-QPREF).** Without the alignment mechanism that PAIL provides, QPREF's RM-derived ranking signal can fight the discriminator's reward signal in the Q-function.
4. **Teacher segment eviction.** After ~500k steps, the buffer fills and teacher segments disappear. The RM then trains on student-only pairs. This is fine when the student policy spans multiple quality levels but degrades when it does not (Ant plateau).

---

## Part 7 — Final RM Scores Across Environments

| Environment | RM Act. Step | Final Acc | Final Loss | Rescores | Final ep_rew | Final ns_pct | Notable |
|-------------|-------------|-----------|------------|---------|-------------|-------------|---------|
| **HalfCheetah-v2** | ~26k | **1.00** | 0.010 | 49 | 7,870 | — | Near-perfect. Teacher eviction at ~650k. Loss drops from 0.24→0.01. |
| **Walker2d-v2** | ~44–73k | **0.97–1.00** | 0.005–0.015 | 47–48 | 3,600–5,850 | 76–124% | Excellent. Accuracy maintains after teacher eviction at ~550k. Best variant: SAIL-PREFRANK. |
| **Hopper-v2** | ~66k | **1.00** | 0.005–0.010 | 46–47 | 2,060–3,240 | 57–90% | Near-perfect from 300k. Accuracy never drops. Slower activation (short episodes → slower buffer fill). |
| **Swimmer-v2** | ~26k | **0.97–0.99** | 0.010–0.025 | 48–49 | 300–330 | 83–95% | Very good. Slow policy learning is NOT RM failure — RM is 0.97 while policy is still near-zero. |
| **Ant-v2** | ~29k | **0.69–0.82** | 0.018–0.035 | 47–49 | 2,800–3,100 | 48–52% | **Degraded.** Accuracy peaks at 0.97 (318k), collapses to 0.69 after 500k. Cause: policy plateau + buffer saturation. |

### Environment-specific Notes

**HalfCheetah-v2:** The strongest RM. HC's wide return range (−300 to 7,900 over training) makes the ranking task easy. Accuracy reaches 1.00 by ~526k and the RM essentially memorizes the preference structure of HC. Final performance (7,870 ep_rew for PAIL-QPREF) is the best across all envs.

**Walker2d-v2:** Strong RM with high policy performance. All variants activate at similar times. SAIL-PREFRANK is the best variant. The one catastrophic failure (PAIL-TAC-D s2) is a data-loading issue, not RM learning failure. Accuracy temporarily dips during teacher segment eviction (~550k) but recovers.

**Hopper-v2:** Near-perfect RM. Activation at 66k (later than HC/Walker/Swimmer) because Hopper's short episodes (~200–500 steps at ~50 steps/seg = ~4 segs/ep) take longer to fill the 500-segment threshold. Once active, Hopper's wide return range (12 to 3,600 over training) makes the RM extremely effective. Accuracy reaches 1.00 and never falls.

**Swimmer-v2:** Good RM accuracy, slow policy learning. Swimmer is unusual: held_out_acc is 0.97–0.99, but ep_rew stays near zero until ~600k. This suggests the policy learning bottleneck is in the discriminator or the policy optimization, not the reward model. The RM correctly ranks segments; the policy simply needs more exploration time to find good trajectories.

**Ant-v2:** The problematic environment. RM works well during the learning phase (0–300k) but degrades when learning stalls. The QPREF variant is particularly affected (final acc 0.69–0.82, ns_pct ~49%). PAIL variants are less affected because Boltzmann collapse already reduced RM influence. **Recommendation:** For Ant, the RM should either be frozen after accuracy peaks (saving the 318k checkpoint), or the segment buffer should be stratified to prevent eviction of high-contrast segments.

---

## Open Issues and Recommended Fixes

### Fix 1: Boltzmann Temperature Scaling

**Problem:** β=1.0 with J values on the scale of hundreds to thousands collapses to point-mass weighting.  
**Fix:** Set β dynamically: `β = max(J_std, 1.0)` where J_std is computed from the current pref_episodes pool. This normalizes the Boltzmann distribution to have meaningful entropy across the pool.  
**Location:** `datasets/teacher_buffer.py:_recompute_pref_weights()` — replace `beta = self._pref_reweight_beta` with `beta = max(self._pref_reweight_beta, float(np.std(j_arr)))`.

### Fix 2: Add RM Episode Return Logging

**Problem:** No direct comparison of RM return vs GT return is logged.  
**Fix:** In `sail.py:562–591` (`rescore_pref_episodes` block), log: `self.logger.record("online_rm/rescored_j_mean", np.mean(j_arr))`, `online_rm/rescored_j_std`, `online_rm/rescored_j_corr_with_gt` (Pearson correlation between RM J and GT ep returns, where GT is stored in each episode dict).

### Fix 3: Ant RM Quality Gate

**Problem:** RM stays "active" at 0.69 accuracy (above hysteresis threshold of 0.55), but is unreliable.  
**Fix:** Raise the deactivation threshold by reducing hysteresis: `activation_hysteresis=0.02` → deactivates when accuracy drops to 0.58. Or add a separate "reliable" flag at 0.85 accuracy that gates QPREF and Soft-TAC.

### Fix 4: Stratified Segment Buffer

**Problem:** Ring buffer evicts teacher segments after ~500k steps, removing the high-contrast anchor pairs.  
**Fix:** Reserve a fixed fraction (e.g., 20%) of the buffer for teacher segments and use a separate ring buffer for student segments. Teacher segments are never evicted.

---

*Analysis end. All metrics from parsed `.out` log files. Code references from `sail_sb3_online/reward_models/`, `datasets/`, `algorithms/`.*
