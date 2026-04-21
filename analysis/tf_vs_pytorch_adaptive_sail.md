# TF vs PyTorch Parity Analysis: Adaptive SAIL and Adaptive+PrefRank

**Date:** 2026-04-03  
**Scope:** Strict implementation-level parity analysis. No code changes. No new jobs.  
**Configurations analyzed:** Adaptive SAIL (LfD mixing) and Adaptive SAIL + Preference Ranking (LfD + PrefRank)  
**Active PyTorch jobs:** 46729448 (Adaptive LfD, seeds 1–2), 46729769 (AdaptPref LfD, seeds 1–2)

---

## Section 1: Teacher Dataset Parity

### 1.1 Dataset Identity

Both TF and PyTorch use the **identical NPZ file**:

```
teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
```

This file is referenced by absolute path in all sbatch scripts. There is no dataset-level divergence.

### 1.2 Dataset Structure

| Field              | Shape      | dtype   | Description                          |
|--------------------|------------|---------|--------------------------------------|
| `obs`              | (4000, 18) | float32 | Observations with TimeFeatureWrapper |
| `actions`          | (4000, 6)  | float32 | Expert actions                       |
| `rewards`          | (4000, 1)  | float32 | Per-step environment rewards         |
| `episode_returns`  | (4,)       | float64 | Pre-computed episode-level returns   |
| `episode_starts`   | (4000,)    | bool    | True at episode boundary indices     |

Episode boundaries: indices [0, 1000, 2000, 3000]. Exactly 4 expert episodes × 1000 steps.

**Obs dimensionality:** 18 = 17 native HalfCheetah-v2 dims + 1 time feature from `TimeFeatureWrapper`. Both TF and PyTorch apply `TimeFeatureWrapper`, so the obs space matches.

### 1.3 Expert Return Statistics

Computed at runtime from the `rewards` field (verified to match stored `episode_returns`):

| Episode | Return    |
|---------|-----------|
| 0       | 6932.27   |
| 1       | 6780.64   |
| 2       | 6741.27   |
| 3       | 6988.28   |
| **Min** | **6741.27** |
| **Max** | 6988.28   |
| **Mean**| 6860.62   |

The filename says "scores_5600" which does **not** match the actual returns (~6741–6988). The filename is misleading. Both implementations correctly use the minimum episode return as the promotion threshold: **6741.27**.

### 1.4 Expert Score Threshold: TF vs PyTorch

**TF implementation** (`stable-baselines/stable_baselines/td3/sail.py`, lines 1362–1368):
```python
# Computed online while loading demos into demo_replay_buffer:
if demo_dones[idx+1] == 1:
    self.expert_scores.append(episode_score)
# At promotion check (line 1543):
if episode_score > self.expert_scores[0]:  # expert_scores[0] = minimum
```
TF computes `episode_score` by summing rewards over the episode during demo loading. After loading 4 episodes, `expert_scores = [6741.27, 6780.64, 6932.27, 6988.28]` (sorted). Threshold = `expert_scores[0]` = **6741.27**.

**PyTorch implementation** (`sail_sb3/utils/callbacks.py`):
```python
# teacher_buffer reads episode_returns from NPZ:
self.expert_returns = sorted(parsed_data['episode_returns'])
self.expert_threshold = self.expert_returns[0]  # = 6741.27
```
PyTorch reads the pre-computed `episode_returns` field and takes the minimum. Result: **6741.27**.

**Verdict: Exact numerical parity.** Both frameworks use 6741.27 as the promotion threshold.

### 1.5 LfD Mixing Data Path: TF vs PyTorch

**TF** (`generate_train_data`, line 454): samples from `self.demo_replay_buffer` (a standard replay buffer loaded with expert transitions). Includes `(obs, action, reward, next_obs, done)` tuples, same as the online replay buffer.

**PyTorch**: samples from `teacher_buffer.sample_batch()` which returns `states`, `actions`, `next_states`, `dones`. `next_states` was added explicitly in this implementation to match TF's `batch_next_obs` field used in Bellman targets.

**Verdict: Functionally equivalent.** The PyTorch `teacher_buffer` now provides the same 4-tuple as TF's `demo_replay_buffer`.

---

## Section 2: TF vs PyTorch Adaptive SAIL — Performance Comparison

### 2.1 Configuration Comparison

| Hyperparameter         | TF (`sail.yml` + hardcoded) | PyTorch (current)         | Match?        |
|------------------------|-----------------------------|---------------------------|---------------|
| `n_timesteps`          | 3,000,000                   | 1,000,000                 | ✗ (see §2.3)  |
| `batch_size`           | 256                         | 256                       | ✓             |
| `learning_starts`      | 10,000                      | 10,000                    | ✓             |
| `train_freq`           | 1,000                       | 1,000                     | ✓             |
| `gradient_steps`       | 1,000                       | 1,000                     | ✓             |
| `gamma`                | 0.99                        | 0.99                      | ✓             |
| `learning_rate`        | 1e-3                        | 1e-3                      | ✓             |
| `policy_kwargs`        | `layers=[400, 300]`         | `net_arch=[400, 300]`     | ✓             |
| `entcoeff`             | 0.01                        | **0.05**                  | ✗ (deliberate)|
| `gradcoeff`            | 10.0                        | 10.0                      | ✓             |
| `disc_gradient_steps`  | 10                          | 10                        | ✓             |
| `disc_batch_size`      | 256                         | 256                       | ✓             |
| `train_disc_freq`      | 500 (hardcoded, line 147)   | 500 (default)             | ✓ (nominal)   |
| Disc fires per 1k env steps | 2 (500 + 1000)        | 1 (only at 1000 boundary) | ✗ (known gap) |
| LfD batch split        | 128 expert + 128 policy     | 128 expert + 128 policy   | ✓             |
| `self.mix = False` after promotion | Yes (line 1569) | Yes (`lfd_active` flag)  | ✓             |
| Obs normalization      | RunningMeanStd in disc      | RunningMeanStd in disc    | ✓             |
| TimeFeatureWrapper     | Yes                         | Yes                       | ✓             |

### 2.2 Remaining Structural Mismatch: Discriminator Firing Rate

TF's main loop (line 1672) fires the discriminator independently every 500 env steps, regardless of when the policy trains. The policy trains every 1000 steps. Result: the discriminator receives **20 gradient steps per 1000 env steps** (2 fires × 10 steps each).

PyTorch's discriminator is triggered inside `learn()` which is called every 1000 steps. Although `disc_train_freq=500`, the check `num_timesteps - last_disc_update >= 500` always passes at this call point, so it fires **once** per 1000-step block: **10 gradient steps per 1000 env steps**.

This 2× discrepancy is a known structural gap. However, based on empirical evidence across 10+ seeds, it does **not** prevent learning in PyTorch when LfD mixing is active. The original 2× disc cadence was hypothesized to help (Fix 1 in the prior plan), but experiments confirmed the critic cold-start (not disc frequency) was the root cause of all failures.

### 2.3 Why PyTorch Reaches ~8000 ep_rew vs TF ~6500

The performance comparison requires careful scoping:

**What "TF ~6500" refers to:**  
The 6500 figure appears in prior CLAUDE.md and plan notes as a reference to TF AdaptPref runs with full 3M-step training. No TF `gail-lfd-adaptive-dynamic` logs with HalfCheetah are available in this repository. The comparison is therefore:
- TF `gail-lfd-adaptive-dynamic` (the LfD-mixed Adaptive mode) has no confirmed log here
- TF AdaptPref performance was reported elsewhere as the target (~5600–6741 range, matching expert threshold)

**Why PyTorch at 1M steps reaches 8100–8330:**

1. **LfD mixing is now working.** Before the fix, the critic cold-start trap kept all seeds at ep_rew ≤ −600 for the full 1M steps. With LfD mixing, the critic receives mixed (expert + policy) batches that produce 2.26–13.9 critic_loss at step 50k (vs ~0.02 in failed runs), enabling real actor gradient.

2. **entcoeff=0.05 prevents disc saturation.** TF used 0.01 which was insufficient for PyTorch's slightly different disc dynamics. With 0.05, disc_loss stabilizes at 1.20–1.26 and surrogate rewards stay at 0.63–0.67 through the entire post-promotion phase. This keeps the imitation signal strong and informative.

3. **The expert threshold (6741) is cleared early.** First promotions occur at ~149k steps (LfD-s2) and ~210k steps (LfD-s1). After clearing the threshold, the teacher buffer grows continuously (170+ promotions by 900k steps), meaning the discriminator is trained against increasingly good student episodes — creating a bootstrapped curriculum.

4. **8000 > 6741 (expert threshold).** The policy significantly exceeds the minimum expert return, which is expected behavior once the adaptive curriculum is running. TF likely also exceeds the threshold in successful runs, but the "6500" comparison point may correspond to a suboptimal configuration or a different measurement window.

5. **PyTorch runs only 1M steps (vs TF's 3M).** At 1M steps, ep_rew is still increasing (~8100–8330 and growing). The trajectory suggests further improvement with more steps. TF at 3M steps would presumably be higher, not lower, if LfD mixing was functioning correctly.

**Conclusion on performance gap:** The ~8000 vs ~6500 comparison is not apples-to-apples. PyTorch at 1M steps outperforms what appears to be a reference TF number from a different run configuration or step count. There is no evidence that PyTorch is algorithmically better — the performance gap is explained by: (a) LfD mixing now functioning correctly, (b) entcoeff tuned for PyTorch's dynamics, (c) possibly different reference baseline for the "6500" figure.

### 2.4 Learning Trajectory: Adaptive SAIL with LfD Mixing (Seed 1, Job 46729448_1)

| Step   | disc_loss | surr_rew | critic_loss | ep_rew_mean | Notes                        |
|--------|-----------|----------|-------------|-------------|------------------------------|
| 12k    | 4.76      | 0.735    | 0.056       | −332        | First training update        |
| 50k    | 0.451     | 0.958    | 2.26        | −493        | Critic bootstrapping         |
| 100k   | 0.389     | 1.09     | 13.9        | +100        | Policy starting to improve   |
| 150k   | 0.658     | 0.917    | 6.57        | +2580       | Strong learning               |
| 200k   | 0.841     | 0.857    | 2.83        | +5370       | First promotions begin (~210k)|
| 300k   | 1.05      | 0.562    | 1.26        | +6740       | Post-promotion disc hardens  |
| 500k   | 1.14      | 0.621    | 1.16        | +7340       | Steady improvement           |
| 700k   | 1.21      | 0.641    | 1.23        | +8100       | Plateau emerging             |
| 900k   | 1.25      | 0.671    | 0.83        | +8180       | Stable high performance      |

**Key observations:**
- critic_loss peaks at ~14 around step 100k due to Q-value scale adjusting to surrogate rewards ~1.0
- Post-promotion (step 210k+), disc_loss gradually increases as teacher buffer fills with good episodes
- surrogate_reward_mean rises from 0.735 → 1.09 → then stabilizes at 0.62–0.68 post-promotion
- ep_rew_mean trajectory: flat (−500 to +100) → rapid rise (+100 → +6740 over 150k steps) → gradual improvement (+6740 → +8180)

---

## Section 3: Adaptive SAIL vs Adaptive+PrefRank Analysis

### 3.1 Current Job Summary (as of ~2026-04-03)

| Job       | Config         | Seed | Steps  | ep_rew | disc_loss | surr_rew | pref_loss | Promotions | First Promo |
|-----------|----------------|------|--------|--------|-----------|----------|-----------|------------|-------------|
| 46729448_1| LfD            | 1    | 936k   | 8,140  | 1.25      | 0.652    | —         | 170        | ~210k       |
| 46729448_2| LfD            | 2    | 712k   | 8,330  | 1.24      | 0.650    | —         | 171        | ~149k       |
| 46729769_1| PrefLfD        | 1    | 756k   | 7,230  | 1.05      | 0.579    | 0.055     | ~125        | ~440k       |
| 46729769_2| PrefLfD        | 2    | 640k   | 7,860  | 1.20      | 0.640    | 0.012     | 128        | ~218k       |

### 3.2 Preference Ranking Loss Behavior

The `pref_rank_disc` loss adds a ranking objective to the discriminator: given pairs of expert episodes sampled from the teacher buffer, the discriminator should assign higher probability (lower disc loss) to episodes with higher cumulative return.

**Pre-promotion behavior (initial teacher buffer only):**

From PrefLfD seed 1 early trajectory:
```
step 12k: pref_loss = 0.756  (initial discriminator update, expert-only pairs)
step 13k: pref_loss = 0.477  (fast decay as ranking is learned)
step 15k: pref_loss = 0.014  (nearly converged)
step 16k: pref_loss = 0.001  (effectively zero)
step 50k: pref_loss = 0.001  (unchanged)
step 100k–440k: pref_loss ≈ 0.0001–0.003 (near-zero throughout)
```

**Why pref_loss collapses to near-zero:**  
The teacher buffer initially contains exactly 4 expert episodes with returns [6741, 6780, 6932, 6988]. The `pref_rank_batch_size=16` samples pairs from these 4 episodes. After ~3,000 discriminator updates, the ranking among these 4 fixed episodes is fully learned. The pref_loss gradient becomes negligible — pref_rank effectively provides **no additional discriminator regularization** during the pre-promotion phase (steps 12k–440k for seed 1).

**Post-promotion behavior:**

When the first student episode is promoted into the teacher buffer (step ~440k for seed 1, ~218k for seed 2), the teacher buffer now contains a mix of expert episodes (~6741–6988 return) and student episodes (~6800+ return). This introduces new ranking pairs with higher return variation.

```
step 440k (seed 1): pref_loss = 0.533  ← spike at first promotion
step 450k: pref_loss = 0.399
step 500k: pref_loss = 0.080
step 550k: pref_loss = 0.113
step 650k: pref_loss = 0.072
step 756k: pref_loss = 0.055
```

Post-promotion pref_loss stabilizes at 0.04–0.15, providing a real regularization effect that keeps the discriminator from over-converging.

### 3.3 PrefRank Effect on Discriminator Convergence

Pre-promotion disc_loss comparison:

| Step  | LfD-s1 disc_loss | PrefLfD-s1 disc_loss |
|-------|-----------------|---------------------|
| 50k   | 0.451           | 0.380               |
| 100k  | 0.389           | 0.230               |
| 150k  | 0.658           | 0.239               |
| 200k  | 0.841           | 0.301               |
| 250k  | 0.978           | 0.370               |
| 300k  | 1.05            | 0.470               |
| 350k  | 1.09            | 0.544               |

**Observation:** PrefLfD disc_loss is consistently lower (0.23–0.54) than LfD (0.39–1.09) in the pre-promotion window. This seems counterintuitive — lower disc_loss in PrefLfD means the discriminator is more confident in distinguishing expert from policy, not less. 

The explanation: PrefLfD converges the discriminator *faster* pre-promotion because the initial high pref_loss (0.756 at step 12k, decaying to 0.001 by step 16k) provides a strong early gradient update that helps the discriminator specialize quickly. After pref_loss collapses, the disc continues from a better-initialized state.

This faster discriminator convergence does **not** help the policy and may **delay** the first promotion:

- LfD-s1: first promotion at ~210k steps (ep_rew ~5700 at promotion)
- PrefLfD-s1: first promotion at ~440k steps (ep_rew ~5600 at promotion)
- LfD-s2: first promotion at ~149k steps
- PrefLfD-s2: first promotion at ~218k steps

### 3.4 PrefRank Effect on Final Performance

| Config  | ep_rew at 640k steps | ep_rew at 756k steps | ep_rew at 936k steps |
|---------|---------------------|---------------------|---------------------|
| LfD-s1  | 7,900 (est.)        | 8,100               | 8,140               |
| LfD-s2  | 8,330               | —                   | —                   |
| PrefLfD-s1 | 6,800 (est.)     | 7,230               | (running)           |
| PrefLfD-s2 | 7,860            | —                   | (running)           |

At comparable step counts, LfD outperforms PrefLfD:
- At ~640k steps: LfD-s2 = 8330, PrefLfD-s2 = 7860 (−470)
- At ~756k steps: LfD-s1 = 8100, PrefLfD-s1 = 7230 (−870)

**Interpretation:** PrefRank is providing negligible benefit and measurable cost in the pre-promotion phase (~0 to 440k steps). The preference ranking signal, while theoretically motivated, cannot activate until the teacher buffer contains student episodes with varied returns. For the 1M-step budget, the training time wasted in pre-promotion PrefLfD-s1 (440k steps before any promotions) is the dominant factor in its underperformance relative to pure LfD.

**Post-promotion trajectory (seed 1 comparison):**  
After first promotion, both configs show a similar convergence pattern: ep_rew rises rapidly from ~5600 to 7000+ within 100–150k additional steps. The post-promotion slope is approximately similar, suggesting PrefRank's post-promotion disc stabilization benefit is minor.

---

## Section 4: Mismatch Table

### 4.1 Resolved Mismatches (Prior Sessions)

| ID  | Component            | TF Behavior              | Previous PyTorch Behavior | Fix Applied          |
|-----|----------------------|--------------------------|--------------------------|----------------------|
| M0a | Obs normalization    | RunningMeanStd in disc   | None                     | Added; disc improved 0.005 → 0.12 |
| M0b | Training frequency   | train_freq=1000, grad=1000 | Mismatched           | Fixed; now matches TF |
| M0c | Gradient penalty     | On normalized states     | On raw states            | Fixed                |
| M2  | LfD mixing (CRITICAL)| `generate_train_data()` 50/50 mix before promotion | Missing entirely | Implemented: critic_loss 0.02 → 2–14 |
| M3  | `self.mix = False`   | Disabled after first promotion | Not implemented   | Implemented via `lfd_active` flag |
| M4  | `next_obs` in expert buffer | Available in `demo_replay_buffer` | Missing | Added `teacher_buffer.next_states` |

### 4.2 Known Remaining Mismatches

| ID  | Component           | TF Behavior           | PyTorch Behavior        | Impact    |
|-----|---------------------|-----------------------|-------------------------|-----------|
| M1  | Disc firing rate    | 20 grad steps / 1000 env steps | 10 grad steps / 1000 env steps | Low — learning works without fix |
| M5  | `entcoeff`          | 0.01 (but may saturate) | **0.05** (deliberate)  | Calibrated to PyTorch disc dynamics |
| M6  | Total training steps| 3,000,000             | 1,000,000               | Accounts for performance plateau |

### 4.3 Non-Mismatches (Confirmed Equivalent)

| Component               | TF                   | PyTorch              | Notes                     |
|-------------------------|----------------------|----------------------|---------------------------|
| Expert threshold        | 6741.27              | 6741.27              | Identical computation     |
| Teacher dataset         | Same NPZ file        | Same NPZ file        | Bit-identical             |
| Batch size (LfD split)  | 128 + 128            | 128 + 128            | `batch_size=256 // 2`     |
| Policy architecture     | `[400, 300]`         | `[400, 300]`         | TF `layers=`, SB3 `net_arch=` |
| Gamma, tau, LR          | 0.99, 0.005, 1e-3    | 0.99, 0.005, 1e-3    | Exact match               |
| TimeFeatureWrapper      | Applied              | Applied              | Obs dim 17 → 18           |
| Promotion condition     | `score > expert_scores[0]` | `score > expert_threshold` | Same logic        |
| Mix disable             | `self.mix = False` after first promotion | `lfd_active = False` when `teacher_buffer grows` | Equivalent |

---

## Section 5: Root Cause Conclusions

### 5.1 Why Previous PyTorch Runs Failed (0 promotions, ep_rew = −600)

The critic cold-start trap was the root cause:

1. **No LfD mixing** → replay buffer contained only policy transitions (random-quality)
2. Cold random policy produces near-constant bad actions → Q-target variance ≈ 0
3. `critic_loss ≈ 0.02` (near-zero) → actor receives near-zero gradient
4. Actor cannot improve → policy stays random → disc saturates (disc_loss → 0.12)
5. Disc assigns ~5% probability to policy → surrogate reward → 0.056 → signal collapses
6. **Result:** 0 promotions across all seeds at 1M steps

### 5.2 Why LfD Mixing Fixed It

1. **LfD mixing** → 50% expert transitions in every critic/actor batch before promotion
2. Expert (obs, next_obs) pairs introduce high-return, diverse trajectories → Q-target variance >> 0
3. `critic_loss ≈ 2–14` → actor receives real gradient signal
4. Policy improves from −600 to +100 by step 100k, +5370 by step 200k
5. First promotions at 149k–210k steps → teacher buffer grows → adaptive curriculum begins
6. **Result:** 170+ promotions, ep_rew 8100–8330 by 700–900k steps

### 5.3 Why PrefRank Provides Limited Benefit in Current Setup

1. Pre-promotion (the critical phase): pref_loss ≈ 0.001 (effectively zero gradient)
   - Only 4 expert episodes with similar returns → all pairs quickly ranked correctly
   - PrefRank cannot regularize the discriminator when all training pairs have near-equal quality
2. Post-promotion: pref_loss rises to 0.04–0.53, providing real signal
   - But by this point, the policy has already cleared the expert threshold
   - The primary benefit of PrefRank (keeping disc from saturating) is achieved by `entcoeff=0.05` in both configs
3. Net effect: PrefLfD promotes later (440k vs 210k for seed 1) and reaches lower ep_rew at comparable steps

**Implication:** PrefRank is most valuable when the teacher buffer contains episodes with diverse, high-variance returns — i.e., when many student episodes have been promoted across a wide score range. In the early curriculum (4 near-equal expert episodes), it provides no advantage.

### 5.4 On the TF vs PyTorch Performance Comparison

No TF `gail-lfd-adaptive-dynamic` logs are available in this repository for direct comparison. The performance comparison is therefore indirect:

- **TF reference** (~6500 ep_rew): Likely from AdaptPref runs at 3M steps, or vanilla Adaptive SAIL without LfD fix
- **PyTorch current** (8100–8330 at 1M steps): With LfD fix + entcoeff=0.05

The PyTorch implementation now correctly implements `gail-lfd-adaptive-dynamic` with `self.mix=True` before first promotion. Given that the expert threshold is 6741 and PyTorch clears it within 149–210k steps, and the max expert return is 6988, reaching 8100–8330 at 1M steps is consistent with correct algorithm behavior (the policy surpasses the expert dataset quality).

---

## Section 6: Summary and Recommendations

### 6.1 Parity Verdict

| Aspect                         | Status       | Evidence                              |
|--------------------------------|--------------|---------------------------------------|
| Teacher dataset                | **Parity** ✓ | Identical file, identical threshold   |
| Expert threshold computation   | **Parity** ✓ | Both = 6741.27                        |
| LfD batch mixing               | **Parity** ✓ | 128+128, disables after promotion     |
| Promotion logic                | **Parity** ✓ | score > min_expert_return             |
| `next_obs` in Bellman target   | **Parity** ✓ | `teacher_buffer.next_states` added    |
| Discriminator firing cadence   | **Gap** ✗    | 10 vs 20 grad steps/1k env steps      |
| entcoeff                       | **Calibrated** | 0.05 vs TF 0.01 (TF value insufficient for PyTorch) |
| Learning outcome               | **Functional** | All 4 seeds promoted; ep_rew 7230–8330 |

### 6.2 Recommended Next Steps (No Implementation)

1. **Run LfD to 3M steps** (matching TF's `n_timesteps`): to determine whether PyTorch's ep_rew plateau at ~8100–8330 is genuinely higher than TF's, or whether TF would also reach this level at 3M steps.

2. **Verify PrefRank benefit with more seeds**: PrefLfD-s1 and s2 show lower performance than LfD-s1 and s2. A 3–5 seed comparison would confirm whether PrefRank is net-negative, neutral, or positive at 1M–3M steps.

3. **Assess disc cadence gap (M1) empirically**: The 2× disc firing mismatch could be tested by adding an intermediate disc update at step 500 within the 1000-step block. Low risk, but evidence suggests it's not blocking learning.

4. **Document entcoeff sensitivity**: TF's 0.01 failed in PyTorch; 0.05 works. Whether this reflects a difference in gradient penalty normalization, optimizer state, or random seed dynamics is an open question worth a controlled ablation.

---

*Analysis compiled from: job logs 46729448_1/2, 46729769_1/2; TF reference code `stable-baselines/stable_baselines/td3/sail.py`; dataset `teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz`; PyTorch implementation `sail_sb3/algorithms/sail.py`, `sail_sb3/datasets/teacher_buffer.py`.*
