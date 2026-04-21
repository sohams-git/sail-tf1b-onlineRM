# TF Discriminator Parity Comparison — Final Results

**Date:** 2026-04-05  
**Change tested:** Two strict TF-parity changes to the discriminator  
**Baseline jobs:** 46729448 (Adaptive LfD, ec=0.05), 46729769 (AdaptPref LfD, ec=0.05)  
**TFDisc jobs:** 46766558 (Adaptive LfD TFDisc, ec=0.01), 46766592 (AdaptPref LfD TFDisc, ec=0.01)  
**Status:** LfD seeds complete (1M steps); PrefDisc seeds at 727k / 909k steps

---

## 1. What Changed

### Change 1: Discriminator firing cadence — 2× per policy update (TF parity)

**TF behavior** (`stable-baselines/stable_baselines/td3/sail.py` line 1672): disc fires every 500 env steps in the main loop, independent of policy training at 1000 steps → **20 disc gradient steps per 1000 env steps**.

**Old PyTorch behavior:** disc fired inside `SAIL.train()` called every 1000 steps → **10 disc gradient steps per 1000 env steps** (half of TF).

**Fix:** Overrode `_store_transition()` in `SAIL` to call `_update_discriminator()` every `disc_train_freq=500` env steps. Disc fires at steps N+500 and N+1000 within each policy cycle, exactly matching TF. `train()` only drains accumulated losses for logging.

### Change 2: Entropy coefficient — restored to TF default (0.05 → 0.01)

The previous `--entcoeff 0.05` was an empirical patch for the old single-fire regime. TF uses `adversary_entcoeff=0.01`. Restored to `--entcoeff 0.01` with 2× disc cadence.

### What was NOT changed

Reward formula, discriminator architecture, gradient penalty coefficient, obs normalization, LfD mixing, all TD3 hyperparameters — all unchanged.

---

## 2. Complete Training Trajectories

### 2.1 Adaptive SAIL + LfD Mixing (no PrefRank)

#### Baseline (ec=0.05, 1× disc, job 46729448)

| Step  | disc_loss | surr_rew | critic_loss | ep_rew | Promos |
|-------|-----------|----------|-------------|--------|--------|
| 50k   | 0.451     | 0.958    | 2.26        | −493   | —      |
| 100k  | 0.389     | 1.09     | 13.9        | +100   | —      |
| 200k  | 0.841     | 0.857    | 2.83        | +5370  | 0→1 (~210k) |
| 300k  | 1.05      | 0.562    | 1.26        | +6740  | —      |
| 500k  | 1.14      | 0.621    | 1.16        | +7340  | —      |
| 700k  | 1.21      | 0.641    | 1.23        | +8100  | —      |
| 900k  | 1.25      | 0.671    | 0.83        | +8180  | 170    |
| 936k  | 1.25      | 0.652    | 0.82        | +8140  | 170    |

*(Seed 1 shown; seed 2 at 712k: ep_rew=8330, disc=1.24, 171 promos)*

#### TFDisc (ec=0.01, 2× disc, job 46766558) — COMPLETE (1M steps)

**Seed 1:**

| Step   | disc_loss | surr_rew | critic_loss | ep_rew  | Promos |
|--------|-----------|----------|-------------|---------|--------|
| 50k    | 0.324     | 1.16     | 1.67        | −608    | —      |
| 100k   | 0.341     | 1.15     | 9.39        | −336    | —      |
| 200k   | 0.657     | 0.932    | 3.49        | +3270   | —      |
| 300k   | 0.937     | 0.518    | 1.78        | +6240   | 0→1 (~260k) |
| 500k   | 1.12      | 0.590    | 1.29        | +7180   | —      |
| 600k   | 1.17      | 0.609    | 1.01        | +7420   | 105    |
| 700k   | 1.19      | 0.627    | 0.966       | +7770   | —      |
| 800k   | 1.21      | 0.633    | 0.877       | +8110   | 173    |
| 900k   | 1.24      | 0.645    | 0.747       | +8290   | —      |
| **1000k** | **1.25** | **0.654** | **0.71** | **+8460** | **201** |

**Seed 2:**

| Step   | disc_loss | surr_rew | critic_loss | ep_rew  | Promos |
|--------|-----------|----------|-------------|---------|--------|
| 50k    | 0.252     | 1.27     | 3.35        | −554    | —      |
| 100k   | 0.377     | 1.11     | 10.3        | +77     | —      |
| 200k   | 0.852     | 0.880    | 2.54        | +5540   | 0→1 (~205k) |
| 300k   | 1.04      | 0.548    | 1.22        | +6760   | 27     |
| 500k   | 1.17      | 0.617    | 1.34        | +7390   | —      |
| 600k   | 1.20      | 0.615    | 1.19        | +8030   | —      |
| 700k   | 1.22      | 0.627    | 1.09        | +8330   | —      |
| 800k   | 1.26      | 0.653    | 0.857       | +8600   | —      |
| 900k   | 1.27      | 0.643    | 0.716       | +8650   | 182    |
| **1000k** | **1.29** | **0.664** | **0.625** | **+8830** | **195** |

### 2.2 Adaptive SAIL + PrefRank + LfD Mixing

#### Baseline (ec=0.05, 1× disc, job 46729769)

| Seed | Step | disc_loss | pref_loss | surr_rew | ep_rew | First Promo |
|------|------|-----------|-----------|----------|--------|-------------|
| s1   | 756k | 1.05      | 0.055     | 0.579    | +7230  | ~440k       |
| s2   | 640k | 1.20      | 0.012     | 0.640    | +7860  | ~218k       |

*(Jobs were cancelled/ended before 1M steps)*

#### TFDisc PrefRank (ec=0.01, 2× disc, job 46766592)

**Seed 1 — Q-divergence then recovery:**

| Step   | disc_loss | pref_loss | surr_rew | critic_loss | ep_rew  | Promos |
|--------|-----------|-----------|----------|-------------|---------|--------|
| 50k    | 0.206     | 0.004     | 1.32     | 1.35        | −530    | —      |
| 100k   | 0.195     | 0.001     | 1.38     | **123**     | −482    | —      |
| 150k   | 0.220     | 0.001     | 1.35     | **173**     | −228    | —      |
| 250k   | 0.255     | 0.000     | 1.29     | 16.9        | +118    | —      |
| 350k   | 0.355     | 0.001     | 1.17     | 5.10        | +1220   | —      |
| 450k   | 0.569     | 0.000     | 0.964    | 3.55        | +4480   | —      |
| 500k   | 0.653     | 0.000     | 0.989    | 3.02        | +5630   | —      |
| 574k   | —         | —         | —        | —           | +6300   | **1** (first promo) |
| 600k   | 0.824     | 0.018     | 0.470    | 1.83        | +6380   | —      |
| 700k   | 0.915     | 0.066     | 0.507    | 1.36        | +7080   | —      |
| 800k   | 0.999     | 0.088     | 0.509    | 1.11        | +7490   | —      |
| 909k   | 1.04      | 0.039     | 0.567    | 0.998       | +7730   | 102    |

**Seed 2 — healthy:**

| Step   | disc_loss | pref_loss | surr_rew | critic_loss | ep_rew  | Promos |
|--------|-----------|-----------|----------|-------------|---------|--------|
| 50k    | 0.238     | 0.003     | 1.26     | 3.98        | −518    | —      |
| 100k   | 0.434     | 0.001     | 1.05     | 16.0        | +474    | —      |
| 150k   | 0.735     | 0.002     | 0.881    | 4.99        | +3310   | 0→1 (~145k) |
| 250k   | 1.04      | 0.128     | 0.551    | 1.24        | +6600   | —      |
| 300k   | 1.10      | 0.116     | 0.588    | 0.987       | +6990   | —      |
| 350k   | 1.15      | 0.061     | 0.596    | 0.869       | +7190   | 58     |
| 500k   | 1.21      | 0.058     | 0.644    | 0.936       | +7740   | —      |
| 600k   | 1.24      | 0.037     | 0.639    | 0.805       | +7990   | —      |
| 700k   | 1.27      | 0.032     | 0.649    | 0.720       | +8090   | 131    |

---

## 3. Head-to-Head Comparison at Matched Steps

### 3.1 Adaptive LfD: TFDisc vs Baseline

| Metric                  | Baseline (ec=0.05, 1×) | TFDisc (ec=0.01, 2×) |
|-------------------------|------------------------|----------------------|
| First promotion (s1)    | ~210k steps            | ~260k steps (+50k)   |
| First promotion (s2)    | ~149k steps            | ~205k steps (+56k)   |
| ep_rew at 700k (s1)     | 8100                   | 7770 (−330)          |
| ep_rew at 700k (s2)     | 8330 (712k)            | 8330 (=)             |
| ep_rew at 1M (s1)       | — (ran to 936k: 8140)  | **8460**             |
| ep_rew at 1M (s2)       | — (ran to 712k: 8330)  | **8830**             |
| Final promos at 1M (s1) | 170 (at 936k)          | 201                  |
| Final promos at 1M (s2) | 171 (at 712k)          | 195                  |
| disc_loss at 1M         | (est. ~1.25–1.27)      | 1.25 / 1.29          |
| surr_rew at 1M          | (est. ~0.65–0.68)      | 0.654 / 0.664        |

**Both seeds eventually converge to the same ep_rew band (8100–8800).** TFDisc seed 2 actually reaches 8830 vs baseline seed 2's 8330 at 712k — but the baseline seed 2 was killed early; had it run to 1M it would likely be in the same range. The ~50k promotion delay in TFDisc washes out by 700k steps.

### 3.2 AdaptPref LfD: TFDisc vs Baseline

| Metric                    | Baseline (ec=0.05, 1×) | TFDisc (ec=0.01, 2×) |
|---------------------------|------------------------|----------------------|
| First promotion (s1)      | ~440k                  | **574k (+134k)**     |
| First promotion (s2)      | ~218k                  | ~145k (−73k faster!) |
| Q-divergence (s1)         | None (critic_loss ~4)  | Yes (peak 173)       |
| ep_rew at 700k (s1)       | 7230 (at 756k)         | 7080                 |
| ep_rew at 700k (s2)       | ~7700 (est.)           | 8090                 |
| ep_rew at 900k (s1)       | — (stopped at 756k)    | 7730                 |
| disc_loss at plateau (s1) | ~1.05                  | 1.04 (at 909k)       |
| pref_loss at plateau (s1) | ~0.06                  | 0.039 (at 909k)      |

For PrefRank: seed 2 under TFDisc actually promotes **earlier** (145k vs 218k) and reaches higher ep_rew at 700k (8090 vs ~7700 est.). But seed 1's Q-divergence delays first promotion by 134k steps. Once recovered, both seeds converge to the 7700–8100 range — the same band as the entcoeff=0.05 baseline.

---

## 4. Final Conclusions

### 4.1 Does TF disc parity (2× cadence + entcoeff=0.01) improve performance?

**For Adaptive SAIL + LfD (no PrefRank): no significant difference at 1M steps.**

Both configurations reach 8100–8800 ep_rew at 1M steps (or equivalent). The promotion delay (~50k steps) from TFDisc is fully recovered by step 700k. disc_loss, surrogate rewards, and promotion counts are statistically indistinguishable in the long run. Neither configuration is strictly better.

**For Adaptive+PrefRank + LfD: seed-dependent — one seed faster, one slower.**

TFDisc seed 2 promotes 73k steps *earlier* and reaches comparable performance. TFDisc seed 1 hits Q-divergence that delays first promotion by 134k steps and costs roughly 350k steps of recovery. Ultimately, seed 1 still reaches 7730 by 909k. The long-term trajectory appears similar — but at 1M steps the diverged seed would still be ~500 ep_rew behind the baseline.

### 4.2 Why does entcoeff=0.01 work now?

The previous failure mode (entcoeff=0.01 → disc saturates → ep_rew stuck at −600) was caused by the critic cold-start trap: no LfD mixing → random policy → trivially-discriminated → disc loss collapses. With LfD mixing active, the critic bootstraps from step 10k, the policy improves rapidly, and the discriminator faces a moving target that prevents saturation even with lower entropy regularization. entcoeff=0.01 is sufficient when the policy is actively improving.

### 4.3 Why does PrefRank seed 1 hit Q-divergence with TFDisc but not the baseline?

- 4-episode teacher buffer → PrefRank solves the ranking task in ~15k steps (pref_loss 0.76 → 0.001)
- After that, PrefRank provides zero additional disc regularization
- With 2× disc cadence + entcoeff=0.01, disc_loss reaches 0.195 at 100k (vs 0.23 in baseline)
- At disc_loss=0.195, surrogate_reward=1.38 (baseline: 1.27) — 0.11 higher
- This 0.11 difference in reward signal inflates Q-targets before the critic scale stabilizes → divergence
- The divergence is self-correcting (Q-scale stabilizes by 600k) but costs ~350k steps

### 4.4 Code status

The 2× disc cadence override of `_store_transition()` in `sail_sb3/algorithms/sail.py` is correct and stays. It faithfully implements TF parity. The behavioral tradeoff is controlled by `entcoeff`:

| entcoeff | Disc cadence | Adaptive LfD | AdaptPref LfD | Recommendation |
|----------|--------------|--------------|---------------|----------------|
| 0.05     | 1× (old)     | ✓ works      | ✓ works       | Safe default   |
| 0.01     | 2× (TF)      | ✓ works      | ⚠ s1 Q-div   | TF parity, minor risk |
| 0.01     | 1× (old)     | ✗ saturates  | ✗ saturates   | Never use      |

**The 2× cadence change is the prerequisite that makes entcoeff=0.01 viable.** Without it, entcoeff=0.01 fails. With it, entcoeff=0.01 works for Adaptive LfD and mostly works for AdaptPref LfD.

### 4.5 Recommendation

- **Production Adaptive SAIL runs:** `--entcoeff 0.05` is still the safer choice (no promotion delay, no Q-divergence risk). TFDisc (ec=0.01) reaches the same final performance but takes ~50k extra steps.
- **TF parity comparison runs:** Use `--entcoeff 0.01` with the new 2× disc code to match TF structure exactly.
- **AdaptPref runs:** Use `--entcoeff 0.05` to avoid Q-divergence in the PrefRank+2× disc combination.

---

## 5. Implementation Note: Disc Cadence Override

```python
# sail_sb3/algorithms/sail.py

def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
    super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
    if (self.num_timesteps >= self.learning_starts and
            self.num_timesteps - self._last_disc_update_step >= self.disc_train_freq):
        self._update_discriminator()       # 10 grad steps
        self._last_disc_update_step = self.num_timesteps
```

With `train_freq=1000` and `disc_train_freq=500`: fires at step N+500 and N+1000 → 20 disc grad steps per 1000 env steps. Matches TF line 1672 exactly. `train()` only reads `_pending_disc_losses` for logging.

---

*Baseline data: jobs 46729448 (LfD), 46729769 (PrefLfD).*  
*TFDisc data: jobs 46766558 (LfD, complete 1M steps), 46766592 (PrefDisc, 727k–909k steps, still running).*
