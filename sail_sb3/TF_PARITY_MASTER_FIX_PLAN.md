# TF Parity Master Fix Plan — Adaptive SAIL

**Date**: 2026-04-03  
**Analysis basis**: Job 46667331 (137k steps), prior diagnosis MDs, TF source code  
**Status**: Fixes implemented (sbatch scripts created), awaiting sbatch results

---

## 1. Current Behavior (Job 46667331)

### Training Timeline

| Env Steps | n_updates | disc_loss | surr_reward | ep_rew_mean | actor_loss |
|-----------|-----------|-----------|-------------|-------------|------------|
| 11k (1st train) | 1,000  | 4.86  | 0.679 | -333 | -2.43 |
| 12k           | 2,000  | 1.65  | 0.707 | -353 | -6.22 |
| 14k           | 4,000  | 1.23  | 0.571 | -386 | -14.4 |
| 18k           | 8,000  | 0.737 | 0.286 | -431 | -17.4 |
| 43k           | 32,000 | 0.391 | 0.180 | -528 | -10.7 |
| 137k (final)  | ~127k  | 0.118 | 0.056 | -600 | -5.9 |

**Observations:**
- `disc_loss` and `surrogate_reward_mean` are perfectly anti-correlated
- `ep_rew_mean` degrades monotonically -325 → -600; never improves
- `actor_loss` peaks at -17.5 around n_updates=7000, then slowly falls (Q-values decaying with rewards)
- **Zero promotions**: policy stuck at -600, threshold at 6741.3
- Disc at convergence assigns ~5% expert probability to policy → reward = -log(0.95) ≈ 0.051

**Disc saturation math**: At surr_reward=0.056, sigmoid(logit)≈0.05, meaning disc has 95% confidence policy≠expert. The entropy regularizer at entcoeff=0.01 is dominated by BCE loss: entropy_loss ≈ -0.01 × 0.3 = -0.003 vs BCE ≈ 0.12 — a 40:1 ratio against the regularizer.

---

## 2. TF Behavior (Source of Truth)

**File**: `stable-baselines/stable_baselines/td3/sail.py`  
**HalfCheetah config**: `stable-baselines/hyperparams/sail.yml`

### Training Loop (per-step, lines 1651–1676):
```python
# Every 1000 steps: policy training
if step % self.train_freq == 0:
    for grad_step in range(self.gradient_steps):  # 1000 gradient steps
        self._train_step(...)

# Every 500 steps (independent of policy): discriminator training
if step % self.train_discriminator_freq == 0:
    if self.num_timesteps >= self.learning_starts:
        self._train_discriminator(...)  # 10 gradient steps
```

**Per-1000-env-steps budget:**
- Discriminator: **20 gradient steps** (fires at step 500 + step 1000)
- Policy: **1000 gradient steps** (fires at step 1000)
- Ratio: 1 disc step per 50 policy steps

### Key hyperparameters:
- `train_freq=1000`, `gradient_steps=1000`
- `train_discriminator_freq=500`, `d_gradient_steps=10`
- `entcoeff=0.01`, `gradcoeff=10.0`
- `batch_size=256`, `d_batch_size=256`

---

## 3. Mismatches

| # | Component | TF | PyTorch | Why It Matters | Evidence |
|---|-----------|-----|---------|----------------|---------|
| **M1** | Disc update frequency | Every **500** env steps (independent loop) | Inside `train()` at 1000-step intervals → effectively every **1000** steps | PyTorch disc gets 2x fewer updates | TF:1672 vs `sail.py:76` |
| **M2** | Disc gradient steps per 1000 env steps | **20** (500 fires twice) | **10** (fires once) | Net disc training 2x lower | Same as M1 |
| **M3** | entcoeff efficacy | 0.01 works at TF's working point | 0.01 insufficient → disc_loss→0.12, surr→0.056 | Entropy regularizer overwhelmed by BCE | Log 46667331: disc_loss 4.86→0.12 |
| **M4** | TF vanilla saturation | **Unknown** — no TF vanilla run available | PyTorch vanilla fails | Critical: if TF also fails, need pref ranking | Ambiguous in diagnosis docs |
| **M5** | Policy degrades immediately | Unknown | -325→-353 at first train step despite surr=0.68 | Critic hasn't bootstrapped; actor exploits bad Q-values | Log: n_updates=1000, rew=-333 already declining |

**Already fixed ✅:**
- ✅ M0a: Obs normalization (RunningMeanStd in Adversary) — disc_loss 0.005→0.12
- ✅ M0b: Training frequencies (train_freq=1000, grad_steps=1000, disc_train_freq=500)
- ✅ M0c: Gradient penalty computed on normalized observations

---

## 4. Root Causes (Ranked)

### RC1 (PRIMARY): Entropy regularization insufficient to prevent disc saturation

The discriminator converges to disc_loss=0.12 which gives surr_reward≈0.056 (disc assigns 5% expert probability to policy). At this reward level, the policy gradient via TD3 is attenuated by a factor of ~0.05 × (1-0.05) ≈ 0.05 relative to a perfectly calibrated disc. Learning cannot proceed.

The `entcoeff=0.01` entropy regularizer contributes a loss of ~0.003 vs total BCE of ~0.12 — it has negligible effect at the converged point. Increasing to 0.05 makes entropy loss ~0.015 vs BCE ~0.3 (new equilibrium target), which should maintain a better-calibrated disc.

**Fix**: `--entcoeff 0.05`

### RC2 (SECONDARY): Disc update frequency mismatch (M1/M2)

PyTorch disc does 10 gradient steps per 1000 env steps; TF does 20. This is a structural mismatch from SB3's `train()` architecture vs TF's per-step loop. However, since PyTorch disc STILL saturates with 2x fewer updates, this alone cannot explain the failure.

**Fix (Option B, no code change)**: `--disc_gradient_steps 20` (doubles disc budget to match TF)

### RC3 (STRUCTURAL): Vanilla Adaptive SAIL may require pref ranking on discriminator

AdaptPref (pref ranking loss weight=0.1) maintains disc_loss≈0.35-0.45 and reaches 5600 reward. The pref ranking loss provides a curriculum signal that prevents the disc from specializing to the current policy distribution. This may be load-bearing for HalfCheetah, not optional.

**Fix**: Enable `--pref_rank_disc --pref_rank_weight 0.1 --pref_rm <path>`

---

## 5. Minimal Fix Sequence

### Step 1 (Run A): Test entcoeff=0.05 alone
```bash
sbatch sail_sb3/HC_Adaptive_entcoeff005.sbatch
```
- **What it tests**: RC1 in isolation
- **Expected**: disc_loss stabilizes 0.25–0.35, surr_reward >0.15, ep_rew starts improving

### Step 2 (Run B): Test entcoeff=0.05 + disc_gradient_steps=20
```bash
sbatch sail_sb3/HC_Adaptive_ec005_disc20.sbatch
```
- **What it tests**: RC1 + RC2 together
- **Expected**: Faster learning than Run A if RC2 matters

### Step 3 (Run C): Vanilla SAIL + entcoeff=0.05 (control)
```bash
sbatch sail_sb3/HC_Vanilla_entcoeff005.sbatch
```
- **What it tests**: Whether basic GAIL learning works with entcoeff fix, without adaptive mechanism
- **Expected**: ep_rew improves above -400 if basic GAIL signal is usable

### Step 4 (Run D): AdaptPref rebase (upper bound)
```bash
sbatch sail_sb3/HC_AdaptPref_rebase.sbatch
```
- **What it tests**: Whether AdaptPref still works after obs normalization + freq fixes
- **Expected**: disc_loss≈0.35-0.45, ep_rew≈3000-5600 (confirms RC3 is the fallback)

---

## 6. Expected Effect of Each Fix

| Config | disc_loss @ convergence | surr_reward | ep_rew @ 300k steps | Notes |
|--------|------------------------|-------------|---------------------|-------|
| Baseline (46667331) | 0.12 | 0.056 | -600 | No learning |
| Fix 1 only (disc_gs=20) | ~0.08 | ~0.04 | -600 | More disc updates → faster saturation, WORSE |
| Fix 2 only (entcoeff=0.05) | 0.25–0.35 | 0.15–0.20 | TBD | Depends on RC1 being primary |
| Fix 1 + Fix 2 | 0.28–0.38 | 0.15–0.25 | Possible -400 to -200 | Best non-pref config |
| AdaptPref | 0.35–0.45 | 0.25–0.40 | ~3000-5600 | Proven working |

**Warning**: Fix 1 alone (more disc updates without higher entcoeff) likely makes things worse. Always pair with Fix 2.

---

## 7. Validation Plan

### Early stopping signal (check at step 50k):
- If `disc_loss < 0.15`: disc is saturating again → run failed
- If `surr_reward < 0.08`: reward too small → no learning possible

### Success signal (check at step 200k):
- `disc_loss > 0.25` sustained (not decaying)
- `surr_reward_mean > 0.15`
- `ep_rew_mean > -400` (meaningful improvement from -600)
- Any `adaptive/promoted_episodes > 0`

### Monitoring command:
```bash
grep -E "ep_rew_mean|disc_loss|surrogate_reward_mean" \
  sail_sb3/logs/HC_Adaptive_ec005_*.out | tail -60
```

### Full monitoring loop:
```bash
for f in sail_sb3/logs/HC_Adaptive_ec005_*.out sail_sb3/logs/HC_Vanilla_ec005_*.out \
          sail_sb3/logs/HC_AdaptPref_rebase_*.out; do
  echo "=== $f ==="; tail -1 "$f"
  grep -E "ep_rew_mean|disc_loss|surrogate_reward_mean" "$f" | tail -5
done
```

---

## 8. When to Re-enable AdaptPref

**If Runs A + B fail** (ep_rew stays at -600 after 300k steps):
- Conclusion: entcoeff tuning alone cannot overcome disc saturation for HalfCheetah
- Action: Use AdaptPref (Run D config) as the primary Adaptive SAIL baseline
- Update CLAUDE.md to reflect that HalfCheetah requires `--pref_rank_disc`

**If Runs A or B succeed**:
- Conclusion: entcoeff=0.05 is the missing piece; either TF uses implicit higher entropy or TF vanilla also fails
- Action: Run 3 seeds, document optimal entcoeff, add to CLAUDE.md defaults

---

## 9. Autonomous Testing Loop (Proposal)

```
1. Submit: sbatch HC_Adaptive_entcoeff005.sbatch HC_Adaptive_ec005_disc20.sbatch
            HC_Vanilla_entcoeff005.sbatch HC_AdaptPref_rebase.sbatch
            (all simultaneously — independent runs)

2. Monitor at 30-min intervals:
   grep -E "disc_loss|surr|ep_rew" sail_sb3/logs/HC_*ec005*.out | tail -20

3. Early failure detection at step 50k:
   if disc_loss < 0.15 → saturating again → queue new run with entcoeff=0.1

4. Success detection at step 200k:
   if ep_rew > -400 → learning! → let run complete to 1M steps

5. Decision tree:
   A succeeds → entcoeff fix is sufficient (RC1 confirmed)
   B > A → disc budget also matters (RC2 confirmed)  
   C fails but D succeeds → pref ranking is load-bearing (RC3 confirmed)
   All fail → deeper architectural issue; escalate
```

---

## Critical Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `sail_sb3/algorithms/sail.py` | PyTorch SAIL; disc trigger | 76 (disc check), 152 (TD3 loop) |
| `sail_sb3/reward_models/adversary.py` | Discriminator; entropy loss | 29 (entcoeff), 137 (entropy_loss) |
| `sail_sb3/scripts/train_sail.py` | CLI; hyperparameter wiring | 125 (--entcoeff), 281 (disc_train_freq) |
| `sail_sb3/utils/callbacks.py` | Adaptive promotion | 80 (threshold check), 86 (promotion) |
| `stable-baselines/stable_baselines/td3/sail.py` | TF reference | 1651 (policy loop), 1672 (disc loop) |
| `stable-baselines/hyperparams/sail.yml` | TF HalfCheetah HPs | lines 4–15 |
| `sail_sb3/logs/HC_Adaptive_46667331_1.out` | Latest baseline log | — |

---

## New sbatch Files Created

| File | Run | Key Change |
|------|-----|-----------|
| `HC_Adaptive_entcoeff005.sbatch` | A | `--entcoeff 0.05` |
| `HC_Adaptive_ec005_disc20.sbatch` | B | `--entcoeff 0.05 --disc_gradient_steps 20` |
| `HC_Vanilla_entcoeff005.sbatch` | C | No `--adaptive`, `--entcoeff 0.05` |
| `HC_AdaptPref_rebase.sbatch` | D | `--pref_rank_disc --pref_rank_weight 0.1 --pref_rm <path>` |

All run with `--array=1-3` (seeds 1, 2, 3) for statistical validity.

---

## Execution Results (~50-60k steps, 2026-04-03)

### Summary

| Run | Job ID | disc_loss @50k | ep_rew @50k | Verdict |
|-----|--------|---------------|-------------|---------|
| A (Adaptive+ec05) | 46670039 | s2:0.325, s3:0.386 | s2:-526, s3:-521 | Mixed — seed 1 showing bounce-back promise |
| B (Adaptive+ec05+gs20) | 46670042 | s2:0.218 | s2:-558 | ❌ EARLY FAILURE — gs=20 accelerates saturation |
| C (Vanilla+ec05) | 46670069 | s2:0.462, s3:0.523 | **s2:-107** | ✅ LEARNING CONFIRMED (seed 2) |
| D (AdaptPref) | 46670070 | s2:~0.48, s3:0.482 | s3:-490 | Mixed — disc stable, ep_rew improving slowly |

### Key finding: entcoeff=0.05 enables a "disc bounce-back" mechanism

In successful runs (Run C seed 2), the discriminator undergoes a two-phase dynamic:
1. **Phase 1 (12k-22k)**: disc_loss drops rapidly (4.66 → 0.556) — same as baseline
2. **Phase 2 (22k-28k)**: disc_loss BOUNCES BACK (0.556 → 0.609) — entropy term wins
3. **Phase 3 (28k+)**: disc_loss stabilizes at 0.46-0.61 → surr_reward stable at 0.22-0.29 → ep_rew improves from -490 to **-101**

This bounce-back did NOT occur in the baseline (entcoeff=0.01) because the entropy term (0.001 × H[D]) was too weak to reverse disc convergence. With 0.05 × H[D], it is sometimes strong enough.

### Confirmed findings

1. ✅ **entcoeff=0.05 can enable learning**: Run C seed 2 improved from -490 to -101 in 22k steps (unprecedented in this codebase)
2. ❌ **disc_gradient_steps=20 hurts**: Confirmed early failure — do NOT combine with entcoeff=0.05
3. ⚠️ **High seed variance**: entcoeff=0.05 is necessary but not always sufficient; some seeds still fail to bounce back
4. ⚠️ **AdaptPref not clearly dominant**: More stable disc but slower policy improvement than the best Run C seed
5. 📍 **Next step**: entcoeff=0.1 for more robust learning across all seeds (see updated minimal fix sequence below)

### Updated minimal fix sequence (post-execution)

**COMPLETED FIXES (entcoeff=0.05 confirmed working):**
- C-s2: ep_rew = **7330** at 1M steps ✅
- C-s3: ep_rew = **7090** at 1M steps ✅
- A-s1: **CRASHED** at 236k on first promotion → teacher_buffer.py dones shape bug
- A-s3: **CRASHED** at 507k on first promotion → same bug

**NEW BLOCKING BUG FOUND: `teacher_buffer.py` line 32**

Both Adaptive seeds that reached expert level (A-s1: 5870, A-s3: 6340) crashed immediately on their first promotion attempt:

```
RuntimeError: Tensors must have same number of dimensions: got 1 and 2
  File sail_sb3/datasets/teacher_buffer.py, line 128:
    self.dones = torch.cat([self.dones, new_dones], dim=0)
```

Root cause: `self.dones` initialized as shape `(N,)` on line 32, but `add_episode()` creates `new_dones` of shape `(T, 1)`. Fix is one line.

**REMAINING FIXES (do NOT implement until approved):**

1. Fix `teacher_buffer.py` line 32: `torch.zeros(N,)` → `torch.zeros(N, 1,)` — unblocks adaptive promotion
2. Test entcoeff=0.1 for 3/3 seed reliability (entcoeff=0.05 succeeds ~2/3 seeds)
