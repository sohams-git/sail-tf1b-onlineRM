# Adaptive SAIL Next Blocker Diagnosis
## STATUS (LATEST)
- Phase 1: Callback + crash fix → DONE
- Phase 2: TF-parity normalization → DONE
- Phase 3: Training balance fix → IN PROGRESS
- Current issue: Adaptive SAIL still not learning (job 46667331)

## 1. Log Inspected
- **PyTorch Adaptive (46666713)**: 141k steps, `ep_rew_mean` at **-601**.
- **Signals**:
  - `disc_loss`: ~0.04 (improved from 0.005, but still extremely low).
  - `surrogate_reward_mean`: **0.012 - 0.014** (extremely small).
  - `actor_loss`: Slowly moving, but no learning signal from discriminator.
  - **No promotions** ever occurred.

## 2. Current Adaptive Behavior
The policy is stuck in a plateau because the discriminator is **dominating** the adversarial game. The student transitions are assigned a near-zero expert probability ($p \approx 0.01$), resulting in a flat, near-zero reward signal ($r \approx 0.012$). The normalization fix prevented a total binary collapse (loss 0.005) but did not restore the adversarial balance.

## 3. What Changed from Previous Fix
The internal observation normalization (obs_rms) is active and being updated (as shown by transition from 0.005 loss to 0.04 loss). However, the discriminator is still too accurate / overconfident.

## 4. Remaining Mismatch vs TF
I have identified two critical mismatches in the training regimen:

1. **Update Regimen (Training Intensity)**:
   - **TF `HalfCheetah-v2`**: `train_freq: 1000` / `gradient_steps: 1000` (1 update per step).
   - **PyTorch Default**: `train_freq: 200` / `gradient_steps: 10` (1 update per 20 steps).
   - **Result**: The policy in TF is updated **20x more often** relative to the discriminator's training interval.

2. **Discriminator Interval**:
   - **TF `SAIL`**: `train_discriminator_freq: 500`.
   - **PyTorch Default**: `disc_train_freq: 200`.
   - **Result**: The PyTorch discriminator is updated **2.5x more frequently** than TF's, giving it an unfair advantage in the min-max game.

3. **Gradient Penalty (Normalization Bug)**:
   - A code inspection reveals that the `_gradient_penalty` method in PyTorch's `Adversary` currently operates on **unnormalized** states, while the main network weights are trained on **normalized** states (via `forward`). This decouples the GP from the actual network weights, likely leading to overconfidence or instability.

## 5. Highest-Priority Blocker
**The Mismatched Training Regimen (Update Balance).**
Even with a perfect discriminator, if the policy is trained 20x less often than in the original study, it cannot adapt to the shifting reward surface. Combined with a discriminator that iterates 2.5x faster, this is a recipe for total adversarial collapse.

## 6. Minimal Next Fix
1. **Sync Sbatch/Defaults**: Update the sbatch files to use `train_freq=1000` and `gradient_steps=1000` to match the TF `sail.yml` for HalfCheetah-v2.
2. **Update `disc_train_freq`**: Change the code/default to `500` steps instead of `200` to match TF.
3. **Fix `_gradient_penalty`**: Ensure observations are normalized **before** computing the gradient penalty, matching the TF graph structure.

## 7. What to Rerun Next
- `sbatch HC_SAIL_Adaptive.sbatch` with the updated training frequencies and the GP fix.
- Expectation: `disc_loss` should rise to 0.1-0.3 and the student should begin learning.
