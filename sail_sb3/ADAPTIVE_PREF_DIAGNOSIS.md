# Diagnosis of SAIL PyTorch Failures (Adaptive & PrefRank)

The current PyTorch implementation suffers from a critical code bug in the adaptive promotion loop and a behavioral mismatch (likely discriminator saturation) that prevents Vanilla and standard Adaptive modes from learning effectively, whereas the PrefRank variant succeeds but crashes.

## 1. Primary Findings

### A. The Crash (Adaptive + PrefRank)
The only variant that successfully learned (`AdaptPref` reached ~5600 reward) crashed upon the first student-to-teacher promotion.
- **Error**: `TypeError: object of type 'generator' has no len()` at line 87 in `sail_sb3/utils/callbacks.py`.
- **Cause**: `EpisodeBuffer.get_episode_return()` is implemented as a generator, but the callback attempts to call `len()` on it and would double-iterate it (once for length, once for array conversion).

### B. Behavioral Mismatch (Vanilla & Adaptive)
Vanilla and standard Adaptive runs are stable at ~ -600 rewards (HalfCheetah).
- **Observation**: Discriminator loss in these runs is near-zero (~0.005), and surrogate rewards are ~0.0018.
- **Diagnosis**: The discriminator is "perfectly" classifying students as non-expert, leading to reward collapse.
- **The PrefRank Exception**: The `AdaptPref` run learned to ~5600 because the ranking loss regularizes the discriminator (loss ~0.4), preventing it from saturating and killing the learning signal.

## 2. Parity Comparison (TF vs PyTorch)

| Feature | TF (Source of Truth) | PyTorch (sail_sb3) | Match? |
| :--- | :--- | :--- | :--- |
| **Vanilla Reward** | `-log(1 - p)` | `-log(1 - prob + 1e-8)` | Yes |
| **EntCoeff** | `1e-3` (w/ normalization) | `0.01` (tuned for scratch) | Likely Divergent |
| **Observation Norm** | `VecNormalize` (common) | **Missing** | **Mismatch** |
| **Adaptive Logic** | `td3/sail.py` hooks | `utils/callbacks.py` | Implementation Parity achieved |
| **PrefRank Loss** | Bradley-Terry | Bradley-Terry | Yes |

## 3. Root Causes
1. **Critical Bug**: `TypeError` in `callbacks.py` blocks all trajectory promotions.
2. **Saturation**: The lack of observation normalization in PyTorch causes the discriminator to capture the raw state distribution perfectly too early, zeroing out rewards.
3. **Thresholding**: Initial expert threshold ~6741 is high; if the student never gets a signal (due to saturation), it never promotes, making Adaptive mode identical to failing Vanilla mode.

## 4. Minimal Fix Plan

### Step 1: Fix the Callback Promotion Bug
Convert `get_episode_return()` to return a list or wrap it in `list()` in `callbacks.py` to allow `len()` and multiple accesses.

### Step 2: Fix EpisodeBuffer format
Ensure the tuple returned by `EpisodeBuffer` exactly matches what `TeacherBuffer.add_episode` expects (currently there is a mismatch in indexing or expected fields).

### Step 3: Address Discriminator Saturation
- Evaluate if adding `VecNormalize` (SB3 wrapper) or increasing `entcoeff` is necessary to match the learning signal quality of `AdaptPref`.
- **Recommendation**: Add a check to `train_sail.py` to enable observation normalization by default for HalfCheetah.

## 5. Rerun Plan
1. **Fix** the code in `callbacks.py` and `episode_buffer.py`.
2. **Rerun** `HC_SAIL_AdaptPref.sbatch` to confirm promotion no longer crashes.
3. **Rerun** `HC_SAIL_Vanilla.sbatch` with observation normalization to fix the baseline learning.
