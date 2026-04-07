# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Purpose

This is a **research reproduction repository** for Self-Adaptive Imitation Learning (SAIL) and preference-based variants (PAIL, TAC, Q-pref). Claude is a **research assistant**, not an autonomous coder. All algorithmic changes require explicit user approval.

---

## =4 CRITICAL: Active vs. Legacy Codebases =4

### **DEFAULT: Work in `sail_sb3/` (PyTorch/SB3)**

**Active Development Codebase:**
- **`sail_sb3/`**  PyTorch + Stable-Baselines3 implementation
- **All new development, debugging, and experiments happen here by default**
- Main entry point: [sail_sb3/scripts/train_sail.py](sail_sb3/scripts/train_sail.py)
- Python environment: `/home/sohams/miniconda3/envs/sail_sb3_env/bin/python`

**Legacy Reference Codebase (READ-ONLY unless explicitly instructed):**
- **`stable-baselines/`**  TensorFlow 1.x implementation (original research code)
- **Use ONLY for:**
  - Parity checking (comparing behavior against original)
  - Reverse engineering original implementation details
  - Understanding algorithm logic from the paper
- **DO NOT edit unless explicitly instructed to modify TensorFlow code**
- When asked to "check the old implementation" or "compare against TF", inspect but DO NOT modify
- Implement any discovered insights in `sail_sb3/` instead

**Workflow Rule:**
```
User: "Add feature X to SAIL"
� Implement in sail_sb3/ (NOT stable-baselines/)

User: "How did the original code handle Y?"
� Read stable-baselines/ to understand
� Explain the logic
� If change needed, implement in sail_sb3/

User: "Fix the TensorFlow implementation"
� OK to edit stable-baselines/ (explicit instruction)
```

---

## STRICT Reproduction Rules

1. **No algorithmic changes** to either codebase without explicit user approval
2. **`sail_sb3/` must match TensorFlow 1.x SAIL behavior exactly**
3. **Full-episode preference ranking** (NOT segment-based like PebBLE)
4. **TimeFeatureWrapper is MANDATORY** for demo compatibility (see below)
5. **Reward formula: `softplus(logits)`**  numerically stable GAIL reward
   - Do NOT switch to `-log(1 - sigmoid(logits))` formulation without approval
   - Do NOT modify reward computation without approval
6. When implementing new features in `sail_sb3/`, verify against TF reference if available

---

## Required Workflow (MANDATORY)

**For all code changes:**
```
1. Read existing implementation code (from sail_sb3/ by default)
2. Explain current logic/behavior
3. Propose specific change with justification
4. WAIT for user approval
5. Only then implement (in sail_sb3/ unless told otherwise)
```

**When comparing implementations:**
```
1. Read stable-baselines/ code (TF reference)
2. Read sail_sb3/ code (active PyTorch)
3. Explain differences/logic
4. Propose changes to sail_sb3/ if needed
5. WAIT for approval
6. Implement in sail_sb3/ (unless explicitly told to edit TF code)
```

---

## Debugging Requirements (ALWAYS ENABLE)

**In [sail_sb3/scripts/train_sail.py](sail_sb3/scripts/train_sail.py), always use `--debug` flag during development.**

### Must Print (Already Implemented):

 **Expert trajectory returns** (mean/std/min/max)  printed at startup (lines 169-186)
- **CRITICAL**: Compute returns at runtime. Do NOT trust dataset filename (e.g., "5600-score" may be inaccurate).
- Use printed statistics as ground truth for expert quality.

 **Discriminator logits** (expert vs policy)  printed when `--debug` enabled (lines 110-129 in [algorithms/sail.py](sail_sb3/algorithms/sail.py))
- Monitor expert_prob and policy_prob to detect saturation

 **Surrogate reward statistics** (mean/min/max/std)  printed when `--debug` enabled (lines 145-156 in [algorithms/sail.py](sail_sb3/algorithms/sail.py))
- Check reward is not collapsing to ~0

 **Buffer statistics** (done fraction, batch size)  printed when `--debug` enabled

� **RM score distribution** (if using preference RM)  check [datasets/teacher_buffer.py](sail_sb3/datasets/teacher_buffer.py)

� **Gradient norms**  add if missing to ensure non-zero gradients

### Debug Output Locations in `sail_sb3/`:
- Startup diagnostics: [train_sail.py:169-186](sail_sb3/scripts/train_sail.py#L169-L186)
- Discriminator: [algorithms/sail.py:110-129](sail_sb3/algorithms/sail.py#L110-L129)
- Rewards: [algorithms/sail.py:145-156](sail_sb3/algorithms/sail.py#L145-L156)
- Critic: [algorithms/sail.py:174-178](sail_sb3/algorithms/sail.py#L174-L178)

---

## Training Correctness Criteria

### Learning (Expected Behavior):
- `rollout/ep_rew_mean` increasing over time
- Surrogate reward mean > 0.1 (typically 0.55.0)
- Discriminator logits: `expert_prob > 0.5`, `policy_prob < 0.5` (but not saturated)
- Q-values increasing

### Collapse (Training Failure):
- Surrogate reward � 0 (mean < 0.01)
- Discriminator saturated: `expert_prob � 1.0` or `policy_prob � 0.0`
- Q-values flat or decreasing
- Rollout reward not improving

### Expected Scales:
- **Surrogate rewards**: typically 0.55.0 range
- **Expert data sanity**: Compute returns at runtime (see startup logs). Do NOT assume filename reflects actual returns.

---

## Reward Flow (Exact Pipeline in `sail_sb3/`)

```
(obs, action) � Discriminator.forward(obs, action) � logits
             � Discriminator.get_reward(obs, action)
             � surrogate_reward = F.softplus(logits)  # Numerically stable GAIL reward
             � SAIL.train(): target_q = surrogate_reward + � * Q_target(s', a')
             � Critic loss = MSE(Q(s,a), target_q)
             � Actor trained to maximize Q(s, �(s))
```

**Code Locations:**
- Discriminator: [sail_sb3/reward_models/adversary.py:119-125](sail_sb3/reward_models/adversary.py#L119-L125)
- Reward replacement: [sail_sb3/algorithms/sail.py:138-141](sail_sb3/algorithms/sail.py#L138-L141)
- Bellman target: [sail_sb3/algorithms/sail.py:172](sail_sb3/algorithms/sail.py#L172)

**IMPORTANT**: Do NOT switch between `softplus(logits)` and `-log(1 - sigmoid(logits))` formulations without explicit approval. The current implementation uses `softplus` for numerical stability.

---

## Implementation Status

### Active Codebase (`sail_sb3/`)  PyTorch/SB3:
-  Vanilla SAIL (TD3 + discriminator)
-  Discriminator with WGAN-GP + entropy regularization
-  Preference ranking loss (full-episode, online)
-  TimeFeatureWrapper
-  Debug logging (`--debug` flag)
-  **PAIL** (adaptive teacher buffer replacement)  IMPLEMENTED (sail_sb3/utils/callbacks.py + sail_sb3/datasets/teacher_buffer.py); promotion dones-shape bug fixed 2026-04-03
- L **TAC** (Temperature-Adaptive Clipping)  NOT implemented
- L **Q-pref loss on critic**  NOT implemented
- Entry point: [sail_sb3/scripts/train_sail.py](sail_sb3/scripts/train_sail.py)

### Legacy Reference (`stable-baselines/`)  TensorFlow 1.x:
-  Vanilla SAIL
-  PAIL (adaptive teacher replacement)
-  TAC (Temperature-Adaptive Clipping)
-  Q-pref, Pref-rank variants
- Entry point: [stable-baselines/run/train_sail.py](stable-baselines/run/train_sail.py)
- **Use only for reference/comparison**

---

## Running Commands

### PRIMARY: PyTorch SAIL (`sail_sb3/`)  Use for All New Experiments

```bash
# Always use explicit python path on HPC (no conda activate)
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python

# Basic SAIL training
$PYTHON sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz \
  --total_timesteps 1000000 \
  --seed 0 \
  --debug  # ALWAYS use during development

# With preference ranking loss
$PYTHON sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz \
  --total_timesteps 1000000 \
  --pref_rank_disc \
  --pref_rm /path/to/reward_model.pt \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --debug

# SLURM submission
sbatch sail_sb3/HC_SAIL_PrefR.sbatch
```

### REFERENCE ONLY: TensorFlow SAIL (`stable-baselines/`)  Do Not Use Unless Instructed

```bash
# Only run if explicitly told to test TF implementation
/path/to/tf1-venv/bin/python stable-baselines/run/train_sail.py \
  --env HalfCheetah-v2 \
  --seed 0 \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --n-episodes 4 \
  --log-dir /path/to/logs
```

---

## HPC Execution Constraints

- **Always use explicit Python binary path:**
  - PyTorch: `/home/sohams/miniconda3/envs/sail_sb3_env/bin/python`
  - TensorFlow (legacy): `~/.venvs/sail-tf1b/bin/python` or similar
- **Never rely on `conda activate` or `source activate` in sbatch scripts**
- **Required environment variables:**
  ```bash
  export MUJOCO_GL=egl                    # Headless rendering
  export OMP_NUM_THREADS=1                # Prevent thread contention
  export TF_CPP_MIN_LOG_LEVEL=2           # Silence TF warnings (TF only)
  ```
- SLURM scripts must specify full paths in shebang and execution

---

## Architecture Overview

### Active Development: `sail_sb3/` (PyTorch)

```
sail_sb3/
   scripts/
      train_sail.py              # Main entry point (USE THIS)
   algorithms/
      sail.py                    # SAIL class (subclasses SB3 TD3)
   reward_models/
      adversary.py               # Discriminator (PyTorch)
      preference_model.py        # Preference RM wrapper
      pref_rm_eval.py           # RM evaluation utilities
   datasets/
      teacher_buffer.py          # Expert data loader
   utils/                         # SB3 callbacks, wrappers
```

### Legacy Reference: `stable-baselines/` (TensorFlow 1.x)

```
stable-baselines/
   run/
      train_sail.py              # TF training script (reference only)
      settings.py                # Per-env configs
      *.sbatch                   # SLURM scripts (legacy)
   stable_baselines/
      td3/
         sail.py                # SAIL algorithm (TF)  read for logic
      gail/
          adversary.py           # Discriminator (TF)  read for architecture
   hyperparams/
       *.yml                      # Hyperparameter configs (copy to sail_sb3 if needed)
```

### Shared Resources:

```
teacher_dataset/                   # Sub-optimal expert demonstrations
   expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
   expert_data_no_img_Ant_scores_1200_episodes_4.npz
   ...                            # Named: {ENV}_scores_{SCORE}_episodes_{N}.npz
                                   # WARNING: Filenames may not reflect actual returns!

pebble_runner/                     # Offline preference RM training scripts
   train_pref_reward_py37_v4.py
```

---

## Key Implementation Details

### TimeFeatureWrapper (MANDATORY in `sail_sb3/`)

**Location**: [sail_sb3/scripts/train_sail.py:23-52](sail_sb3/scripts/train_sail.py#L23-L52)

- Appends time-remaining feature `[1 � 0]` to observations
- **HalfCheetah-v2**: 17-dim native � **18-dim with wrapper**
- **CRITICAL**: Expert demos collected WITH wrapper applied
- Skipping causes dimension mismatch or distribution shift � training fails
- Applied in `make_env_with_time()` factory function

**Wrapping Order (inside-out):**
```
gym.make(env_id) � Monitor � TimeFeatureWrapper
```

Monitor MUST be applied before TimeFeatureWrapper so SB3 can read episode info dict.

---

### Discriminator Architecture (`sail_sb3/`)

**Location**: [sail_sb3/reward_models/adversary.py](sail_sb3/reward_models/adversary.py)

- Two hidden layers (256 units), tanh activation
- Input: concatenated (state, action)
- Output: scalar logits (pre-sigmoid)

**Regularization:**
- WGAN-GP (gradient penalty): `gradcoeff=10.0`
- Entropy loss: `entcoeff=0.05` (**confirmed working default** — prevents saturation; 0.01 is insufficient)

---

### Discriminator Training Schedule (`sail_sb3/`)

**Location**: [sail_sb3/algorithms/sail.py:65-130](sail_sb3/algorithms/sail.py#L65-L130)

- Updated every `disc_train_freq` **environment steps** (default: 200)
- **NOT** updated every gradient step (would overpower policy)
- Each update: `disc_gradient_steps=10` batches
- Batch size: `disc_batch_size=256` (independent from critic batch)

**Key logic:**
```python
if self.num_timesteps - self._last_disc_update_step >= self.disc_train_freq:
    # Update discriminator for disc_gradient_steps batches
    # ...
    self._last_disc_update_step = self.num_timesteps
```

---

### Reward Computation (`sail_sb3/`)

**Location**: [sail_sb3/reward_models/adversary.py:119-125](sail_sb3/reward_models/adversary.py#L119-L125)

**Formula:**
```python
reward = F.softplus(logits)  # Numerically stable GAIL reward
```

**Note**: This is equivalent to `-log(1 - sigmoid(logits))` but avoids `log(0)` issues. The `softplus` formulation is numerically stable and matches the GAIL paper.

**Do NOT modify without explicit approval.**

**Usage in SAIL:**
- Replaces environment rewards in Bellman target (line 172 of [sail.py](sail_sb3/algorithms/sail.py#L172))
- `target_q = surrogate_rewards + (1 - dones) * gamma * target_q`

---

### Hyperparameter Defaults (`sail_sb3/`)

**Location**: [sail_sb3/scripts/train_sail.py:86-117](sail_sb3/scripts/train_sail.py#L86-L117)

**Matches TF exactly (with confirmed empirical corrections):**
- TD3: `batch_size=256`, `learning_rate=1e-3`, `policy_delay=2`, `tau=0.005`, `gamma=0.99`
- Discriminator: `disc_lr=3e-4`, `disc_batch_size=256`, `disc_train_freq=500`, `disc_gradient_steps=10`
- Regularization: `entcoeff=0.05`, `gradcoeff=10.0`

**⚠️ entcoeff HISTORY**: Original TF default was 0.01, but this causes discriminator saturation in PyTorch (confirmed empirically: ep_rew stuck at -600 for all 1M steps). `entcoeff=0.05` confirmed to achieve ep_rew 7090–7330 on HalfCheetah. Do NOT revert to 0.01 without a specific reason. Testing `entcoeff=0.1` for further reliability improvement (jobs 46694874, 46694875).

**⚠️ disc_gradient_steps warning**: Do NOT increase `disc_gradient_steps` beyond 10. Setting it to 20 was tested (job 46670042) and 2/3 seeds failed — more disc updates accelerates saturation without higher entcoeff, and with higher entcoeff it makes learning unstable.

---

## Common Issues & Solutions (for `sail_sb3/`)

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **Reward collapse** | Surrogate reward mean < 0.05 | Verify `--entcoeff 0.05` (new default); if still collapsing try `--entcoeff 0.1` |
| **Dimension mismatch** | "Expected obs dim X, got Y" | Verify TimeFeatureWrapper applied to env AND check expert data compatibility |
| **Expert returns mismatch** | Startup shows returns != filename | **This is expected!** Use runtime-computed returns as ground truth. Ignore filename. |
| **Discriminator saturation** | disc_loss < 0.15 after 50k steps | `entcoeff=0.05` is now default; if still saturating use `--entcoeff 0.1` |
| **No learning** | Rollout reward flat | Check surrogate reward > 0.15; if reward < 0.1 disc is saturated — increase entcoeff |
| **Adaptive promotion crash** | `RuntimeError: Tensors must have same number of dimensions: got 1 and 2` in `teacher_buffer.py:128` | **FIXED** (2026-04-03): `sail_sb3/datasets/teacher_buffer.py` lines 30-37 now use `reshape(-1,1)` / `zeros(N,1)` |
| **Grad explosion** | Loss → NaN | Reduce `--learning_rate 3e-4` or increase `--gradcoeff 20` |

---

## Comparison Workflow (TF vs. PyTorch)

When asked to verify behavior matches the original:

1. **Read TF implementation** ([stable-baselines/stable_baselines/td3/sail.py](stable-baselines/stable_baselines/td3/sail.py))
2. **Read PyTorch implementation** ([sail_sb3/algorithms/sail.py](sail_sb3/algorithms/sail.py))
3. **Compare**:
   - Training schedule (when disc/critic/actor updated)
   - Loss functions (BCE, entropy, GP)
   - Reward formula
   - Hyperparameters
4. **Document differences** (e.g., "TF uses freq=500, PyTorch uses 200")
5. **Propose fix to PyTorch** (if mismatch found)
6. **WAIT for approval**
7. **Edit `sail_sb3/` only** (unless told to fix TF code)

---

## Summary

- **Default workspace**: `sail_sb3/` (PyTorch)
- **Legacy reference**: `stable-baselines/` (TensorFlow, read-only)
- **Always use `--debug` flag** during development
- **Compute expert returns at runtime** (ignore filename)
- **Reward formula**: `softplus(logits)` (do not change)
- **TimeFeatureWrapper is mandatory** (HalfCheetah: 17�18 dims)
- **Workflow**: read � explain � propose � WAIT � implement
