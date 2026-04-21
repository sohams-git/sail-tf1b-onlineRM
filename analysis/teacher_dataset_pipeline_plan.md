# Teacher Dataset Pipeline — Implementation Plan

**Scope:** Principled, reusable pipeline for converting trained teacher checkpoints into
SAIL-compatible datasets and validating them via smoke-test runs.

---

## 1. Checkpoint → Dataset Generation

**Script:** `stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py`

This script is already environment-agnostic and reusable. Key design points:

- **Input:** `--checkpoint_dir`, `--env_id`, `--n_trajs`, `--save_dir`
- **Output:** SAIL-compatible `.npz` with exact keys SAIL's `ExpertDataset` expects
- **n_trajs is an argument**, never hardcoded
- **TimeFeatureWrapper** is embedded in the script (self-contained, no imports from sail_sb3)
- **Informative filename:** `expert_data_no_img_{ENV}_scores_{ACTUAL_RETURN}_episodes_{N}_ep{TRAINING_EP}.npz`
- **Optional canonical name** via `--sail_compat_name`

**Output contract (verified against existing SAIL datasets):**

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `obs` | `(N, obs_dim)` | float32 | 18 for HalfCheetah (17 + TimeFeature) |
| `actions` | `(N, act_dim)` | float32 | |
| `rewards` | `(N, 1)` | float32 | 2D — SAIL expects this shape |
| `episode_returns` | `(n_trajs,)` | float64 | |
| `episode_starts` | `(N,)` | object | `True` at ep start, `np.array([False])` elsewhere |

**Verification:** `_verify()` function inside the script checks all keys, shapes, dtypes,
episode-count consistency, and `demo_dones` derivation.

---

## 2. Dataset Validation

After generation, verify:
1. Load the `.npz` and print all keys + shapes
2. `obs.shape[1]` must match expected obs dim for the environment
3. `actions.shape[1]` must match env action dim
4. `rewards.shape` must be `(N, 1)` not `(N,)`
5. `episode_returns.shape[0]` must equal `n_trajs`
6. `episode_starts` must have dtype=object and count episode boundaries == `n_trajs`

The `_verify()` function inside the generation script performs all of these checks
automatically after saving.

---

## 3. Smoke-Test Sbatch — Without Modifying the Original

**Rule:** Never edit the original sbatch. Create a new one that:
- Is a modified copy of the relevant template
- Points to the new dataset via `--expert_data`
- Uses a reduced `--total_timesteps` (10k for smoke test)
- Has distinct `--job-name` and log paths so it cannot be confused with real runs
- Lives in a dedicated directory: `sail_sb3/HC_sbatch/`

**Naming convention for smoke-test sbatch:**
```
{ENV}_smoke_{CHECKPOINT_TAG}_{ALGO_VARIANT}.sbatch
e.g. HC_smoke_ep1000_AdaptiveLfD.sbatch
```

**Smoke-test success criteria:**
- Job starts without error
- Dataset loads (startup prints expert trajectory returns)
- Training begins (at least through `learning_starts`)
- No `KeyError`, shape mismatch, or dtype errors

---

## 4. Generalization to Other Environments

The pipeline has three reusable components:

### A. Teacher Training (already reusable)
`stable-baselines/scripts/train_halfcheetah_teacher_checkpoints.py` accepts:
- `--env_id` (any MuJoCo gym env)
- `--save_root` (all outputs isolated under this root)
- `--total_episodes`, `--checkpoint_every`, `--n_eval_episodes`, `--target_return`

To support a new environment: run the same script with `--env_id Ant-v2`, etc.
`TimeFeatureWrapper` is embedded and the obs-dim is auto-inferred from the gym env.

### B. Checkpoint Selection
After training, checkpoints are named `ep{N}_return{R}/` and logged in `training_log.csv`.
Selection criteria (choose one per experiment):
- `best/` symlink → highest eval return ever seen
- `latest/` symlink → most recent checkpoint
- `target{T}_reached_ep*/` → first time a specific return threshold was crossed
- Manual: pick any `ep{N}_return{R}/` from `training_log.csv`

### C. Dataset Generation (already reusable)
```bash
python stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py \
    --checkpoint_dir /path/to/ep{N}_return{R} \
    --env_id {ENV_ID} \
    --n_trajs {N} \
    --save_dir /path/to/teacher_dataset_{ENV}/
```

To add a new environment: pass the correct `--env_id`. The script auto-infers obs/act dims
from the gym environment. `TimeFeatureWrapper` is always applied.

### D. Environment-Specific Sbatch Creation
For each environment, create a directory: `sail_sb3/{ENV}_sbatch/`
Inside, maintain sbatch variants:
- `{ENV}_smoke_{CHECKPOINT}_{VARIANT}.sbatch` — 10k step smoke test
- `{ENV}_full_{CHECKPOINT}_{VARIANT}.sbatch` — full-budget run

Template: copy the closest existing sbatch and change `--expert_data`, `--env`, and
budget. The `--expert_data` path is the only required change per environment.

### E. Smoke Test
Same structure as HalfCheetah: 10k steps, distinct job name/logs, check startup.

---

## 5. Current Pipeline State (as of 2026-04-06)

| Stage | Status | Notes |
|---|---|---|
| Teacher training (HC) | ✓ Running (job 46834955) | SB3 TD3, TimeFeatureWrapper, 2M steps |
| Checkpoint at ep1000 | ✓ Available | mean_return=9094.16, ep1000_return9094 |
| Dataset generation script | ✓ Reusable | `generate_sail_dataset_from_checkpoint.py` |
| HC dataset (4 trajs) | → Generated now | `teacher_dataset_HC/` |
| Smoke-test sbatch | → Created now | `sail_sb3/HC_sbatch/HC_smoke_ep1000_AdaptiveLfD.sbatch` |
| Smoke test run | → Pending | job submitted after dataset verification |

---

## 6. Directory Layout

```
sail-tf1b-onlineRM/
  stable-baselines/scripts/
    train_halfcheetah_teacher_checkpoints.py   # Teacher training (reusable)
    generate_sail_dataset_from_checkpoint.py   # Dataset generation (reusable)

  teacher_training/
    halfcheetah/
      teacher_checkpoints/
        ep{N}_return{R}/   model.zip + meta.json
        best  -> ...       symlink
        latest -> ...      symlink
      teacher_logs/
        training_log.csv
      run_config.json

  teacher_dataset_HC/                          # HC-specific datasets
    expert_data_no_img_HalfCheetah_scores_{R}_episodes_{N}_ep{E}.npz
    *_gen_meta.json

  sail_sb3/
    HC_sbatch/                                 # HC sbatch library
      HC_smoke_ep1000_AdaptiveLfD.sbatch       # Smoke test
      (future full-run sbatches here)
    scripts/train_sail.py                      # SAIL entry point
    logs/                                      # Full-run logs
    HC_sbatch_smoke_logs/                      # Smoke-test logs (separate)
```
