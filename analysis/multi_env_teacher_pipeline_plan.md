# Multi-Environment Teacher Pipeline — Implementation Plan

**Scope:** Extend the working HalfCheetah teacher pipeline to Ant-v2, Walker2d-v2,
Swimmer-v2, and Hopper-v2 in a principled, reusable way.

---

## 1. What is Shared vs. Environment-Specific

### Shared (one script handles all envs)

| Component | Script | Key argument |
|---|---|---|
| Teacher training | `stable-baselines/scripts/train_teacher_checkpoints.py` | `--env_id` |
| Dataset generation | `stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py` | `--env_id` |

Both scripts are fully isolated from TF1 / sail_sb3 code. `TimeFeatureWrapper` is
embedded in each script (no external imports).

### Environment-specific

| Component | What differs |
|---|---|
| Teacher training sbatch | `--env_id`, `--target_return`, log/save paths |
| Dataset directories | one per env under repo root |
| SAIL sbatch directories | one per env under `sail_sb3/` |
| Obs / act dim | auto-inferred from gym — no hardcoding |

---

## 2. Generalization Changes

### A. Training script (generalized from HC-specific)

**Old:** `train_halfcheetah_teacher_checkpoints.py` — hardcodes `halfcheetah` subdir.

**New:** `train_teacher_checkpoints.py` — derives subdir from `--env_id`:
```python
env_short = args.env_id.split("-")[0].lower()   # e.g. "ant", "walker2d"
run_root  = os.path.join(args.save_root, env_short)
```
Everything else (TD3, TimeFeatureWrapper, checkpoint logic, CSV logging) is identical.

### B. Dataset generation script (already mostly generic)

**Fix:** Remove hardcoded `obs.shape[1] == 18` and `acts.shape[1] == 6` assertions in
`_verify()`. Replace with dynamic checks (2D shape consistency, episode count). Obs and
act dims are auto-inferred from the gym environment at rollout time.

---

## 3. Training Parameterization Per Environment

| Env | `--env_id` | `--target_return` | Total episodes | Checkpoint every | EP_LEN note |
|---|---|---|---|---|---|
| HalfCheetah-v2 | HalfCheetah-v2 | 6988 | 2000 | 100 | Always 1000 steps |
| Ant-v2 | Ant-v2 | 3000 | 2000 | 100 | Always 1000 steps |
| Walker2d-v2 | Walker2d-v2 | 2500 | 2000 | 100 | Early-termination env |
| Swimmer-v2 | Swimmer-v2 | 200 | 2000 | 100 | Always 1000 steps |
| Hopper-v2 | Hopper-v2 | 2500 | 2000 | 100 | Early-termination env |

**Early-termination note:** Walker2d and Hopper end episodes early (fall over). Training
chunks are sized as `checkpoint_every × 1000 steps`. For these envs, actual episode count
per chunk will exceed `checkpoint_every` (more short episodes fit in the same step budget).
Checkpoint naming uses the step count at eval time; the episode label is approximate.

---

## 4. SAIL Dataset Compatibility Across Environments

The dataset format is identical for all envs — only dims differ:

| Key | Shape | Dtype | All envs |
|---|---|---|---|
| `obs` | `(N, obs_dim+1)` | float32 | +1 = TimeFeature |
| `actions` | `(N, act_dim)` | float32 | |
| `rewards` | `(N, 1)` | float32 | 2D — SAIL requires this |
| `episode_returns` | `(n_trajs,)` | float64 | |
| `episode_starts` | `(N,)` | object | True at start, np.array([False]) elsewhere |

Confirmed against existing SAIL datasets:
- Ant: obs=(N,112), act=(N,8)
- Walker2d: obs=(N,18), act=(N,6) — early termination, N < n_trajs×1000
- Swimmer: obs=(N,9), act=(N,2)
- Hopper: obs=(N,12), act=(N,3) — early termination

The `_verify()` function checks all keys, shapes, dtypes, and episode-count consistency
without hardcoding environment-specific dims.

---

## 5. Directory Layout

```
sail-tf1b-onlineRM/
  stable-baselines/scripts/
    train_teacher_checkpoints.py          # Generalized teacher training (all envs)
    train_halfcheetah_teacher_checkpoints.py  # Legacy HC-only (kept, not modified)
    generate_sail_dataset_from_checkpoint.py  # Dataset generation (all envs, fixed)

  stable-baselines/run/
    teacher_train_{ENV}_checkpoints.sbatch    # One per env

  teacher_training/
    halfcheetah/  ant/  walker2d/  swimmer/  hopper/
      teacher_checkpoints/
        ep{NNNN}_return{RRR}/   model.zip + meta.json
        best  ->  ...
        latest -> ...
      teacher_logs/
        training_log.csv
      run_config.json

  teacher_dataset_HC/      # HalfCheetah datasets
  teacher_dataset_Ant/
  teacher_dataset_Walker2d/
  teacher_dataset_Swimmer/
  teacher_dataset_Hopper/
    expert_data_no_img_{ENV}_scores_{R}_episodes_{N}_ep{E}.npz
    *_gen_meta.json

  sail_sb3/
    HC_sbatch/          # HalfCheetah sbatch library
    Ant_sbatch/
    Walker2d_sbatch/
    Swimmer_sbatch/
    Hopper_sbatch/
      {ENV}_smoke_{CHECKPOINT}_{VARIANT}.sbatch
      {ENV}_full_{CHECKPOINT}_{VARIANT}.sbatch
    logs/               # Full-run logs (all envs)
    HC_sbatch_smoke_logs/
    Ant_sbatch_smoke_logs/
    Walker2d_sbatch_smoke_logs/
    Swimmer_sbatch_smoke_logs/
    Hopper_sbatch_smoke_logs/
```

---

## 6. Naming Conventions

### Teacher training outputs
```
teacher_training/{env_short}/teacher_checkpoints/ep{NNNN}_return{RRR}/
teacher_training/{env_short}/teacher_checkpoints/target{T}_reached_ep{NNNN}_return{RRR}/
teacher_training/{env_short}/teacher_checkpoints/best  ->  ...
teacher_training/{env_short}/teacher_checkpoints/latest -> ...
teacher_training/{env_short}/teacher_logs/training_log.csv
```
`env_short` = lowercase first token of env_id: `ant`, `walker2d`, `swimmer`, `hopper`

### Generated datasets
```
teacher_dataset_{ENV}/expert_data_no_img_{ENV}_scores_{ACTUAL_RETURN}_episodes_{N}_ep{TRAINING_EP}.npz
```
`ENV` = first token of env_id: `Ant`, `Walker2d`, `Swimmer`, `Hopper`

### Sbatch files
```
sail_sb3/{ENV}_sbatch/{ENV}_smoke_{CHECKPOINT}_{VARIANT}.sbatch   # 10k steps
sail_sb3/{ENV}_sbatch/{ENV}_full_{CHECKPOINT}_{VARIANT}.sbatch    # 1M steps
```

### Teacher training sbatch
```
stable-baselines/run/teacher_train_{env_short}_checkpoints.sbatch
```

---

## 7. Validation Steps Per Environment

After teacher training completes:
1. Confirm `training_log.csv` exists and has rows for each checkpoint
2. Confirm `best/` and `latest/` symlinks resolve

After dataset generation:
1. `_verify()` inside the script — checks all keys, shapes, dtypes, episode count
2. Confirm `rewards.shape == (N, 1)` (not (N,))
3. Confirm `episode_starts.dtype == object` and boundary count == n_trajs
4. Confirm `obs.shape[1]` matches `env.observation_space.shape[0] + 1`

Smoke test (after dataset + sbatch exist):
1. Run 10k steps with `--debug`
2. Check expert returns print at startup
3. Check discriminator training begins (no KeyError, shape error)
4. Check EXIT CODE == 0

---

## 8. Current Pipeline State (as of 2026-04-19)

| Stage | HC | Ant | Walker2d | Swimmer | Hopper |
|---|---|---|---|---|---|
| Generalized training script | → creating now | ← uses same | ← | ← | ← |
| Dataset gen script (env-agnostic _verify) | → fixing now | ← | ← | ← | ← |
| Teacher training sbatch | ✓ | → creating | → | → | → |
| Teacher training job | ✓ done | → submitting | → | → | → |
| Datasets | ✓ (ep400,ep1000) | → after training | → | → | → |
| Sbatch library | ✓ HC_sbatch/ | → after training | → | → | → |
