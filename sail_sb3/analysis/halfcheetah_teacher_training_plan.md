# HalfCheetah Teacher Training Plan

**Purpose:** One full teacher training run for HalfCheetah-v2 that saves checkpoints every
100 episodes, so we can later generate SAIL teacher datasets from any saved checkpoint.

---

## 1. Source of Truth: What Algorithm Trains the Teacher?

**Algorithm: TD3 (Twin Delayed DDPG)**  
**Framework: stable-baselines TF1 (sail-tf1b)**

Script:
```
sail-tf1b/stable-baselines/scripts/make_halfcheetah_teachers_tf1.py
```

That script is the canonical teacher generator in this repo. All existing `.npz` datasets
in `SAIL/teacher_dataset/` and `sail-tf1b/stable-baselines/teacher_dataset/legacy/` were
produced with TD3 + this script (or a predecessor using the same hyperparameters).

### TD3 Hyperparameters (from the existing script)

| Param | Value |
|---|---|
| Policy | MlpPolicy (layers=[400, 300]) |
| Buffer size | 1,000,000 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Action noise | NormalActionNoise σ=0.1 |
| Gamma | 0.99 |
| Train freq | 1 (update every step) |
| Gradient steps | 1 |
| Episode horizon | 1000 steps |
| Steps ↔ episodes | total_timesteps = episodes × 1000 |

---

## 2. Existing Dataset Format — Exact SAIL-Compatible Schema

Confirmed by inspecting live files:

```
Keys in .npz:
  obs              shape=(N, 18)     float32   ← observations (HalfCheetah state dim)
  actions          shape=(N, 6)      float32   ← actions
  rewards          shape=(N, 1)      float32   ← per-step rewards, 2D not 1D
  episode_returns  shape=(n_traj,)   float64   ← one scalar per trajectory
  episode_starts   shape=(N,)        object    ← True at episode start, False elsewhere
```

**episode_starts format:** dtype=object, first step of each episode is Python `True`,
all other steps are `np.array([False])`. This is a legacy artifact of the original
data generation. Replicate exactly to guarantee compatibility with `ExpertDataset`.

**episode_returns for existing expert (5600-score file):** `[6014, 6004, 5999, 6051]`  
→ the teacher trained for ~500 episodes achieves ~6000 mean return on HalfCheetah-v2.

---

## 3. Score Convention — How SAIL Uses It

### In dataset filenames

| File | Score field | Meaning |
|---|---|---|
| `expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz` | 5600 | `optimal_score` from `settings.py` — a **reference constant**, not actual return |
| `legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_4_legacy_exact.npz` | 500 | Training episode budget used to create this teacher |

### In settings.py

```python
'HalfCheetah-v2': {
    'optimal_score': 5600,   # Used to construct the dataset filename SAIL looks for
    'max_score': 5000,
    'demo_score': 5600,
    ...
}
```

### How train_sail.py finds the dataset

```python
data_save_dir = os.path.join(
    "../../teacher_dataset",
    "expert_data_no_img_{}_scores_{}_episodes_{}".format(
        args.env.split('-')[0],     # "HalfCheetah"
        config['optimal_score'],    # 5600 (hardcoded in settings.py)
        config['n_episodes']        # 1, 4, 10, ...
    )
)
data_save_dir = '{}.npz'.format(data_save_dir)
```

**SAIL automatically looks for** the file named with `scores_5600`. The filename score is
not validated against actual returns — it is just a naming convention.

### Chosen convention for the new pipeline

- **Checkpoint names** use the **actual mean eval return** (integer, e.g. `return4832`)
  so you can immediately see how good each checkpoint is.
- **Generated dataset filenames** use the **actual mean return of the collected rollouts**
  in the `scores_XXX` position.
- To use a generated dataset with SAIL unchanged: either symlink it to the
  `scores_5600_episodes_N` path, or see Section 7 below.

---

## 4. Script to Modify / Create

### New: `train_halfcheetah_teacher_checkpoints.py`

Location: `sail-tf1b/stable-baselines/scripts/train_halfcheetah_teacher_checkpoints.py`

Does NOT modify `make_halfcheetah_teachers_tf1.py` — that script stays as-is.

**What the new script does:**
1. Trains TD3 in chunks of `--checkpoint_every` episodes (default 100).
2. After each chunk, runs `--n_eval_episodes` deterministic rollouts to measure mean return.
3. Saves a model checkpoint named `ep{NNNN}_return{RRR}/model.zip`.
4. Saves `meta.json` alongside each checkpoint.
5. Appends a row to `training_log.csv` in the logs dir.
6. Maintains `best/` and `latest/` symlinks.

### New: `generate_sail_dataset_from_checkpoint.py`

Location: `sail-tf1b/stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py`

Takes a checkpoint path and generates a SAIL-compatible `.npz` file. Key behaviors:
- Saves the exact keys `obs`, `actions`, `rewards`, `episode_returns`, `episode_starts`
  required by `ExpertDataset`.
- `rewards` shape is `(N, 1)` to match existing datasets.
- `episode_starts` uses the same object-dtype format as existing files.
- Dataset filename includes the actual mean return of the collected rollouts.

---

## 5. Checkpoint Saving Every 100 Episodes

Implementation pattern:

```python
# Train in chunks
for chunk_start in range(0, total_episodes, checkpoint_every):
    chunk_end   = min(chunk_start + checkpoint_every, total_episodes)
    chunk_steps = (chunk_end - chunk_start) * EP_LEN         # 1000 steps/episode

    model.learn(chunk_steps, reset_num_timesteps=False)       # continues from prior state
    current_episode = chunk_end

    # Eval
    mean_return = evaluate(model, env_id, n_eval_episodes)

    # Save
    ckpt_dir = os.path.join(ckpt_root, f"ep{current_episode:04d}_return{mean_return:.0f}")
    model.save(os.path.join(ckpt_dir, "model"))               # → model.zip
    save_meta(ckpt_dir, current_episode, mean_return, ...)
    update_symlinks(ckpt_root, ckpt_dir, is_best)
    append_csv_log(log_dir, current_episode, mean_return)
```

`reset_num_timesteps=False` is critical: it tells stable-baselines to continue the global
timestep counter so the replay buffer and training state are preserved across chunks.

---

## 6. Evaluation at Checkpoint Time

- Run `n_eval_episodes` (default 10) **deterministic** rollouts using `model.predict(obs, deterministic=True)`.
- Report mean return as the checkpoint score.
- This score goes into the checkpoint directory name and `meta.json`.
- Also report min/max for diagnostics.

---

## 7. Checkpoint Naming Convention

```
teacher_checkpoints/halfcheetah/
  ep0100_return1243/
    model.zip
    meta.json
  ep0200_return2891/
    model.zip
    meta.json
  ...
  ep0500_return6012/
    model.zip
    meta.json
  best -> ep0500_return6012/           ← symlink to best so far
  latest -> ep0500_return6012/         ← symlink to most recent checkpoint
```

`meta.json` fields:
```json
{
  "episode":         500,
  "timestep":        500000,
  "mean_return":     6012.3,
  "min_return":      5940.1,
  "max_return":      6080.7,
  "seed":            0,
  "env_id":          "HalfCheetah-v2",
  "is_best":         true,
  "policy_layers":   [400, 300],
  "noise_sigma":     0.1,
  "checkpoint_dir":  "ep0500_return6012"
}
```

---

## 8. Dataset Generation from a Checkpoint

Given any saved checkpoint, run:

```bash
python generate_sail_dataset_from_checkpoint.py \
  --checkpoint_dir  teacher_checkpoints/halfcheetah/ep0300_return4820 \
  --env_id          HalfCheetah-v2 \
  --n_trajs         4 \
  --save_dir        teacher_dataset/
```

The script will:
1. Load the TD3 model from `checkpoint_dir/model.zip`.
2. Roll out `n_trajs` deterministic episodes.
3. Package in SAIL-compatible format.
4. Save as `expert_data_no_img_HalfCheetah_scores_{ACTUAL_RETURN}_episodes_{n_trajs}_ep{episode}.npz`.

### To use the generated dataset with SAIL unchanged

SAIL constructs the path `../../teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz`.
You have two options:

**Option A (symlink):** Replace or create a symlink:
```bash
ln -sfn /path/to/your/generated.npz \
  sail-tf1b/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
```

**Option B (name it directly):** Pass `--sail-compat-name` to the generation script, which
saves a copy with the exact SAIL-expected filename alongside the informative one.

---

## 9. Compatibility Checklist

| Aspect | Existing SAIL Dataset | New Generated Dataset |
|---|---|---|
| `obs` key | ✓ `obs` shape (N,18) float32 | ✓ same |
| `actions` key | ✓ `actions` shape (N,6) float32 | ✓ same |
| `rewards` key | ✓ `rewards` shape (N,1) float32 2D | ✓ same |
| `episode_returns` key | ✓ float64 (n_traj,) | ✓ same |
| `episode_starts` key | ✓ object dtype, True/array([False]) | ✓ replicated |
| Loaded by `ExpertDataset` | ✓ | ✓ |
| `demo_dones` derived from it | ✓ | ✓ |
| No `next_obs` key needed | ✓ ExpertDataset derives it | ✓ |
| Env wrapper | None (raw gym.make) | None (raw gym.make) |

**No environment wrappers are used for teacher training or data collection.**
The SAIL training script (`train_sail.py`) may apply `TimeFeatureWrapper` to the student
environment, but the teacher dataset is collected without any wrapper — matching the
existing files.

---

## 10. Output Directory Structure

```
sail-tf1b/
  teacher_training/
    halfcheetah/
      teacher_checkpoints/
        ep0100_return{R}/
          model.zip
          meta.json
        ep0200_return{R}/
          model.zip
          meta.json
        ...
        best   -> ep{N}_return{R}/   (symlink)
        latest -> ep{N}_return{R}/   (symlink)
      teacher_logs/
        training_log.csv
      teacher_eval/
        (optional: per-checkpoint eval rollout data)
  teacher_dataset/
    expert_data_no_img_HalfCheetah_scores_{R}_episodes_{N}_ep{E}.npz
    (symlinks to scores_5600 names if needed for SAIL)
```

---

## 11. Files Created

| File | Purpose |
|---|---|
| `scripts/train_halfcheetah_teacher_checkpoints.py` | Full teacher training run with checkpoint saving every 100 episodes |
| `scripts/generate_sail_dataset_from_checkpoint.py` | Dataset generation from any saved checkpoint |
| `run/teacher_train_hc_checkpoints.sbatch` | Slurm job for teacher training |
| `run/teacher_gen_dataset_hc.sbatch` | Slurm job for dataset generation |

---

## 12. Usage

### Launch teacher training

```bash
cd sail-tf1b/stable-baselines/run
sbatch teacher_train_hc_checkpoints.sbatch
```

Edit the sbatch `TOTAL_EPISODES`, `SEED`, and `SAVE_ROOT` as needed.

### Check progress

```bash
cat sail-tf1b/teacher_training/halfcheetah/teacher_logs/training_log.csv
```

### Generate a dataset from a specific checkpoint

```bash
cd sail-tf1b/stable-baselines/run
sbatch teacher_gen_dataset_hc.sbatch
```

Set `CHECKPOINT_DIR` and `N_TRAJS` in that sbatch.

Or directly:

```bash
python scripts/generate_sail_dataset_from_checkpoint.py \
  --checkpoint_dir /path/to/ep0500_return6012 \
  --env_id HalfCheetah-v2 \
  --n_trajs 4 \
  --save_dir ../../teacher_dataset/
```

### Use the generated dataset with SAIL

```bash
# Option A: symlink the generated file to the SAIL-expected path
ln -sfn /abs/path/to/generated.npz \
  sail-tf1b/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz

# Then run SAIL normally:
python run/train_sail.py --env HalfCheetah-v2 --algo sail \
  --task gail-lfd-adaptive-dynamic --n-episodes 4 ...
```
