# Meta-World Integration Plan
**Target repo:** `/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM`  
**Date drafted:** 2026-04-23  
**Status:** Planning only — nothing implemented yet

---

## Table of Contents

1. [High-Level Strategy](#1-high-level-strategy)
2. [Recommended Meta-World Scope](#2-recommended-meta-world-scope)
3. [Plan 1 — Teacher Training, Checkpointing, and Dataset Generation](#3-plan-1--teacher-training-checkpointing-and-dataset-generation)
4. [Plan 2 — Integrating Meta-World into sail_sb3_online](#4-plan-2--integrating-meta-world-into-sail_sb3_online)
5. [Non-Regression and Safety Requirements](#5-non-regression-and-safety-requirements)
6. [Suggested Directory and File Layout](#6-suggested-directory-and-file-layout)
7. [Validation Checklist](#7-validation-checklist)
8. [Recommended Order of Execution](#8-recommended-order-of-execution)

---

## 1. High-Level Strategy

### Core principle

All Meta-World code paths must be **strictly opt-in**. Every new file, wrapper, and
argument introduced for Meta-World must be gated behind an explicit flag or a clearly
named entry point. Existing MuJoCo scripts, datasets, and sbatch jobs must remain
runnable without any change.

### Why this is non-trivial

Meta-World is architecturally different from the MuJoCo gym environments already in
the repo in three important ways:

| Dimension | MuJoCo (existing) | Meta-World (new) |
|-----------|-------------------|-----------------|
| Registration | `gym.make("HalfCheetah-v2")` | No gym registry; manual construction via `metaworld.MT1` API |
| Success metric | Cumulative reward only | `info["success"]` boolean per step |
| Action space | 6–17 dim continuous | 4-dim (Δx, Δy, Δz, gripper) for all v2 tasks |
| Observation space | 17–27 dim | 39-dim for all v2 tasks |
| Task randomization | Single task per env | Requires `env.set_task(task)` before each episode |

Because `gym.make()` does not work for Meta-World, the existing
`make_env_with_time()` factory in `sail_sb3_online/scripts/train_sail.py` cannot be
called for Meta-World environments — a separate factory function is required.

### Safe integration path

```
Stage 0  Install & verify Meta-World in the cluster environment
    ↓
Stage 1  Build a completely standalone Meta-World teacher training pipeline
         (new scripts, new folder, no shared code with sail_sb3_online)
    ↓
Stage 2  Generate and validate a SAIL-compatible NPZ dataset
         (existing expert_loader.py and teacher_buffer.py must read it without change)
    ↓
Stage 3  Add Meta-World environment support to sail_sb3_online
         (isolated in a new mw_env_utils.py; detected at runtime via a new flag)
    ↓
Stage 4  Run adaptive SAIL on Meta-World using the validated dataset
    ↓
Stage 5  Extend to preference ranking and online RM variants
```

Each stage has a green-light gate in the [Validation Checklist](#7-validation-checklist).
Do not proceed to the next stage until the gate is cleared.

---

## 2. Recommended Meta-World Scope

### 2.1 MT1 vs MT10 vs MT50

**Start with MT1 (single-task).**

- MT10 and MT50 require multi-task policy architectures and per-task reward shaping.
  SAIL's current discriminator and TD3 backbone are designed for single-task continuous
  control. Using MT10/MT50 before the single-task pipeline is validated is premature.
- MT1 gives a standard single-task MDP with a fixed task goal sampled at episode start.
  This matches the structure of the existing MuJoCo experiments exactly.
- Switching to MT10 later requires only changing the env constructor — the dataset
  format, discriminator, and training loop do not need to change.

### 2.2 Recommended first tasks

| Task | Why start here |
|------|----------------|
| `reach-v2` | Lowest complexity; 3-DoF reaching. Expert ceiling is easy to achieve. Teacher converges fast (< 500 k steps with SAC). Useful as a smoke-test benchmark. |
| `push-v2` | Moderate complexity; object manipulation with a clear goal. Reward is dense. Teacher quality is easy to measure by success rate. |
| `pick-place-v2` | The canonical Meta-World benchmark task. Slightly harder than push. Good for showing SAIL can generalize. |

Start with `reach-v2` for the teacher pipeline smoke test, then `push-v2` for the
first real SAIL experiment.

### 2.3 Tasks to avoid initially

| Task | Reason to defer |
|------|----------------|
| `assembly-v2`, `disassemble-v2` | Very long horizon; sparse structured success. Expert training takes much longer. |
| `bin-picking-v2` | Requires precise depth perception; high stochasticity. |
| `box-close-v2` | Requires long-horizon planning; teacher training time is > 2 M steps. |
| Any `MT10`/`MT50` task | Multi-task setup; incompatible with single-task SAIL architecture at this stage. |

### 2.4 Why these tasks fit SAIL

SAIL learns by matching the teacher's state-action distribution via adversarial imitation.
For this to work well:

1. The teacher policy must be reliably good (high success rate, not just high reward).
2. The teacher dataset must cover the episode start distribution.
3. The discriminator must be able to distinguish teacher from student behavior.

`reach-v2`, `push-v2`, and `pick-place-v2` all have dense rewards, short-to-medium
episode horizons (200 steps), and well-behaved observation spaces — exactly the regime
where SAIL has succeeded on MuJoCo. The success metric in `info["success"]` also
gives a clean teacher quality signal that does not require score normalization.

---

## 3. Plan 1 — Teacher Training, Checkpointing, and Dataset Generation

### Overview

This plan is entirely self-contained. Nothing from `sail_sb3` or `sail_sb3_online` is
modified. All new code lives under a new top-level directory `metaworld/`.

---

### Phase 1.0 — Environment setup and installation

**Step 1.0.1 — Install Meta-World in the cluster environment**

- Meta-World requires a compatible MuJoCo version.  The repo already uses `mujoco-py`
  with MuJoCo 2.x, which Meta-World v2 is compatible with.
- Install via: `pip install metaworld` (or from the `rlworkgroup/metaworld` GitHub
  source if pinning a specific commit is needed for reproducibility).
- Verify: open a Python shell and run
  ```python
  import metaworld
  ml1 = metaworld.MT1("reach-v2", seed=0)
  env = ml1.train_classes["reach-v2"]()
  task = ml1.train_tasks[0]
  env.set_task(task)
  obs, _ = env.reset()
  print(obs.shape)   # expect (39,)
  ```
- Pin the exact `metaworld` version in a new
  `metaworld/requirements.txt` (separate from `sail_sb3/requirements.txt`).

**Step 1.0.2 — Verify environment interface**

Confirm the following about the chosen tasks before writing any training code:

- `obs.shape` is `(39,)` for all three starter tasks.
- `action_space.shape` is `(4,)` for all three starter tasks.
- `info["success"]` is present after every `env.step()`.
- Episode length is 200 steps (Meta-World default for v2 tasks).
- `env.reset()` returns `(obs, info)` — note Meta-World follows the **new** Gym API
  (`reset()` returns a tuple) while the repo currently uses Gym 0.21.0 which returns
  only `obs`. This **must** be handled in the wrapper.

---

### Phase 1.1 — Teacher training script

**Step 1.1.1 — Create `metaworld/teacher_training/train_mw_teacher.py`**

This is a new, standalone training script. It must not import anything from
`sail_sb3/` or `sail_sb3_online/`.

What this script must do:

1. Accept command-line arguments:
   - `--task` (e.g., `reach-v2`)
   - `--seed` (integer)
   - `--total_timesteps` (e.g., `1_000_000`)
   - `--checkpoint_freq` (save every N steps, e.g., `50_000`)
   - `--checkpoint_dir` (output root, e.g., `teacher_training/metaworld/reach-v2/`)
   - `--algo` (default `sac`; `td3` as optional alternative)
   - `--eval_freq` (how often to compute success rate, e.g., `10_000`)
   - `--n_eval_episodes` (e.g., `20`)
   - `--wandb` (flag to enable W&B logging)
   - `--run_name` (for W&B and output directory naming)

2. Construct the Meta-World environment using the MT1 API (see Phase 1.2 for the
   wrapper).

3. Train a policy using SB3's `SAC` (preferred for Meta-World) or `TD3`.

   - Use default SB3 SAC hyperparameters as the starting point.
   - Set `learning_starts = 1000` to avoid random-data warmup artifacts.
   - Use a replay buffer size of 1_000_000.

4. At every `checkpoint_freq` steps:
   - Call the checkpoint saving function (Phase 1.3).
   - Run `n_eval_episodes` evaluation rollouts.
   - Compute and log: mean episodic return, success rate (fraction of episodes where
     `any(info["success"])` is True during the episode).

5. Log to W&B if `--wandb` is set, under a project named `mw-teacher-training`.

6. Save a `training_summary.json` at the end of training with the best checkpoint
   path, best success rate, and total wall time.

**Step 1.1.2 — Do NOT use `TimeFeatureWrapper` in teacher training**

`TimeFeatureWrapper` is a SAIL-specific wrapper that appends a time feature
`[1.0 → 0.0]` to the observation. The teacher policy should be trained with the
raw 39-dim observation. The time feature will be added later when the teacher
dataset is collected for SAIL.

This matches the existing pattern: in the MuJoCo pipeline, the teacher is trained
on the raw observation space, and `TimeFeatureWrapper` is applied at dataset
collection time.

---

### Phase 1.2 — Meta-World environment wrapper

**Step 1.2.1 — Create `metaworld/utils/mw_env_utils.py`**

This file must contain:

1. `MetaWorldWrapper(gym.Wrapper)` — a thin wrapper that:
   - Adapts the Meta-World `reset()` return format (tuple `(obs, info)`) to the old
     Gym 0.21.0 format (returns only `obs`) so SB3 1.8.0 does not break.
   - Re-samples a random task from the task pool at the start of each episode
     (calls `env.set_task(random.choice(tasks))` in `reset()`).
   - Passes `info["success"]` through in the `info` dict from `step()`.
   - **Does not** modify observations or actions.

2. `make_mw_teacher_env(task_name, seed)` — factory function for teacher training:
   ```
   metaworld.MT1(task_name, seed=seed)
   → MetaWorldWrapper (task re-sampling + API adaptation)
   → gym.wrappers.TimeLimit(max_episode_steps=200) if not already set
   → Monitor (SB3 episode logging)
   ```
   Returns a callable (for use with `DummyVecEnv` or `SubprocVecEnv`).

3. `make_mw_sail_env(task_name, seed)` — factory function for SAIL integration
   (Phase 4). Same as above, but also applies `TimeFeatureWrapper` at the end.
   This function is defined here but **not used until Phase 4**.

4. `MW_TASKS` — a list of the supported/tested task names.

5. `MW_EXPERT_RETURNS` — a dictionary mapping task names to reference expert
   success rates and reference return values collected after teacher training.
   This will be populated incrementally as teachers are trained and evaluated.

**Step 1.2.2 — Keep `metaworld/utils/mw_env_utils.py` fully standalone**

This file must not import from `sail_sb3_online/utils/env_utils.py` or any existing
SAIL utility. It is a new file in a new location.

---

### Phase 1.3 — Checkpoint saving

**Step 1.3.1 — Checkpoint directory structure**

Each checkpoint is saved as a subdirectory under the experiment output root:

```
teacher_training/metaworld/{task}/{run_name}/
    checkpoints/
        step_{N}/
            model.zip             ← SB3 model.save() output
            meta.json             ← checkpoint metadata (see below)
        step_{M}/
            ...
    training_summary.json         ← best checkpoint path, final success rate
    train.log                     ← console output mirror
```

**Step 1.3.2 — `meta.json` format per checkpoint**

```json
{
  "task":             "reach-v2",
  "algo":             "sac",
  "step":             100000,
  "seed":             0,
  "mean_return":      450.3,
  "success_rate":     0.95,
  "n_eval_episodes":  20,
  "wall_time_hours":  0.42,
  "model_file":       "model.zip",
  "git_hash":         "abc123"
}
```

Include `success_rate` in every checkpoint's metadata. This is critical: for
Meta-World tasks, **success rate is the primary quality metric**, not mean return.
A policy with return 450 but success rate 0.4 is not a good teacher for SAIL.

**Step 1.3.3 — Checkpoint saving function**

Create `metaworld/teacher_training/checkpoint_utils.py` with:

- `save_checkpoint(model, step, meta_dict, output_dir)` — saves `model.zip` and
  `meta.json` under `{output_dir}/checkpoints/step_{step}/`.
- `load_best_checkpoint(output_dir, metric="success_rate")` — scans all `meta.json`
  files in `{output_dir}/checkpoints/`, returns the path to the best checkpoint
  by the given metric.
- `list_checkpoints(output_dir)` — returns a sorted list of
  `(step, meta_dict, model_path)` tuples.

---

### Phase 1.4 — Checkpoint selection

**Step 1.4.1 — Selection criteria**

After training completes, select a checkpoint for dataset generation using the
following criteria (in order of priority):

1. **Success rate ≥ 0.90** in `n_eval_episodes = 20` rollouts. This is the minimum
   teacher quality threshold for SAIL to have anything meaningful to imitate.
2. Among checkpoints meeting criterion 1, choose the one with the **highest mean
   episodic return** as a tiebreaker.
3. If no checkpoint reaches 0.90 success rate, train longer or tune SAC
   hyperparameters before generating the dataset.

**Step 1.4.2 — Selection script**

Create `metaworld/teacher_training/select_checkpoint.py`:
- Reads all `meta.json` files in a given run directory.
- Prints a sorted table of checkpoints by success rate and return.
- Accepts `--output_dir` and `--metric` arguments.
- Prints the recommended checkpoint path to stdout so it can be piped into the
  dataset generation script.

---

### Phase 1.5 — Dataset generation

**Step 1.5.1 — Create `metaworld/dataset_generation/generate_mw_dataset.py`**

This is a standalone script that takes a trained teacher checkpoint and produces
a SAIL-compatible NPZ file. It must not modify any file in `sail_sb3_online/`.

Arguments:
- `--task` (e.g., `reach-v2`)
- `--checkpoint_path` (path to `model.zip`)
- `--n_episodes` (number of expert episodes to collect, e.g., `20`)
- `--output_path` (path to output `.npz` file)
- `--seed`
- `--add_time_feature` (flag; if set, appends time feature to each observation
  before saving — required for SAIL compatibility)

What the script must do:

1. Load the SB3 model from `checkpoint_path`.
2. Construct the environment using `make_mw_teacher_env(task, seed)` from
   `metaworld/utils/mw_env_utils.py`.
3. If `--add_time_feature` is set, wrap the env in `TimeFeatureWrapper` (the same
   class used in `sail_sb3_online/scripts/train_sail.py`) so that observations
   are 40-dim (39 + time feature).
4. Roll out the teacher for `n_episodes` episodes. At each step, collect:
   - `obs` (before step, **after** time feature if enabled)
   - `action` (from teacher policy, deterministic)
   - `reward` (from environment)
   - `done` (episode termination)
   - `next_obs` (after step)
   - `success` (from `info["success"]`)
5. After collection, compute:
   - Mean episode return across all collected episodes.
   - Success rate across all collected episodes.
   - Print a summary to stdout.
6. Save as a compressed NPZ file with the following keys (matching the existing
   format expected by `sail_sb3_online/datasets/expert_loader.py`):
   ```
   obs                → [N_transitions, obs_dim] float32
   actions            → [N_transitions, act_dim] float32
   rewards            → [N_transitions] float32
   episode_starts     → [N_transitions] bool
   next_observations  → [N_transitions, obs_dim] float32
   dones              → [N_transitions] bool
   ```
   Additionally save a separate `dataset_meta.json` alongside the NPZ with:
   ```json
   {
     "task": "reach-v2",
     "n_episodes": 20,
     "n_transitions": 4000,
     "obs_dim": 40,
     "act_dim": 4,
     "mean_return": 463.1,
     "success_rate": 0.95,
     "checkpoint_path": "...",
     "time_feature_added": true,
     "seed": 0
   }
   ```

**Step 1.5.2 — Why `episode_starts` instead of `dones` inversion**

The existing `expert_loader.py` computes `dones` from `episode_starts` via
bitwise inversion with a shift. Follow the same convention: set
`episode_starts[0] = True`, `episode_starts[t] = True` at the first step of each
new episode. This is identical to the existing MuJoCo dataset format.

**Step 1.5.3 — Deterministic vs stochastic teacher rollouts**

Use **deterministic** rollouts for dataset generation:
- For SAC: set `deterministic=True` in `model.predict()`.
- For TD3: always deterministic.

This matches existing practice in the MuJoCo teacher pipeline and produces cleaner
imitation targets.

---

### Phase 1.6 — Dataset validation and smoke tests

Before using the dataset for SAIL, run the following checks:

**Step 1.6.1 — Format check**

Write a small validation script
`metaworld/dataset_generation/validate_mw_dataset.py` that:

1. Loads the NPZ with `load_expert_npz()` from the existing
   `sail_sb3_online/datasets/expert_loader.py`.  
   **If this import fails or throws, the dataset format is wrong — fix it before
   proceeding.**
2. Checks that all required keys (`obs`, `actions`, `rewards`, `episode_starts`,
   `next_observations`, `dones`) are present.
3. Checks that `obs.shape[1]` matches the expected obs_dim (40 if time feature
   was added, 39 otherwise).
4. Checks that `actions.shape[1]` is 4.
5. Reconstructs episode boundaries from `episode_starts` and verifies the number
   of episodes matches the intended `n_episodes`.
6. Computes and prints mean per-episode return.

**Step 1.6.2 — TeacherBuffer load check**

Open a Python shell and manually construct a `TeacherBuffer` from the new NPZ:
```python
from sail_sb3_online.datasets.teacher_buffer import TeacherBuffer
from sail_sb3_online.datasets.expert_loader import load_expert_npz
data = load_expert_npz("path/to/mw_reach_v2_dataset.npz")
buf = TeacherBuffer(data, device="cpu")
batch = buf.sample(256)
print(batch.observations.shape)   # expect (256, 40)
print(batch.actions.shape)        # expect (256, 4)
```
If this succeeds without any shape or dtype errors, the dataset is compatible with
the existing training pipeline.

**Step 1.6.3 — Visualization check**

Plot a few trajectories from the dataset and visually confirm that:
- Observations are in a reasonable range (Meta-World obs are not normalized;
  values are in roughly [-1, 2] meters).
- Actions are in `[-1, 1]` (Meta-World clips to this range).
- Episode lengths are 200 or fewer timesteps (no run-on episodes).

---

## 4. Plan 2 — Integrating Meta-World into sail_sb3_online

### Overview

This plan begins only after Plan 1 is complete and the NPZ dataset has passed all
validation checks. The goal is to run adaptive SAIL on Meta-World inside
`sail_sb3_online` while keeping all existing MuJoCo paths untouched.

---

### Phase 2.1 — Environment factory isolation

**Step 2.1.1 — Create `sail_sb3_online/utils/mw_env_utils.py`**

Copy `metaworld/utils/mw_env_utils.py` (from Plan 1) into `sail_sb3_online/utils/`.
The `sail_sb3_online` copy is the one that will be imported during training. The
version in `metaworld/` remains the standalone reference.

This file exports:
- `make_mw_sail_env(task_name, seed)` — used at training time (with TimeFeatureWrapper).
- `is_metaworld_task(env_id: str) -> bool` — returns `True` if the env_id string
  starts with `"mw:"` (the naming convention for Meta-World tasks in this repo).
- `MW_EXPERT_RETURNS` — dict for success-rate normalization logging.

**Step 2.1.2 — Naming convention for Meta-World task IDs**

Use a `mw:` prefix to distinguish Meta-World tasks from MuJoCo gym IDs:
- `mw:reach-v2`
- `mw:push-v2`
- `mw:pick-place-v2`

The `--env` argument in `train_sail.py` will pass strings like `mw:reach-v2`.
The existing MuJoCo paths pass strings like `HalfCheetah-v2`. These two formats are
mutually exclusive and easily distinguishable.

**Step 2.1.3 — Do NOT modify `make_env_with_time()` in `train_sail.py`**

The existing function at `sail_sb3_online/scripts/train_sail.py:make_env_with_time()`
must remain unchanged. Instead, add a new top-level factory dispatcher — a single
function `make_env(env_id, seed)` in a new location (see Step 2.1.4) that checks the
prefix and routes to the correct factory.

**Step 2.1.4 — Create `sail_sb3_online/utils/env_factory.py`**

New file — not a modification of the existing `env_utils.py` stub. Contents:

```
is_metaworld_task(env_id)  →  bool
make_env(env_id, seed)     →  callable
    if is_metaworld_task(env_id):
        task_name = env_id.split(":", 1)[1]   # "mw:reach-v2" → "reach-v2"
        return make_mw_sail_env(task_name, seed)
    else:
        return make_env_with_time(env_id, seed)   # existing MuJoCo factory
```

`train_sail.py` will import `make_env` from `env_factory.py` and replace the direct
call to `make_env_with_time()`. This is the only required change to the existing
training script, and it is backward-compatible: if the env_id does not start with
`mw:`, behaviour is identical to before.

---

### Phase 2.2 — Observation and action space compatibility

**Step 2.2.1 — Observation dimension handling**

The existing `EXPERT_RETURNS` dict in `train_sail.py` uses environment name as key
for success normalization. Add Meta-World entries using the `mw:` prefix convention:

```python
EXPERT_RETURNS = {
    # Existing MuJoCo entries — DO NOT CHANGE
    "HalfCheetah-v2": 9094.0,
    "Walker2d-v2":    4717.0,
    "Hopper-v2":      3606.0,
    "Ant-v2":         5813.0,
    "Swimmer-v2":     359.0,
    # New Meta-World entries — add after teacher training
    "mw:reach-v2":       None,   # populate after teacher training
    "mw:push-v2":        None,
    "mw:pick-place-v2":  None,
}
```

For Meta-World experiments, log **success rate** as the primary metric, not
normalized score. See Phase 2.4 for the logging changes.

**Step 2.2.2 — TimeFeatureWrapper compatibility**

`TimeFeatureWrapper` appends a `[1.0 → 0.0]` linear time feature to the observation.
It is a generic Gym wrapper and does not contain any MuJoCo-specific logic.
It will work with Meta-World's 39-dim observations, producing 40-dim observations.

No changes to `TimeFeatureWrapper` are needed.

**Step 2.2.3 — Discriminator input dimension**

The discriminator (`Adversary`) takes `concat(obs, action)` as input. Its input
dimension is `obs_dim + act_dim`. For Meta-World with time feature:
`40 + 4 = 44`. For HalfCheetah: `18 + 6 = 24`. The discriminator is instantiated
at training time using the actual `env.observation_space.shape[0]` and
`env.action_space.shape[0]`, so no change is needed — it will automatically adapt.

---

### Phase 2.3 — Adaptive SAIL for Meta-World (first experiment)

**Step 2.3.1 — What to do first**

Run the simplest possible configuration first:
- **Adaptive SAIL only** (`--adaptive` flag).
- No preference ranking, no QPREF, no online RM.
- Use `reach-v2` as the first task.
- Use the NPZ dataset generated in Plan 1.

Command structure (informational only — do not run until all gates are cleared):
```
python sail_sb3_online/scripts/train_sail.py \
    --env mw:reach-v2 \
    --expert_data teacher_dataset/metaworld/reach-v2/expert_reach_v2_seed0.npz \
    --adaptive \
    --total_timesteps 500000 \
    --seed 0 \
    --wandb
```

**Step 2.3.2 — Sbatch script**

Create `sail_sb3_online/sbatch/mw_adaptive_reach.sbatch`. This is a new file.
Do not modify any existing sbatch scripts. The new sbatch file should:
- Request appropriate GPU/CPU resources.
- Activate the correct conda environment.
- Run the command from Step 2.3.1.
- Set `--output` and `--error` to `sail_logs/metaworld/`.

**Step 2.3.3 — What to watch during the first run**

- The discriminator loss should decrease and stabilize within the first 50 k steps.
- The student episode return should gradually increase.
- The adaptive promotion rate (fraction of student episodes promoted to the teacher
  buffer) should be non-zero but not 100%.
- If the success rate never increases above 10%, the teacher dataset quality may
  be insufficient — go back and check the checkpoint used for dataset generation.

---

### Phase 2.4 — Meta-World-specific evaluation metrics

**Step 2.4.1 — Success rate logging**

Meta-World's primary metric is **success rate** (fraction of episodes where the
task is completed), not normalized score. The existing W&B logging in `train_sail.py`
logs `episode/normalized_score` using `EXPERT_RETURNS[env_id]`.

Add a parallel logging path for Meta-World tasks:
- In the episode completion callback (or at the end of each rollout), check if
  `env_id` starts with `mw:`.
- If yes, additionally log `episode/success_rate` using the `info["success"]`
  signal aggregated over the episode.

This is a **new `if is_metaworld_task(env_id):` branch** added to the eval/logging
code. It does not touch the existing MuJoCo logging path.

**Step 2.4.2 — What NOT to change**

Do not rename, remove, or alter the existing `episode/normalized_score`,
`episode/return`, or `disc/loss` logging keys. Those are used by existing W&B
dashboards and sbatch jobs.

---

### Phase 2.5 — What to postpone

The following features should **not** be added during the initial Meta-World
adaptive SAIL run. They are deferred until `reach-v2` adaptive SAIL is working:

| Deferred Feature | When to add |
|-----------------|-------------|
| Preference ranking (`--pref_rank_disc`) | After adaptive SAIL success rate > 0.50 on `reach-v2` |
| QPREF (`--qpref`) | After preference ranking baseline is validated |
| Online RM (`OnlineRMManager`) | After offline preference ranking works |
| `push-v2` and `pick-place-v2` | After `reach-v2` pipeline is validated end-to-end |
| MT10 / multi-task | After single-task results are published or satisfactory |

---

### Phase 2.6 — Extending to preference ranking

Once adaptive SAIL is working for `reach-v2`, extend to preference ranking:

**Step 2.6.1 — Offline preference RM for Meta-World**

The existing preference RM training pipeline (in `pebble_runner/` or the offline
RM scripts) generates a `.pt` or `.npz` RM file from teacher episodes.
Run this pipeline on the Meta-World teacher dataset to get an offline RM.

This step requires no new code — the teacher buffer's `sample_pref_pairs()` method
works on any NPZ dataset, regardless of env.

**Step 2.6.2 — Preference ranking run**

Add `--pref_rank_disc --pref_rm path/to/mw_rm.pt --pref_rank_weight 0.1` to the
existing Meta-World sbatch. No new training code is needed.

---

### Phase 2.7 — Extending to online RM

Once offline preference ranking works, extend to `sail_sb3_online`'s online RM:

**Step 2.7.1 — SegmentStore population**

`SegmentStore` populates from both teacher and student segments via
`add_student_episode()` and `add_segment()`. For Meta-World, segments are 50-step
windows (same as MuJoCo default). No change to `SegmentStore` is needed.

**Step 2.7.2 — RM activation gates**

The three gates in `OnlineRMManager` (min_segments, min_rm_updates, min_rm_accuracy)
apply identically to Meta-World. No change needed.

**Step 2.7.3 — Success rate as RM quality signal**

When evaluating online RM quality on Meta-World, consider logging RM-predicted score
vs. true success rate as a diagnostic. Add this as a new W&B logging key under
`rm/success_rate_corr` — gated behind `is_metaworld_task(env_id)`.

---

### Phase 2.8 — Failure points to watch for

| Failure mode | Likely cause | How to diagnose |
|---|---|---|
| Discriminator loss explodes at step 0 | Observation range mismatch (teacher dataset obs_dim wrong) | Check `obs.shape` in dataset vs env |
| Student never improves after 200k steps | Teacher success rate < 0.90 in dataset | Check `dataset_meta.json` |
| `TimeFeatureWrapper` shape error | Meta-World reset API returning tuple; wrapper sees wrong obs | Check `MetaWorldWrapper` adaptation |
| SB3 `DummyVecEnv` error on reset | Meta-World `reset()` returns `(obs, info)` tuple | Ensure `MetaWorldWrapper.reset()` returns only `obs` |
| Adaptive promotion never triggers | Episode return threshold too high (MuJoCo threshold applied to Meta-World) | Verify threshold is computed from Meta-World teacher returns |
| `EXPERT_RETURNS[env_id]` KeyError | New env_id not added to the dict | Add `mw:task-v2` entry |

---

## 5. Non-Regression and Safety Requirements

### 5.1 Existing MuJoCo experiments must keep working

- **No modification** to `sail_sb3_online/scripts/train_sail.py` beyond:
  1. Importing `make_env` from the new `env_factory.py` instead of calling
     `make_env_with_time()` directly.
  2. Adding `mw:` entries to `EXPERT_RETURNS` (non-breaking; existing keys untouched).
  3. One new `if is_metaworld_task(env_id):` branch in the eval logging section.
- The replacement of `make_env_with_time()` with `make_env()` in `train_sail.py` is
  purely a routing dispatch. If `env_id` does not start with `mw:`, it calls
  `make_env_with_time()` exactly as before.

### 5.2 Existing sbatch scripts must not be broken

- None of the existing `*.sbatch` files will be changed.
- All existing sbatch scripts pass MuJoCo env names (e.g., `HalfCheetah-v2`) to
  `--env`. These do not start with `mw:` and will never be routed to the Meta-World
  factory.

### 5.3 Existing dataset loading must not be broken

- `expert_loader.py` and `teacher_buffer.py` must not be modified.
- The Meta-World NPZ dataset is designed to be readable by these existing classes
  (validated in Phase 1.6).

### 5.4 Meta-World code paths are opt-in only

- Meta-World is activated only when `--env` starts with `mw:`.
- The `metaworld` Python package is imported only inside `mw_env_utils.py` and
  only at function call time (lazy import, not at module top level). If `metaworld`
  is not installed, importing `env_factory.py` or `train_sail.py` will still succeed
  as long as no `mw:` env is requested.

### 5.5 Changes must be minimal and localized

| Change | Where | Impact on existing code |
|--------|--------|------------------------|
| New `make_env()` dispatcher | `sail_sb3_online/utils/env_factory.py` (new file) | None |
| New `mw_env_utils.py` | `sail_sb3_online/utils/` (new file) | None |
| Import `make_env` from `env_factory` | `sail_sb3_online/scripts/train_sail.py` (1 line change) | Fully backward-compatible |
| Add `mw:` keys to `EXPERT_RETURNS` | `sail_sb3_online/scripts/train_sail.py` | Non-breaking dict append |
| New success rate logging branch | `sail_sb3_online/scripts/train_sail.py` | Gated behind `is_metaworld_task()` |

Total changes to existing files: **≤ 5 lines in `train_sail.py`**.
Everything else is new files.

---

## 6. Suggested Directory and File Layout

```
sail-tf1b-onlineRM/
│
├── metaworld/                              ← NEW: all Meta-World standalone code
│   ├── requirements.txt                   ← pinned metaworld version
│   ├── utils/
│   │   └── mw_env_utils.py                ← MetaWorldWrapper, make_mw_teacher_env,
│   │                                          make_mw_sail_env, MW_TASKS,
│   │                                          MW_EXPERT_RETURNS, is_metaworld_task
│   ├── teacher_training/
│   │   ├── train_mw_teacher.py            ← standalone teacher training script
│   │   ├── checkpoint_utils.py            ← save/load/list checkpoints
│   │   └── select_checkpoint.py           ← checkpoint selection helper
│   └── dataset_generation/
│       ├── generate_mw_dataset.py         ← NPZ dataset collection from checkpoint
│       └── validate_mw_dataset.py         ← format + TeacherBuffer compatibility check
│
├── teacher_training/
│   ├── HalfCheetah/                       ← existing (DO NOT TOUCH)
│   ├── Ant/                               ← existing (DO NOT TOUCH)
│   └── metaworld/                         ← NEW: output directory for MW teachers
│       ├── reach-v2/
│       │   └── sac_seed0/
│       │       ├── checkpoints/
│       │       │   ├── step_50000/
│       │       │   │   ├── model.zip
│       │       │   │   └── meta.json
│       │       │   └── step_100000/
│       │       │       └── ...
│       │       └── training_summary.json
│       └── push-v2/
│           └── ...
│
├── teacher_dataset/
│   ├── expert_data_no_img_HalfCheetah_*.npz   ← existing (DO NOT TOUCH)
│   └── metaworld/                             ← NEW: MW NPZ datasets
│       ├── reach-v2/
│       │   ├── expert_reach_v2_seed0.npz
│       │   └── dataset_meta.json
│       └── push-v2/
│           └── ...
│
├── sail_sb3_online/
│   ├── scripts/
│   │   └── train_sail.py                  ← MINIMAL CHANGE: import make_env,
│   │                                          add mw: to EXPERT_RETURNS,
│   │                                          add success_rate logging branch
│   ├── utils/
│   │   ├── env_utils.py                   ← existing (DO NOT TOUCH)
│   │   ├── env_factory.py                 ← NEW: make_env() dispatcher
│   │   └── mw_env_utils.py                ← NEW: copied from metaworld/utils/
│   └── sbatch/
│       ├── [existing sbatch files]        ← DO NOT TOUCH
│       └── mw_adaptive_reach.sbatch       ← NEW
│
├── sail_logs/
│   ├── [existing MuJoCo logs]             ← DO NOT TOUCH
│   └── metaworld/                         ← NEW: Meta-World experiment logs
│       └── reach-v2/
│
└── analysis/
    ├── [existing analysis docs]           ← DO NOT TOUCH
    └── metaworld_integration_plan.md      ← this file
```

---

## 7. Validation Checklist

### Gate A: Teacher training → Checkpointing

- [ ] `metaworld` package installs cleanly in the cluster conda environment
- [ ] `MetaWorldWrapper.reset()` returns a single numpy array (not a tuple)
- [ ] `MetaWorldWrapper.step()` includes `info["success"]` in returned info dict
- [ ] SAC training starts without error and episodic return increases over 50k steps
- [ ] `checkpoint_utils.save_checkpoint()` writes `model.zip` and `meta.json`
- [ ] `checkpoint_utils.load_best_checkpoint()` returns the correct path
- [ ] Eval rollouts compute `success_rate` correctly (not always 0 or always 1)
- [ ] At least one checkpoint reaches `success_rate ≥ 0.90` before proceeding

### Gate B: Checkpointing → Dataset generation

- [ ] `select_checkpoint.py` prints a sorted table with correct success rates
- [ ] The selected checkpoint has `success_rate ≥ 0.90` in `meta.json`
- [ ] `generate_mw_dataset.py` completes without error for `n_episodes = 5` (smoke test)
- [ ] Output NPZ file exists and is non-empty
- [ ] `dataset_meta.json` shows `success_rate ≥ 0.90` from the collection rollouts
- [ ] `add_time_feature = True` was set; `obs_dim = 40` in `dataset_meta.json`

### Gate C: Dataset generation → Adaptive SAIL

- [ ] `validate_mw_dataset.py` passes all format checks without error
- [ ] `load_expert_npz()` from the existing `expert_loader.py` reads the file cleanly
- [ ] Manual `TeacherBuffer(data, device="cpu")` construction succeeds
- [ ] `batch.observations.shape == (256, 40)` and `batch.actions.shape == (256, 4)`
- [ ] `env_factory.py` routes `mw:reach-v2` to `make_mw_sail_env()` correctly
- [ ] `env_factory.py` routes `HalfCheetah-v2` to `make_env_with_time()` correctly
  (existing path not broken)
- [ ] `train_sail.py` runs for 1000 steps on `mw:reach-v2` without crashing
- [ ] Discriminator loss is finite (not NaN) after 1000 steps
- [ ] Existing HalfCheetah sbatch script still runs without error after the changes
  to `train_sail.py` (run a 1000-step smoke test)

### Gate D: Adaptive SAIL → Preference ranking

- [ ] Student success rate on `reach-v2` exceeds 0.50 within 500k steps
- [ ] Adaptive promotion is occurring (non-zero fraction of episodes promoted)
- [ ] W&B logs show `episode/success_rate` for Meta-World and `episode/normalized_score`
  for MuJoCo (both present in respective runs, neither overwriting the other)
- [ ] An offline preference RM has been trained on the Meta-World teacher dataset
- [ ] The RM file (`.pt` or `.npz`) loads without error via `pref_rm_eval.py`
- [ ] A `--pref_rank_disc` run on `reach-v2` starts without error (1000-step smoke test)

---

## 8. Recommended Order of Execution

The following is the exact recommended sequence, from first test to first real SAIL run.

```
Step 1  Install metaworld
        pip install metaworld
        Verify import + obs shape (Phase 1.0.1 and 1.0.2 above)

Step 2  Implement MetaWorldWrapper and mw_env_utils.py
        File: metaworld/utils/mw_env_utils.py
        Test: manually call make_mw_teacher_env("reach-v2", seed=0), reset, step

Step 3  Implement train_mw_teacher.py and checkpoint_utils.py
        Files: metaworld/teacher_training/train_mw_teacher.py
               metaworld/teacher_training/checkpoint_utils.py

Step 4  Run a short teacher training smoke test (5000 steps, no sbatch)
        python metaworld/teacher_training/train_mw_teacher.py \
            --task reach-v2 --seed 0 --total_timesteps 5000 \
            --checkpoint_freq 2500 --checkpoint_dir teacher_training/metaworld/reach-v2/smoke/
        Verify: checkpoints saved, meta.json written, no crash

Step 5  Submit full teacher training sbatch for reach-v2
        Target: success_rate ≥ 0.90
        Monitor with W&B

Step 6  Run select_checkpoint.py to identify best checkpoint
        Verify success_rate ≥ 0.90 before proceeding

Step 7  Implement generate_mw_dataset.py and validate_mw_dataset.py
        Files: metaworld/dataset_generation/generate_mw_dataset.py
               metaworld/dataset_generation/validate_mw_dataset.py

Step 8  Generate dataset for reach-v2 (Gate B smoke test: 5 episodes)
        python metaworld/dataset_generation/generate_mw_dataset.py \
            --task reach-v2 --checkpoint_path ... \
            --n_episodes 5 --output_path teacher_dataset/metaworld/reach-v2/smoke.npz \
            --add_time_feature
        Verify: obs_dim=40, act_dim=4, success_rate≥0.90

Step 9  Generate full dataset (20 episodes)
        Same command, --n_episodes 20, output to final path

Step 10 Run validate_mw_dataset.py on the full dataset
        Verify all Gate C format checks pass

Step 11 Implement env_factory.py and mw_env_utils.py in sail_sb3_online/utils/
        Files: sail_sb3_online/utils/env_factory.py
               sail_sb3_online/utils/mw_env_utils.py

Step 12 Apply minimal changes to sail_sb3_online/scripts/train_sail.py
        (import make_env, add mw: keys to EXPERT_RETURNS, add success_rate branch)

Step 13 Run 1000-step smoke test on MuJoCo (non-regression check)
        python sail_sb3_online/scripts/train_sail.py \
            --env HalfCheetah-v2 --expert_data teacher_dataset/expert_data_no_img_HalfCheetah_*.npz \
            --total_timesteps 1000
        Must complete without error

Step 14 Run 1000-step smoke test on Meta-World (Gate C)
        python sail_sb3_online/scripts/train_sail.py \
            --env mw:reach-v2 \
            --expert_data teacher_dataset/metaworld/reach-v2/expert_reach_v2_seed0.npz \
            --adaptive --total_timesteps 1000 --seed 0
        Must complete without error, finite discriminator loss

Step 15 Submit full adaptive SAIL run on reach-v2
        sbatch sail_sb3_online/sbatch/mw_adaptive_reach.sbatch
        Monitor W&B: watch for success_rate, disc_loss, adaptive promotion

Step 16 (only after Gate D is cleared) Implement and run preference ranking on reach-v2
        Train offline RM on reach-v2 teacher dataset
        Add --pref_rank_disc --pref_rm flags to the reach-v2 sbatch
        Run smoke test (1000 steps), then full run

Step 17 Extend to push-v2
        Repeat Steps 5–16 with --task push-v2
        Everything else (wrappers, factory, train_sail.py) is already in place
```

---

*End of plan. Nothing has been implemented. This document is a planning artifact only.*
