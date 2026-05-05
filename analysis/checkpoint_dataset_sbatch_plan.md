# Checkpoint Dataset + Sbatch Plan

**Date:** 2026-04-19  
**Scope:** Generate 7 teacher datasets from sub-optimal checkpoints across 4 environments;
create matching sail_sb3 and sail_sb3_online sbatch files; smoke-validate one pair.

---

## 1. Dataset Generation

**Script:** `stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py`  
**Python:** `/home/sohams/miniconda3/envs/sail_sb3_env/bin/python`  
**Arguments:** `--checkpoint_dir`, `--env_id`, `--n_trajs`, `--save_dir`

All 7 datasets will be generated with the same script. `n_trajs` is always an argument,
never hardcoded. `_verify()` runs automatically after save and checks all SAIL keys.

---

## 2. Dataset Save Locations

Convention follows the existing `teacher_dataset_HC/` pattern (flat dirs at repo root):

| Env | Checkpoint | n_trajs | Save dir |
|---|---|---|---|
| Ant-v2 | ep0200_return2858 | 4 | `teacher_dataset_Ant/` |
| Ant-v2 | ep1000_return3653 | 4 | `teacher_dataset_Ant/` |
| Hopper-v2 | ep0200_return820 | 10 | `teacher_dataset_Hopper/` |
| Hopper-v2 | ep0900_return3061 | 10 | `teacher_dataset_Hopper/` |
| HalfCheetah-v2 | ep0200_return3335 | 4 | `teacher_dataset_HC/` |
| Walker2d-v2 | ep0100_return2598 | 10 | `teacher_dataset_Walker2d/` |
| Walker2d-v2 | ep0300_return3891 | 10 | `teacher_dataset_Walker2d/` |

---

## 3. Dataset Naming

The generation script produces informative filenames using the **actual rollout mean return**
(not the checkpoint training return):

```
expert_data_no_img_{ENV}_scores_{ACTUAL_RETURN}_episodes_{N}_ep{TRAINING_EP}.npz
```

Examples:
```
teacher_dataset_Ant/expert_data_no_img_Ant_scores_????_episodes_4_ep200.npz
teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_????_episodes_10_ep200.npz
teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_????_episodes_4_ep200.npz
teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_????_episodes_10_ep100.npz
```

The `????` score is filled in at generation time with the measured rollout return.

---

## 4. Sbatch Structure for sail_sb3

**Target folders** (per user instruction):
- `sail_sb3/Ant_sbatch/`
- `sail_sb3/HC_sbatch/`
- `sail_sb3/Hopper_sbatch/`
- `sail_sb3/Walker2d_sbatch/`

**Variants created per dataset:**

Only **SAIL** (base Adaptive + LfD, no offline RM) is created here. The existing
`Ant_1200_sbatch/`, `Hopper_1500_sbatch/` etc. require pre-trained offline RM models
(BPref). Those RM models have only been trained for the original optimal datasets
(`scores_1200`, `scores_1500`, etc.), not for the new sub-optimal checkpoints.
Creating PREF-D, QPREF, TAC-D variants would require offline RM training first.

**Sbatch naming:**
```
{ENV}_ep{EP}_n{N}_SAIL.sbatch         # full run
{ENV}_ep{EP}_n{N}_SAIL_smoke.sbatch   # smoke test (10k steps)
```

Examples:
```
sail_sb3/Ant_sbatch/Ant_ep0200_n4_SAIL.sbatch
sail_sb3/Ant_sbatch/Ant_ep1000_n4_SAIL.sbatch
sail_sb3/HC_sbatch/HC_ep0200_n4_SAIL.sbatch
sail_sb3/Hopper_sbatch/Hopper_ep0200_n10_SAIL.sbatch
sail_sb3/Hopper_sbatch/Hopper_ep0900_n10_SAIL.sbatch
sail_sb3/Walker2d_sbatch/Walker2d_ep0100_n10_SAIL.sbatch
sail_sb3/Walker2d_sbatch/Walker2d_ep0300_n10_SAIL.sbatch
```

**Script called:** `sail_sb3/scripts/train_sail.py`  
**PYTHONPATH:** `${REPO_ROOT}` (same as existing Ant_1200_sbatch)

---

## 5. Sbatch Structure for sail_sb3_online

**Target folders** (created if absent):
- `sail_sb3_online/Ant_sbatch/`
- `sail_sb3_online/HC_sbatch/`
- `sail_sb3_online/Hopper_sbatch/`
- `sail_sb3_online/Walker2d_sbatch/`

**Variant created:** **PAIL** (PAIL + online_rm). The online RM is trained from scratch
during the run — no pre-trained offline RM needed. This is the canonical online variant.

**Sbatch naming:**
```
{ENV}_ep{EP}_n{N}_PAIL.sbatch         # full run
{ENV}_ep{EP}_n{N}_PAIL_smoke.sbatch   # smoke test (10k steps)
```

**Script called:** `sail_sb3_online/scripts/train_sail.py`  
**PYTHONPATH:** `${REPO_ROOT}/sail_sb3_online`

**Per-env `pref_beta`** (from existing online sbatches):
- Ant-v2: 5.0
- Hopper-v2: 1.0
- HalfCheetah-v2: 1.0
- Walker2d-v2: 1.0

---

## 6. Dataset Validation

`_verify()` runs inside the generation script automatically. It checks:
1. All 5 required keys present
2. `obs.ndim == 2`
3. `actions.ndim == 2`
4. `rewards.shape == (N, 1)` — 2D, SAIL requires this
5. `len(obs) == len(actions) == len(rewards) == len(episode_starts)`
6. `episode_returns.ndim == 1`
7. `episode_starts.dtype == object`
8. Episode boundary count == `len(episode_returns)`
9. `demo_dones` derivation consistent with episode count

After generation, shapes are reported for each dataset.

---

## 7. Smoke Validation Strategy

**One sbatch from sail_sb3 + one from sail_sb3_online.**

Selected pair (Ant ep0200, 4 trajs — simplest new dataset):
- `sail_sb3/Ant_sbatch/Ant_ep0200_n4_SAIL_smoke.sbatch` (10k steps, seed 0)
- `sail_sb3_online/Ant_sbatch/Ant_ep0200_n4_PAIL_smoke.sbatch` (10k steps, seed 0)

Both are **separate smoke sbatches** — the full-run sbatches are never edited.

**Success criteria:**
- EXIT CODE 0
- Expert returns printed at startup (no KeyError, no shape mismatch)
- Discriminator begins training
- `rollout/ep_rew_mean` appears in logs

---

## 8. No Full Runs During Validation

Smoke sbatches use `--total_timesteps 10000`. The full-run sbatches (`*_SAIL.sbatch`,
`*_PAIL.sbatch`) are created but **not submitted** during this step. Only smoke sbatches
are submitted.

---

## 9. Directory Layout After Implementation

```
teacher_dataset_Ant/
  expert_data_no_img_Ant_scores_????_episodes_4_ep200.npz
  expert_data_no_img_Ant_scores_????_episodes_4_ep1000.npz

teacher_dataset_Hopper/
  expert_data_no_img_Hopper_scores_????_episodes_10_ep200.npz
  expert_data_no_img_Hopper_scores_????_episodes_10_ep900.npz

teacher_dataset_HC/
  expert_data_no_img_HalfCheetah_scores_????_episodes_4_ep200.npz
  (+ existing ep300, ep400, ep1000 files)

teacher_dataset_Walker2d/
  expert_data_no_img_Walker2d_scores_????_episodes_10_ep100.npz
  expert_data_no_img_Walker2d_scores_????_episodes_10_ep300.npz

sail_sb3/
  Ant_sbatch/
    Ant_ep0200_n4_SAIL.sbatch
    Ant_ep0200_n4_SAIL_smoke.sbatch   ← smoke-tested
    Ant_ep1000_n4_SAIL.sbatch
    logs/
  HC_sbatch/
    HC_ep0200_n4_SAIL.sbatch
    (+ existing ep300, ep400, ep1000 sbatches)
    logs/
  Hopper_sbatch/
    Hopper_ep0200_n10_SAIL.sbatch
    Hopper_ep0900_n10_SAIL.sbatch
    logs/
  Walker2d_sbatch/
    Walker2d_ep0100_n10_SAIL.sbatch
    Walker2d_ep0300_n10_SAIL.sbatch
    logs/

sail_sb3_online/
  Ant_sbatch/
    Ant_ep0200_n4_PAIL.sbatch
    Ant_ep0200_n4_PAIL_smoke.sbatch   ← smoke-tested
    Ant_ep1000_n4_PAIL.sbatch
    logs/
  HC_sbatch/
    HC_ep0200_n4_PAIL.sbatch
    logs/
  Hopper_sbatch/
    Hopper_ep0200_n10_PAIL.sbatch
    Hopper_ep0900_n10_PAIL.sbatch
    logs/
  Walker2d_sbatch/
    Walker2d_ep0100_n10_PAIL.sbatch
    Walker2d_ep0300_n10_PAIL.sbatch
    logs/
```
