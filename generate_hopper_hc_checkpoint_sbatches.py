"""
generate_hopper_hc_checkpoint_sbatches.py
==========================================
Generates all sail_sb3 (offline) and sail_sb3_online sbatches for new
Hopper-v2 and HalfCheetah-v2 checkpoint-based datasets, mirroring exactly
the existing sbatch folders:
  sail_sb3/Hopper_sbatch/     (9 offline variants)
  sail_sb3_online/Hopper_sbatch/  (7 online variants)
  sail_sb3/HC_sbatch/         (9 offline variants)
  sail_sb3_online/HC_sbatch/  (7 online variants)

Skips any sbatch file that already exists. Skips any dataset whose NPZ
file does not exist on disk.
"""

import os

REPO_ROOT = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"

# ── Offline RM paths ────────────────────────────────────────────────────────
HOPPER_PREF_RM = (
    "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/"
    "hopper_pebble_1M/reward_model_1000000_0.pt"
)
HC_PREF_RM = (
    "/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/"
    "pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt"
)

# ── Online RM common args (same for all envs) ───────────────────────────────
ONLINE_RM_ARGS = """\
  --online_rm \\
  --rm_segment_len 50 \\
  --rm_train_freq 1000 \\
  --rm_gradient_steps 10 \\
  --rm_batch_size 256 \\
  --rm_lr 3e-4 \\
  --rm_max_segments 10000 \\
  --rm_min_segments 500 \\
  --rm_min_updates 50 \\
  --rm_min_acc 0.60 \\
  --rm_tie_margin 0.5 \\
  --rm_rescore_freq 20000 \\"""

# ── Common sbatch header ────────────────────────────────────────────────────
def _header(job_name, mem, time_limit, log_path, is_array_1_3=False):
    array_spec = "1-3" if is_array_1_3 else "1-5"
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}_%A_%a.out
#SBATCH --array={array_spec}
#SBATCH --account=sabymath0
"""

def _env_exports():
    return """\
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"""

def _write_if_new(path, content):
    if os.path.exists(path):
        print(f"  [SKIP already exists] {os.path.basename(path)}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [CREATED] {os.path.basename(path)}")
    return True


# ============================================================================
#  HOPPER
# ============================================================================

HOPPER_DATASETS = [
    {
        "ep": "0100", "score": None, "n": 10,  # score filled after rollout
        "npz": "teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_{score}_episodes_10_ep100.npz",
    },
]

def _find_hopper_npz(ep):
    """Glob for the actual NPZ (score determined by rollout)."""
    import glob
    pattern = os.path.join(
        REPO_ROOT, f"teacher_dataset_Hopper",
        f"expert_data_no_img_Hopper_scores_*_episodes_10_ep{ep}.npz"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _hopper_offline_sail(d, ep, score, n, npz_rel):
    tag = f"Hopper_ep{ep}_n{n}_SAIL"
    log = f"{REPO_ROOT}/sail_sb3/Hopper_sbatch/logs/{tag}"
    body = f"""\
{_header(f"Hop_ep{ep}_n{n}_SAIL", "8G", "24:00:00", log)}
# FULL RUN — SAIL + Adaptive + LfD | Hopper-v2
# Dataset: {npz_rel}

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Hopper_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Hopper"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Hopper-v2_ep{ep}_sail"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Hopper-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hopper_offline_pref(d, ep, score, n, npz_rel, variant):
    """Generates all pref offline variants for Hopper."""
    pref_rm = HOPPER_PREF_RM
    obs_dim = 11
    pref_beta = 1.0

    if variant == "PAIL":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "SAIL_PREFRANK":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_PREFRANK"
        extra = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.03 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_PREFRANK":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_PREFRANK"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --pref_rank_disc \\
  --pref_rank_weight 0.03 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_TACD"
        extra = f"""\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.03 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_TACD":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_TACD"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --soft_tac \\
  --soft_tac_weight 0.03 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        extra = f"""\
  --adaptive_score_source gt \\
  --qpref \\
  --qpref_weight 0.03 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_QPREF":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log = f"{REPO_ROOT}/sail_sb3/Hopper_sbatch/logs/{tag}"
    short_v = variant.lower().replace("_", "")
    body = f"""\
{_header(f"Hop_ep{ep}s{score}_{variant}", "12G", "24:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Hopper_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {variant} + Adaptive + LfD | Hopper-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Hopper"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Hopper-v2_ep{ep}_s{score}_{short_v}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Hopper-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hopper_offline_discrewabl(d, ep, score, n, npz_rel):
    tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_PREFRANK_DiscRewAbl"
    log = f"{REPO_ROOT}/sail_sb3/Hopper_sbatch/logs/{tag}"
    pref_rm = HOPPER_PREF_RM
    body = f"""\
{_header(f"Hop_ep{ep}s{score}_SAILPD_DiscRew", "12G", "24:00:00", log, is_array_1_3=True)}
# Discriminator reward ablation — SAIL + PREFRANK-Disc (Hopper-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Hopper_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: SAIL + PREFRANK-Disc + Adaptive + LfD  [disc reward ablation] | Hopper-v2"
echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Hopper"
export WANDB_NAME="Hopper_ep{ep}_score{score}_n{n}_SAIL_PREFRANK_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Hopper-v2_ep{ep}_s{score}_sail_prefrank_discrew_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Hopper-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.03 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim 11 \\
  --entcoeff 0.05 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hopper_online(d, ep, score, n, npz_rel, variant):
    """Generates online variants for Hopper."""
    pref_beta = 1.0

    if variant == "PAIL":
        tag = f"Hopper_ep{ep}_n{n}_PAIL"
        mem = "8G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFRANK":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_PREFRANK"
        mem = "12G"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.03 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFRANK":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_PREFRANK"
        mem = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.03 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_TACD"
        mem = "12G"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.03 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_TACD"
        mem = "12G"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --soft_tac \\
  --soft_tac_weight 0.03 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        mem = "12G"
        extra = """\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    elif variant == "PAIL_QPREF":
        tag = f"Hopper_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        mem = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log = f"{REPO_ROOT}/sail_sb3_online/Hopper_sbatch/logs/{tag}"
    short_v = variant.lower().replace("_", "")
    body = f"""\
{_header(f"Hop_ep{ep}s{score}_{variant}" if "SAIL" in tag or "s" not in tag else f"Hop_ep{ep}_{n}_{variant}", mem, "12:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3_online/Hopper_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: Online RM + {variant} + Adaptive + LfD | Hopper-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Hopper"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Hopper-v2_ep{ep}_{short_v}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"${{PYTHON_BIN}}" sail_sb3_online/scripts/train_sail.py \\
  --env Hopper-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
{ONLINE_RM_ARGS}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


# ============================================================================
#  HALFCHEETAH
# ============================================================================

def _hc_offline_sail(d, ep, score, n, npz_rel):
    tag = f"HC_ep{ep}_n{n}_SAIL"
    log = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    body = f"""\
{_header(f"HC_ep{ep}_n{n}_SAIL", "8G", "24:00:00", log)}
# FULL RUN — SAIL + Adaptive + LfD | HalfCheetah-v2
# Dataset: {npz_rel}

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/HC_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="HalfCheetah-v2_ep{ep}_sail"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env HalfCheetah-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --entcoeff 0.01 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hc_offline_pref(d, ep, score, n, npz_rel, variant):
    """Generates all pref offline variants for HalfCheetah."""
    pref_rm = HC_PREF_RM
    pref_beta = 1.0

    if variant == "PAIL":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL"
        entcoeff = "0.01"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "SAIL_PREFD":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_PREFD"
        entcoeff = "0.01"
        extra = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "PAIL_PREFD":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_PREFD"
        entcoeff = "0.01"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\"""
    elif variant == "SAIL_TACD":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_TACD"
        entcoeff = "0.05"
        extra = f"""\
  --adaptive_score_source rm \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "PAIL_TACD":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_TACD"
        entcoeff = "0.05"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        entcoeff = "0.05"
        extra = f"""\
  --adaptive_score_source gt \\
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "PAIL_QPREF":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        entcoeff = "0.05"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    short_v = variant.lower().replace("_", "")
    body = f"""\
{_header(f"HC_ep{ep}s{score}_{variant}", "8G", "24:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/HC_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {variant} + Adaptive + LfD | HalfCheetah-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="HalfCheetah-v2_ep{ep}_s{score}_{short_v}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env HalfCheetah-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
  --entcoeff {entcoeff} \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hc_offline_discrewabl(d, ep, score, n, npz_rel):
    tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_PREFD_DiscRewAbl"
    log = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    pref_rm = HC_PREF_RM
    body = f"""\
{_header(f"HC_ep{ep}s{score}_SAILPD_DiscRew", "8G", "24:00:00", log, is_array_1_3=True)}
# Discriminator reward ablation — SAIL + PREF-D-Disc (HalfCheetah-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/HC_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: SAIL + PREF-D-Disc + Adaptive + LfD  [disc reward ablation] | HalfCheetah-v2"
echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="HC_ep{ep}_score{score}_n{n}_SAIL_PREFD_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="HalfCheetah-v2_ep{ep}_s{score}_sail_prefd_discrew_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env HalfCheetah-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --entcoeff 0.01 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _hc_online(d, ep, score, n, npz_rel, variant):
    """Generates online variants for HalfCheetah."""
    pref_beta = 1.0

    if variant == "PAIL":
        tag = f"HC_ep{ep}_n{n}_PAIL"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFD":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_PREFD"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFD":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_PREFD"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\"""
    elif variant == "SAIL_TACD":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_TACD"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_TACD"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag = f"HC_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        extra = """\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    elif variant == "PAIL_QPREF":
        tag = f"HC_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log = f"{REPO_ROOT}/sail_sb3_online/HC_sbatch/logs/{tag}"
    short_v = variant.lower().replace("_", "")
    body = f"""\
{_header(f"HC_ep{ep}s{score}_{variant}", "12G", "12:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3_online/HC_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: Online RM + {variant} + Adaptive + LfD | HalfCheetah-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_OnlineRM_HalfCheetah"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="HalfCheetah-v2_ep{ep}_{short_v}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"${{PYTHON_BIN}}" sail_sb3_online/scripts/train_sail.py \\
  --env HalfCheetah-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
{ONLINE_RM_ARGS}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


# ============================================================================
#  MAIN
# ============================================================================

def generate_hopper(npz_path, npz_rel, ep, score, n):
    offline_dir = os.path.join(REPO_ROOT, "sail_sb3/Hopper_sbatch")
    online_dir  = os.path.join(REPO_ROOT, "sail_sb3_online/Hopper_sbatch")
    d = {}
    created = []

    # Offline
    fname, body = _hopper_offline_sail(d, ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    for v in ["PAIL", "SAIL_PREFRANK", "PAIL_PREFRANK", "SAIL_TACD",
              "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _hopper_offline_pref(d, ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(offline_dir, fname), body):
            created.append(fname)

    fname, body = _hopper_offline_discrewabl(d, ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    # Online
    for v in ["PAIL", "SAIL_PREFRANK", "PAIL_PREFRANK", "SAIL_TACD",
              "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _hopper_online(d, ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(online_dir, fname), body):
            created.append(fname)

    return created


def generate_hc(npz_path, npz_rel, ep, score, n):
    offline_dir = os.path.join(REPO_ROOT, "sail_sb3/HC_sbatch")
    online_dir  = os.path.join(REPO_ROOT, "sail_sb3_online/HC_sbatch")
    d = {}
    created = []

    # Offline
    fname, body = _hc_offline_sail(d, ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD", "SAIL_TACD",
              "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _hc_offline_pref(d, ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(offline_dir, fname), body):
            created.append(fname)

    fname, body = _hc_offline_discrewabl(d, ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    # Online
    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD", "SAIL_TACD",
              "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _hc_online(d, ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(online_dir, fname), body):
            created.append(fname)

    return created


def main():
    import glob

    total = 0

    # ── Hopper ep0100 ──────────────────────────────────────────────────────
    hop_pattern = os.path.join(
        REPO_ROOT, "teacher_dataset_Hopper",
        "expert_data_no_img_Hopper_scores_*_episodes_10_ep100.npz"
    )
    hop_matches = glob.glob(hop_pattern)
    if not hop_matches:
        print("Hopper ep100: dataset not found — skipping")
    else:
        npz_path = hop_matches[0]
        npz_rel = os.path.relpath(npz_path, REPO_ROOT)
        fname = os.path.basename(npz_path)
        # extract score from filename
        score = int(fname.split("_scores_")[1].split("_")[0])
        n = 10
        ep = "100"
        print(f"\nHopper ep{ep} score{score} n{n}: {fname}")
        created = generate_hopper(npz_path, npz_rel, ep, score, n)
        print(f"  Created {len(created)} sbatches")
        total += len(created)

    # ── HalfCheetah ep100 ─────────────────────────────────────────────────
    hc_pattern = os.path.join(
        REPO_ROOT, "teacher_dataset_HC",
        "expert_data_no_img_HalfCheetah_scores_*_episodes_4_ep100.npz"
    )
    hc_matches = glob.glob(hc_pattern)
    if not hc_matches:
        print("HalfCheetah ep100: dataset not found — skipping")
    else:
        npz_path = hc_matches[0]
        npz_rel = os.path.relpath(npz_path, REPO_ROOT)
        fname = os.path.basename(npz_path)
        score = int(fname.split("_scores_")[1].split("_")[0])
        n = 4
        ep = "100"
        print(f"\nHalfCheetah ep{ep} score{score} n{n}: {fname}")
        created = generate_hc(npz_path, npz_rel, ep, score, n)
        print(f"  Created {len(created)} sbatches")
        total += len(created)

    print(f"\nTotal sbatches created: {total}")


if __name__ == "__main__":
    main()
