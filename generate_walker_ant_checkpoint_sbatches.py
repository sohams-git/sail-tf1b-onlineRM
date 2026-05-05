"""
generate_walker_ant_checkpoint_sbatches.py
==========================================
Generates all sail_sb3 (offline) and sail_sb3_online sbatches for new
Walker2d-v2 and Ant-v2 checkpoint-based datasets, mirroring exactly
the existing sbatch folders:

  sail_sb3/Walker2d_sbatch/      (9 offline variants per checkpoint)
  sail_sb3_online/Walker2d_sbatch/  (7 online variants per checkpoint)
  sail_sb3/Ant_sbatch/           (9 offline variants per checkpoint)
  sail_sb3_online/Ant_sbatch/    (7 online variants per checkpoint)

Skips any sbatch file that already exists.
Skips any dataset whose NPZ file does not exist on disk.
"""

import os
import glob

REPO_ROOT   = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN  = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"

# ── Offline RM paths ─────────────────────────────────────────────────────────
WALKER_PREF_RM = (
    "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/"
    "walker2d_pebble_1M/reward_model_1000000_0.pt"
)
ANT_PREF_RM = (
    "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/"
    "ant_pebble_1M/reward_model_1000000_0.pt"
)

# ── Walker2d checkpoints to process ──────────────────────────────────────────
WALKER_CHECKPOINTS = [
    {"ep": "0020", "n": 10,
     "npz_glob": "teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_*_episodes_10_ep20.npz"},
    {"ep": "0040", "n": 10,
     "npz_glob": "teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_*_episodes_10_ep40.npz"},
]

# ── Ant checkpoints to process ────────────────────────────────────────────────
ANT_CHECKPOINTS = [
    {"ep": "0020", "n": 4,
     "npz_glob": "teacher_dataset_Ant/expert_data_no_img_Ant_scores_*_episodes_4_ep20.npz"},
    {"ep": "0040", "n": 4,
     "npz_glob": "teacher_dataset_Ant/expert_data_no_img_Ant_scores_*_episodes_4_ep40.npz"},
]

# ── Online RM args ────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _env_exports():
    return """\
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"""

def _header(job_name, mem, time_limit, log_path, array="1-5"):
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
#SBATCH --array={array}
#SBATCH --account=sabymath0
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

def _find_npz(pattern_rel):
    matches = glob.glob(os.path.join(REPO_ROOT, pattern_rel))
    return matches[0] if matches else None

def _score_from_fname(fname):
    return int(os.path.basename(fname).split("_scores_")[1].split("_")[0])


# ============================================================================
#  WALKER2D sbatch builders
# ============================================================================

def _w2d_offline_sail(ep, score, n, npz_rel, offline_dir):
    tag  = f"Walker2d_ep{ep}_n{n}_SAIL"
    log  = f"{REPO_ROOT}/sail_sb3/Walker2d_sbatch/logs/{tag}"
    body = f"""\
{_header(f"W2d_ep{ep}_n{n}_SAIL", "8G", "24:00:00", log)}
# FULL RUN — SAIL + Adaptive + LfD | Walker2d-v2
# Dataset: {npz_rel}

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Walker2d-v2_ep{ep}_sail"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Walker2d-v2 \\
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


def _w2d_offline_pref(ep, score, n, npz_rel, variant):
    pref_rm  = WALKER_PREF_RM
    obs_dim  = 17
    pref_beta = 1.0

    if variant == "PAIL":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "SAIL_PREFRANK":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_PREFRANK"
        extra = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_PREFRANK":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_PREFRANK"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_TACD"
        extra = f"""\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.1 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_TACD"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --soft_tac \\
  --soft_tac_weight 0.1 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_QPREF"
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
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_QPREF":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        extra = f"""\
  --adaptive_score_source gt \\
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

    log   = f"{REPO_ROOT}/sail_sb3/Walker2d_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"W2d_ep{ep}s{score}_{variant}", "12G", "24:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {variant} + Adaptive + LfD | Walker2d-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Walker2d-v2_ep{ep}_s{score}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Walker2d-v2 \\
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


def _w2d_offline_discrewabl(ep, score, n, npz_rel):
    tag  = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_PREFRANK_DiscRewAbl"
    log  = f"{REPO_ROOT}/sail_sb3/Walker2d_sbatch/logs/{tag}"
    pref_rm = WALKER_PREF_RM
    body = f"""\
{_header(f"W2d_ep{ep}s{score}_SAILPD_DiscRew", "12G", "24:00:00", log, array="1-3")}
# Discriminator reward ablation — SAIL + PREFRANK-Disc (Walker2d-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: SAIL + PREFRANK-Disc + Adaptive + LfD  [disc reward ablation] | Walker2d-v2"
echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="Walker2d_ep{ep}_score{score}_n{n}_SAIL_PREFRANK_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Walker2d-v2_ep{ep}_s{score}_sail_prefrank_discrew_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Walker2d-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim 17 \\
  --entcoeff 0.05 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _w2d_online(ep, score, n, npz_rel, variant):
    pref_rm  = WALKER_PREF_RM
    obs_dim  = 17
    pref_beta = 1.0

    if variant == "PAIL":
        tag   = f"Walker2d_ep{ep}_n{n}_PAIL"
        mem   = "8G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFRANK":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_PREFRANK"
        mem   = "12G"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFRANK":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_PREFRANK"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_TACD"
        mem   = "12G"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.1 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_TACD"
        mem   = "12G"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --soft_tac \\
  --soft_tac_weight 0.1 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        mem   = "12G"
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
        tag   = f"Walker2d_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        mem   = "12G"
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

    log   = f"{REPO_ROOT}/sail_sb3_online/Walker2d_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"W2d_ep{ep}s{score}_{variant}", mem, "12:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3_online/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: Online RM + {variant} + Adaptive + LfD | Walker2d-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Walker2d"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Walker2d-v2_ep{ep}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"${{PYTHON_BIN}}" sail_sb3_online/scripts/train_sail.py \\
  --env Walker2d-v2 \\
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
#  ANT sbatch builders
# ============================================================================

def _ant_offline_sail(ep, score, n, npz_rel):
    tag  = f"Ant_ep{ep}_n{n}_SAIL"
    log  = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    body = f"""\
{_header(f"Ant_ep{ep}_n{n}_SAIL", "12G", "24:00:00", log)}
# FULL RUN — SAIL + Adaptive + LfD | Ant-v2
# Dataset: {npz_rel}

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_ep{ep}_sail"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
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


def _ant_offline_pref(ep, score, n, npz_rel, variant):
    pref_rm   = ANT_PREF_RM
    obs_dim   = 111
    pref_beta = 5.0   # Ant uses pref_beta=5.0 (from reference sbatches)

    if variant == "PAIL":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "SAIL_PREFD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_PREFD"
        extra = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_PREFD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_PREFD"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_TACD"
        extra = f"""\
  --adaptive_score_source rm \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_TACD"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_QPREF"
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
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_QPREF":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        extra = f"""\
  --adaptive_score_source gt \\
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

    log   = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"Ant_ep{ep}s{score}_{variant}", "12G", "24:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {variant} + Adaptive + LfD | Ant-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_ep{ep}_s{score}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
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


def _ant_offline_discrewabl(ep, score, n, npz_rel):
    tag  = f"Ant_ep{ep}_score{score}_n{n}_SAIL_PREFD_DiscRewAbl"
    log  = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    pref_rm = ANT_PREF_RM
    body = f"""\
{_header(f"Ant_ep{ep}s{score}_SAILPD_DiscRew", "12G", "24:00:00", log, array="1-3")}
# Discriminator reward ablation — SAIL + PREF-D-Disc (Ant-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: SAIL + PREF-D-Disc + Adaptive + LfD  [disc reward ablation] | Ant-v2"
echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_ep{ep}_score{score}_n{n}_SAIL_PREFD_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_ep{ep}_s{score}_sail_prefd_discrew_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
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
  --pref_expect_obs_dim 111 \\
  --entcoeff 0.05 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _ant_online(ep, score, n, npz_rel, variant):
    pref_rm   = ANT_PREF_RM
    obs_dim   = 111
    pref_beta = 5.0

    if variant == "PAIL":
        tag   = f"Ant_ep{ep}_n{n}_PAIL"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_PREFD"
        mem   = "12G"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_PREFD"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_TACD"
        mem   = "12G"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_TACD"
        mem   = "12G"
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
        tag   = f"Ant_ep{ep}_score{score}_n{n}_SAIL_QPREF"
        mem   = "12G"
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
        tag   = f"Ant_ep{ep}_score{score}_n{n}_PAIL_QPREF"
        mem   = "12G"
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

    log   = f"{REPO_ROOT}/sail_sb3_online/Ant_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"Ant_ep{ep}s{score}_{variant}", mem, "12:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3_online/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: Online RM + {variant} + Adaptive + LfD | Ant-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_ep{ep}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"${{PYTHON_BIN}}" sail_sb3_online/scripts/train_sail.py \\
  --env Ant-v2 \\
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

def generate_walker(npz_path, npz_rel, ep, score, n):
    offline_dir = os.path.join(REPO_ROOT, "sail_sb3/Walker2d_sbatch")
    online_dir  = os.path.join(REPO_ROOT, "sail_sb3_online/Walker2d_sbatch")
    created = []

    # Offline: SAIL + 7 pref variants + DiscRewAbl = 9
    fname, body = _w2d_offline_sail(ep, score, n, npz_rel, offline_dir)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(os.path.join(offline_dir, fname))

    for v in ["PAIL", "SAIL_PREFRANK", "PAIL_PREFRANK",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _w2d_offline_pref(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(offline_dir, fname), body):
            created.append(os.path.join(offline_dir, fname))

    fname, body = _w2d_offline_discrewabl(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(os.path.join(offline_dir, fname))

    # Online: 7 variants (no SAIL-only, no DiscRewAbl)
    for v in ["PAIL", "SAIL_PREFRANK", "PAIL_PREFRANK",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _w2d_online(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(online_dir, fname), body):
            created.append(os.path.join(online_dir, fname))

    return created


def generate_ant(npz_path, npz_rel, ep, score, n):
    offline_dir = os.path.join(REPO_ROOT, "sail_sb3/Ant_sbatch")
    online_dir  = os.path.join(REPO_ROOT, "sail_sb3_online/Ant_sbatch")
    created = []

    # Offline: SAIL + 7 pref variants + DiscRewAbl = 9
    fname, body = _ant_offline_sail(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(os.path.join(offline_dir, fname))

    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _ant_offline_pref(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(offline_dir, fname), body):
            created.append(os.path.join(offline_dir, fname))

    fname, body = _ant_offline_discrewabl(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(os.path.join(offline_dir, fname))

    # Online: 7 variants
    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _ant_online(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(online_dir, fname), body):
            created.append(os.path.join(online_dir, fname))

    return created


def main():
    total = 0

    # ── Walker2d ──────────────────────────────────────────────────────────────
    print("\n=== Walker2d-v2 ===")
    for ckpt in WALKER_CHECKPOINTS:
        ep  = ckpt["ep"]
        n   = ckpt["n"]
        npz = _find_npz(ckpt["npz_glob"])
        if npz is None:
            print(f"  Walker2d ep{ep}: dataset not found — skipping ({ckpt['npz_glob']})")
            continue
        npz_rel = os.path.relpath(npz, REPO_ROOT)
        score   = _score_from_fname(npz)
        print(f"\n  Walker2d ep{ep} score{score} n{n}: {os.path.basename(npz)}")
        created = generate_walker(npz, npz_rel, ep, score, n)
        print(f"  → {len(created)} sbatches created")
        total  += len(created)

    # ── Ant ───────────────────────────────────────────────────────────────────
    print("\n=== Ant-v2 ===")
    for ckpt in ANT_CHECKPOINTS:
        ep  = ckpt["ep"]
        n   = ckpt["n"]
        npz = _find_npz(ckpt["npz_glob"])
        if npz is None:
            print(f"  Ant ep{ep}: dataset not found — skipping ({ckpt['npz_glob']})")
            continue
        npz_rel = os.path.relpath(npz, REPO_ROOT)
        score   = _score_from_fname(npz)
        print(f"\n  Ant ep{ep} score{score} n{n}: {os.path.basename(npz)}")
        created = generate_ant(npz, npz_rel, ep, score, n)
        print(f"  → {len(created)} sbatches created")
        total  += len(created)

    print(f"\n=== DONE: {total} total sbatches created ===")


if __name__ == "__main__":
    main()
