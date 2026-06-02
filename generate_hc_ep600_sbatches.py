"""
generate_hc_ep600_sbatches.py
==============================
Generates all sail_sb3 (offline) and sail_sb3_online sbatches for the new
HalfCheetah ep0600 checkpoint dataset, mirroring exactly the HC_ep0200
reference sbatches in sail_sb3/HC_sbatch/ and sail_sb3_online/HC_sbatch/.

Offline variants (9): SAIL, PAIL, SAIL_PREFD, PAIL_PREFD,
                       SAIL_TACD, PAIL_TACD, SAIL_QPREF, PAIL_QPREF,
                       SAIL_PREFD_DiscRewAbl
Online variants (7):  PAIL, SAIL_PREFD, PAIL_PREFD,
                      SAIL_TACD, PAIL_TACD, SAIL_QPREF, PAIL_QPREF

Key HC constants (from reference sbatches):
  pref_rm         : /nfs/turbo/umd-sridas/.../reward_model_ts.pt
  pref_rank_batch_size: 16
  pref_beta       : 1.0
  soft_tac_weight : 0.5 (offline), 0.5 (online)
  qpref_weight    : 0.05 (offline), 0.1 (online)
  entcoeff        : 0.01 for SAIL/PAIL/PREFD variants (offline)
                    0.05 for TACD/QPREF variants (offline) and ALL online
  DiscRewAbl      : airl_backward_kl, array=1-3
"""

import os
import glob

REPO_ROOT  = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
HC_PREF_RM = (
    "/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/"
    "pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt"
)

OFFLINE_DIR = os.path.join(REPO_ROOT, "sail_sb3/HC_sbatch")
ONLINE_DIR  = os.path.join(REPO_ROOT, "sail_sb3_online/HC_sbatch")

# ep label used in filenames (zero-padded to 4 digits)
EP_LABEL = "ep0600"
# glob pattern to find the actual NPZ (score determined by rollout)
NPZ_GLOB = "teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_*_episodes_4_ep600.npz"

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
        print(f"  [SKIP exists] {os.path.basename(path)}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [CREATED] {os.path.basename(path)}")
    return True


# ── Offline sbatch builders ───────────────────────────────────────────────────

def _offline_sail(ep, score, n, npz_rel):
    tag  = f"HC_{ep}_n{n}_SAIL"
    log  = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    body = f"""\
{_header(f"HC_{ep}_n{n}_SAIL", "8G", "24:00:00", log)}
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
export WANDB_GROUP="HalfCheetah-v2_{ep}_sail"
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


def _offline_pref(ep, score, n, npz_rel, variant):
    pref_rm = HC_PREF_RM
    pref_beta = 1.0

    if variant == "PAIL":
        tag     = f"HC_{ep}_score{score}_n{n}_PAIL"
        entcoef = "0.01"
        extra   = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "SAIL_PREFD":
        tag     = f"HC_{ep}_score{score}_n{n}_SAIL_PREFD"
        entcoef = "0.01"
        extra   = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "PAIL_PREFD":
        tag     = f"HC_{ep}_score{score}_n{n}_PAIL_PREFD"
        entcoef = "0.01"
        extra   = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\"""
    elif variant == "SAIL_TACD":
        tag     = f"HC_{ep}_score{score}_n{n}_SAIL_TACD"
        entcoef = "0.05"
        extra   = f"""\
  --adaptive_score_source rm \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\"""
    elif variant == "PAIL_TACD":
        tag     = f"HC_{ep}_score{score}_n{n}_PAIL_TACD"
        entcoef = "0.05"
        extra   = f"""\
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
        tag     = f"HC_{ep}_score{score}_n{n}_SAIL_QPREF"
        entcoef = "0.05"
        extra   = f"""\
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
        tag     = f"HC_{ep}_score{score}_n{n}_PAIL_QPREF"
        entcoef = "0.05"
        extra   = f"""\
  --adaptive_score_source gt \\
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

    log   = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"HC_{ep}s{score}_{variant}", "8G", "24:00:00", log)}
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
export WANDB_GROUP="HalfCheetah-v2_{ep}_s{score}_{short}"
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
  --entcoeff {entcoef} \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _offline_discrewabl(ep, score, n, npz_rel):
    tag = f"HC_{ep}_score{score}_n{n}_SAIL_PREFD_DiscRewAbl"
    log = f"{REPO_ROOT}/sail_sb3/HC_sbatch/logs/{tag}"
    body = f"""\
{_header(f"HC_{ep}s{score}_SAILPD_DiscRew", "8G", "24:00:00", log, array="1-3")}
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
export WANDB_NAME="HC_{ep}_score{score}_n{n}_SAIL_PREFD_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="HalfCheetah-v2_{ep}_s{score}_sail_prefd_discrew_${{DISC_REWARD_TYPE}}"
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
  --pref_rm "{HC_PREF_RM}" \\
  --entcoeff 0.01 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


# ── Online sbatch builders ────────────────────────────────────────────────────

def _online(ep, score, n, npz_rel, variant):
    pref_rm   = HC_PREF_RM
    pref_beta = 1.0

    if variant == "PAIL":
        tag   = f"HC_{ep}_n{n}_PAIL"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFD":
        tag   = f"HC_{ep}_score{score}_n{n}_SAIL_PREFD"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFD":
        tag   = f"HC_{ep}_score{score}_n{n}_PAIL_PREFD"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"HC_{ep}_score{score}_n{n}_SAIL_TACD"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag   = f"HC_{ep}_score{score}_n{n}_PAIL_TACD"
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
        tag   = f"HC_{ep}_score{score}_n{n}_SAIL_QPREF"
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
        tag   = f"HC_{ep}_score{score}_n{n}_PAIL_QPREF"
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

    log   = f"{REPO_ROOT}/sail_sb3_online/HC_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"HC_{ep}s{score}_{variant}", "12G", "12:00:00", log)}
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
export WANDB_GROUP="HalfCheetah-v2_{ep}_{short}"
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    npz_matches = glob.glob(os.path.join(REPO_ROOT, NPZ_GLOB))
    if not npz_matches:
        print(f"Dataset not found: {NPZ_GLOB} — aborting")
        return

    npz_path = npz_matches[0]
    npz_rel  = os.path.relpath(npz_path, REPO_ROOT)
    fname    = os.path.basename(npz_path)
    score    = int(fname.split("_scores_")[1].split("_")[0])
    n        = 4
    ep       = EP_LABEL
    print(f"Dataset: {fname}  (score={score})")

    created = []

    # ── offline ──────────────────────────────────────────────────────────────
    print("\n-- offline --")
    fname_sb, body = _offline_sail(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(OFFLINE_DIR, fname_sb), body):
        created.append(fname_sb)

    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname_sb, body = _offline_pref(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(OFFLINE_DIR, fname_sb), body):
            created.append(fname_sb)

    fname_sb, body = _offline_discrewabl(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(OFFLINE_DIR, fname_sb), body):
        created.append(fname_sb)

    # ── online ───────────────────────────────────────────────────────────────
    print("\n-- online --")
    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname_sb, body = _online(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(ONLINE_DIR, fname_sb), body):
            created.append(fname_sb)

    print(f"\n=== DONE: {len(created)} sbatches created ===")


if __name__ == "__main__":
    main()
