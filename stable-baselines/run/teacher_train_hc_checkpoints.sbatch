#!/bin/bash
#SBATCH --job-name=hc_teacher_ckpt
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ckpt_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ckpt_%j.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== HalfCheetah Teacher Training  (SB3 TD3, isolated) =="
date
hostname
echo "JOB=$SLURM_JOB_ID"

# ============================================================
# Configuration
# ============================================================
REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM

# Isolated Python — SB3 (PyTorch).  Zero TF1 / sail_sb3 dependencies.
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python

# All teacher training outputs stay inside this repo
SAVE_ROOT="${REPO}/teacher_training"

ENV_ID="HalfCheetah-v2"
TOTAL_EPISODES=2000          # 2000 ep × 1000 steps = 2M steps
CHECKPOINT_EVERY=100         # checkpoint every 100 episodes
N_EVAL_EPISODES=10           # deterministic eval rollouts at each checkpoint
TARGET_RETURN=6988           # special ckpt: first time mean return >= this
SEED=0
# ============================================================

mkdir -p "${REPO}/stable-baselines/run/logs" "${SAVE_ROOT}"

# ---- Headless MuJoCo (CPU) ----
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Config:"
echo "  REPO             = ${REPO}"
echo "  PYTHON           = ${PYTHON}"
echo "  SAVE_ROOT        = ${SAVE_ROOT}"
echo "  TOTAL_EPISODES   = ${TOTAL_EPISODES}"
echo "  CHECKPOINT_EVERY = ${CHECKPOINT_EVERY}"
echo "  N_EVAL_EPISODES  = ${N_EVAL_EPISODES}"
echo "  TARGET_RETURN    = ${TARGET_RETURN}"
echo "  SEED             = ${SEED}"

cd "${REPO}"

PYTHONHASHSEED="${SEED}" "${PYTHON}" -u \
    stable-baselines/scripts/train_halfcheetah_teacher_checkpoints.py \
    --save_root        "${SAVE_ROOT}" \
    --env_id           "${ENV_ID}" \
    --total_episodes   "${TOTAL_EPISODES}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    --n_eval_episodes  "${N_EVAL_EPISODES}" \
    --target_return    "${TARGET_RETURN}" \
    --seed             "${SEED}"

echo ""
echo "== Training complete =="
echo "Checkpoints:"
ls -lh "${SAVE_ROOT}/halfcheetah/teacher_checkpoints/" 2>/dev/null || true
echo ""
echo "Training log:"
cat "${SAVE_ROOT}/halfcheetah/teacher_logs/training_log.csv" 2>/dev/null || true

date
echo "DONE."
