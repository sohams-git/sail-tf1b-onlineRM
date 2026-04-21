#!/bin/bash
#SBATCH --job-name=swimmer_teacher_ckpt
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ckpt_swimmer_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ckpt_swimmer_%j.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== Swimmer-v2 Teacher Training  (SB3 TD3, isolated) =="
date
hostname
echo "JOB=$SLURM_JOB_ID"

REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
SAVE_ROOT="${REPO}/teacher_training"

ENV_ID="Swimmer-v2"
TOTAL_EPISODES=2000
CHECKPOINT_EVERY=100
N_EVAL_EPISODES=10
TARGET_RETURN=200
SEED=0

mkdir -p "${REPO}/stable-baselines/run/logs" "${SAVE_ROOT}"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Config:"
echo "  ENV_ID           = ${ENV_ID}"
echo "  TOTAL_EPISODES   = ${TOTAL_EPISODES}"
echo "  CHECKPOINT_EVERY = ${CHECKPOINT_EVERY}"
echo "  TARGET_RETURN    = ${TARGET_RETURN}"
echo "  SEED             = ${SEED}"

cd "${REPO}"

PYTHONHASHSEED="${SEED}" "${PYTHON}" -u \
    stable-baselines/scripts/train_teacher_checkpoints.py \
    --save_root        "${SAVE_ROOT}" \
    --env_id           "${ENV_ID}" \
    --total_episodes   "${TOTAL_EPISODES}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    --n_eval_episodes  "${N_EVAL_EPISODES}" \
    --target_return    "${TARGET_RETURN}" \
    --seed             "${SEED}"

echo ""
echo "== Training complete =="
ls -lh "${SAVE_ROOT}/swimmer/teacher_checkpoints/" 2>/dev/null || true
cat "${SAVE_ROOT}/swimmer/teacher_logs/training_log.csv" 2>/dev/null || true
date
echo "DONE."
