#!/bin/bash
#SBATCH --job-name=ant_teacher_100ep
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ant_100ep_ckpt20_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/teacher_ant_100ep_ckpt20_%j.err
#SBATCH --account=sabymath0

# SHORT teacher run — 100 episodes, checkpoint every 20 episodes.
# Output dir: teacher_training/ant_100ep_ckpt20/ant/
# Does NOT overwrite teacher_training/ant/ (different save_root).

set -euo pipefail
echo "== Ant-v2 Teacher Training  (100ep, ckpt every 20) =="
date
hostname
echo "JOB=$SLURM_JOB_ID"

REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
SAVE_ROOT="${REPO}/teacher_training/ant_100ep_ckpt20"

ENV_ID="Ant-v2"
TOTAL_EPISODES=100
CHECKPOINT_EVERY=20
N_EVAL_EPISODES=10
TARGET_RETURN=3000
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
echo "  SAVE_ROOT        = ${SAVE_ROOT}"

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
ls -lh "${SAVE_ROOT}/ant/teacher_checkpoints/" 2>/dev/null || true
cat "${SAVE_ROOT}/ant/teacher_logs/training_log.csv" 2>/dev/null || true
date
echo "DONE."
