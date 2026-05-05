#!/bin/bash
#SBATCH --job-name=swimmer_retrain345
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/swimmer_retrain345_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/swimmer_retrain345_%j.err
#SBATCH --account=sabymath0

# Swimmer-v2 teacher RETRAIN — fixes the original failed run.
#
# Original run (job 48290417) was stuck at -17 to -21 for all 2M steps.
# Root cause: learning_starts=100 (SB3 default) is too low for Swimmer-v2's
# tiny per-step reward scale (~0.009/step at zero-action policy).  The
# Q-network got bad early estimates and the policy locked into backward swimming.
#
# Fixes: learning_starts=10000, action_noise_sigma=0.3, lr=3e-4, seed=42.
# Budget: 3M steps (3000 episodes * 1000 steps), targeting ~345 return.
# Output: teacher_training/swimmer_retrain_target345/  (old checkpoints untouched)

set -euo pipefail

REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
SAVE_ROOT="${REPO}/teacher_training/swimmer_retrain_target345"

echo "== Swimmer-v2 Teacher RETRAIN (target ~345) =="
date
hostname
echo "JOB=${SLURM_JOB_ID}"
echo ""
echo "Config:"
echo "  save_root        = ${SAVE_ROOT}"
echo "  total_episodes   = 3000  (3,000,000 steps)"
echo "  learning_starts  = 10000"
echo "  action_noise_sig = 0.3"
echo "  learning_rate    = 3e-4"
echo "  target_return    = 345"
echo "  seed             = 42"

mkdir -p "${REPO}/stable-baselines/run/logs" "${SAVE_ROOT}"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO}"

PYTHONHASHSEED=42 "${PYTHON}" -u \
    stable-baselines/scripts/retrain_swimmer_teacher.py \
    --save_root          "${SAVE_ROOT}" \
    --env_id             Swimmer-v2 \
    --total_episodes     3000 \
    --checkpoint_every   100 \
    --n_eval_episodes    10 \
    --target_return      345 \
    --seed               42 \
    --learning_starts    10000 \
    --action_noise_sigma 0.3 \
    --learning_rate      3e-4

EXIT_CODE=$?

echo ""
echo "== Training complete =="
echo "Checkpoints:"
ls -lh "${SAVE_ROOT}/swimmer/teacher_checkpoints/" 2>/dev/null || true
echo ""
echo "Training log (last 5 rows):"
tail -5 "${SAVE_ROOT}/swimmer/teacher_logs/training_log.csv" 2>/dev/null || true
date
echo "EXIT CODE: ${EXIT_CODE}"
echo "DONE."
exit $EXIT_CODE
