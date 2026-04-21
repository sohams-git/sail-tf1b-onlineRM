#!/bin/bash
#SBATCH --job-name=HC_smoke_ep1000_ALfD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/HC_sbatch_smoke_logs/HC_smoke_ep1000_ALfD_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/HC_sbatch_smoke_logs/HC_smoke_ep1000_ALfD_%j.err
#SBATCH --account=sabymath0

# SMOKE TEST — do NOT use for real runs.
# Dataset: teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_9099_episodes_4_ep1000.npz
# Checkpoint: ep1000_return9094 (mean_return=9094.16)
# Budget: 10k steps (not a real training run)
# Success criteria: dataset loads, expert returns printed, training begins past learning_starts

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_9099_episodes_4_ep1000.npz"

echo "========================================"
echo "SMOKE TEST: HC ep1000 Adaptive SAIL + LfD"
echo "START: $(date)"
echo "JOB: ${SLURM_JOB_ID}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}" || exit 1

PYTHONPATH="${REPO_ROOT}" \
"${PYTHON_BIN}" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 10000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.01 \
  --seed 0 \
  --device cpu \
  --debug

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: ${EXIT_CODE}"
echo "========================================"

exit $EXIT_CODE
