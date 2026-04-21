#!/bin/bash
#SBATCH --job-name=HC_full_ep0400_ALfD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/logs/HC_full_ep0400_ALfD_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/logs/HC_full_ep0400_ALfD_%A_%a.err
#SBATCH --array=1-3
#SBATCH --account=sabymath0

# FULL RUN — Adaptive SAIL + LfD mixing
# Dataset: teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_7372_episodes_4_ep400.npz
# Checkpoint: ep0400_return6718 (mean_return=6718.47, sub-optimal teacher)
# Budget: 1M steps, 3 seeds

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_7372_episodes_4_ep400.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${REPO_ROOT}/sail_sb3/logs"

echo "========================================"
echo "FULL RUN: HC ep0400 Adaptive SAIL + LfD"
echo "START: $(date)"
echo "JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_HC_ep0400_AdaptiveLfD_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_ep0400_adaptive_lfd"
export WANDB_SILENT=true

cd "${REPO_ROOT}" || exit 1

PYTHONPATH="${REPO_ROOT}" \
"${PYTHON_BIN}" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.01 \
  --seed "${SEED}" \
  --device cpu \
  --debug

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: ${EXIT_CODE}"
echo "========================================"

exit $EXIT_CODE
