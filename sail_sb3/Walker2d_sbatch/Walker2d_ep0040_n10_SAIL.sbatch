#!/bin/bash
#SBATCH --job-name=W2d_ep0040_n10_SAIL
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Walker2d_sbatch/logs/Walker2d_ep0040_n10_SAIL_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

# FULL RUN — SAIL + Adaptive + LfD | Walker2d-v2
# Dataset: teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_502_episodes_10_ep40.npz

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_502_episodes_10_ep40.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${REPO_ROOT}/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="Walker2d_ep0040_n10_SAIL_s${SEED}"
export WANDB_GROUP="Walker2d-v2_ep0040_sail"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON_BIN}" sail_sb3/scripts/train_sail.py \
  --env Walker2d-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu \
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
