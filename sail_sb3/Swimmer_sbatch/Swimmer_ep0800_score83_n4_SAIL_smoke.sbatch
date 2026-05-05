#!/bin/bash
#SBATCH --job-name=Swim_ep0800_SAIL_smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Swimmer_sbatch/logs/Swimmer_ep0800_SAIL_smoke_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Swimmer_sbatch/logs/Swimmer_ep0800_SAIL_smoke_%j.err
#SBATCH --account=sabymath0

# SMOKE TEST — 10k steps, seed 0 only. Do NOT use for real runs.
# Dataset: teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_83_episodes_4_ep800.npz

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_83_episodes_4_ep800.npz"

mkdir -p "${REPO_ROOT}/sail_sb3/Swimmer_sbatch/logs"

echo "========================================"
echo "SMOKE TEST: Swimmer ep0800 n4 SAIL (sail_sb3)"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON_BIN}" sail_sb3/scripts/train_sail.py \
  --env Swimmer-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 10000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.05 \
  --seed 0 \
  --device cpu \
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
