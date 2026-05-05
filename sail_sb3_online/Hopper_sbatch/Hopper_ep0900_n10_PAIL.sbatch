#!/bin/bash
#SBATCH --job-name=Hop_ep0900_n10_PAIL
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Hopper_sbatch/logs/Hopper_ep0900_n10_PAIL_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

# FULL RUN — Online RM + PAIL + Adaptive + LfD | Hopper-v2
# Dataset: teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_3083_episodes_10_ep900.npz
# Checkpoint: ep0900_return3061 (stronger teacher)

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_3083_episodes_10_ep900.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${REPO_ROOT}/sail_sb3_online/Hopper_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_OnlineRM_Hopper"
export WANDB_NAME="Hop_ep0900_n10_PAIL_s${SEED}"
export WANDB_GROUP="Hopper-v2_ep0900_online_pail"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"${PYTHON_BIN}" sail_sb3_online/scripts/train_sail.py \
  --env Hopper-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_reweight_teacher \
  --pref_beta 1.0 \
  --pref_max_teacher_trajs 500 \
  --online_rm \
  --rm_segment_len 50 \
  --rm_train_freq 1000 \
  --rm_gradient_steps 10 \
  --rm_batch_size 256 \
  --rm_lr 3e-4 \
  --rm_max_segments 10000 \
  --rm_min_segments 500 \
  --rm_min_updates 50 \
  --rm_min_acc 0.60 \
  --rm_tie_margin 0.5 \
  --rm_rescore_freq 20000 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu \
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
