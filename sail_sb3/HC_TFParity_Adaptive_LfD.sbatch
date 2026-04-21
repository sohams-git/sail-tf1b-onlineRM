#!/bin/bash
#SBATCH --job-name=HC_TFP_A_LfD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=sail_sb3/HC_sbatch_logs/HC_TFParity_Adaptive_LfD_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

echo "========================================"
echo "START: $(date)"
echo "Mode:  TF-Parity Adaptive SAIL + LfD mixing"
echo "       ring buffer: teacher_buffer_size=1000 (TF demo_buffer_size=1000)"
echo "       expert_scores: insertion order (TF parity, threshold=first-loaded episode)"
echo "       disc fires every 500 env steps (2x per policy update)"
echo "       entcoeff=0.01 (TF default)"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_TFParity_Adaptive_LfD_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_tfparity_adaptive_lfd"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.01 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
