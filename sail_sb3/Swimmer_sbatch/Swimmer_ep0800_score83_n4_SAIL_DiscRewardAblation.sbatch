#!/bin/bash
#SBATCH --job-name=Swim_ep0800_SAIL_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Swimmer_sbatch/logs/SAIL-DiscRewardAblation_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

# ============================================================
# Discriminator reward ablation: gail_js
# ============================================================
DISC_REWARD_TYPE="gail_js"

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_83_episodes_4_ep800.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Swimmer_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL + DiscRewardAblation + Adaptive + LfD | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       dataset: ep0800 score83 n4"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Swimmer"
export WANDB_NAME="Swimmer_ep0800sc83_SAIL_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Swimmer-v2_ep0800_sail_discrew_${DISC_REWARD_TYPE}"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env Swimmer-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
