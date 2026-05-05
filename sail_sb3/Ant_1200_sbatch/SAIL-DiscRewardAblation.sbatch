#!/bin/bash
#SBATCH --job-name=Ant_1200_SAIL_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_1200_sbatch/logs/SAIL-DiscRewardAblation_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -e

# ============================================================
# Discriminator reward ablation
# Switch DISC_REWARD_TYPE to test different formulations:
#   gail_js          (default — softplus(logits))
#   airl_backward_kl (logits)
#   fairl_forward_kl (-logits * exp(logits), clamped)
#   gail_heuristic   (-softplus(-logits))
# ============================================================
DISC_REWARD_TYPE="gail_js"

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Ant_1200_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL (base) + Adaptive + LfD | Ant-v2"
echo "       ring buffer: teacher_buffer_size=1000"
echo "       adaptive_score_source: gt (default)"
echo "       expert threshold: 1215.9 (episode 4, ring-truncated)"
echo "       disc fires every 500 env steps"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_1200_SAIL_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Ant-v2_1200_sail_discrew_${DISC_REWARD_TYPE}"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --entcoeff 0.05 \
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
