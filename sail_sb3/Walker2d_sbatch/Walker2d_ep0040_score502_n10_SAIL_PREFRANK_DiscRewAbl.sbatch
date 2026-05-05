#!/bin/bash
#SBATCH --job-name=W2d_ep0040s502_SAILPD_DiscRew
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Walker2d_sbatch/logs/Walker2d_ep0040_score502_n10_SAIL_PREFRANK_DiscRewAbl_%A_%a.out
#SBATCH --array=1-3
#SBATCH --account=sabymath0

# Discriminator reward ablation — SAIL + PREFRANK-Disc (Walker2d-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_502_episodes_10_ep40.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${REPO_ROOT}/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Mode: SAIL + PREFRANK-Disc + Adaptive + LfD  [disc reward ablation] | Walker2d-v2"
echo "disc_reward_type: ${DISC_REWARD_TYPE}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="Walker2d_ep0040_score502_n10_SAIL_PREFRANK_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Walker2d-v2_ep0040_s502_sail_prefrank_discrew_${DISC_REWARD_TYPE}"
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
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 32 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/walker2d_pebble_1M/reward_model_1000000_0.pt" \
  --pref_expect_obs_dim 17 \
  --entcoeff 0.05 \
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
