#!/bin/bash
#SBATCH --job-name=HC_6500_SAILPD_DiscRew
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/SAIL-PREF-D-DiscRewardAblation_%A_%a.out
#SBATCH --array=1-3
#SBATCH --account=sabymath0

set -e

# ============================================================
# Discriminator reward ablation — SAIL + PrefRank-Disc (HC)
#
# To switch reward type, edit DISC_REWARD_TYPE below:
#   gail_js          (default — softplus(logits), matches original behavior)
#   airl_backward_kl (logits)
#   fairl_forward_kl (-logits * exp(logits), clamped for stability)
#   gail_heuristic   (-softplus(-logits))
# ============================================================
DISC_REWARD_TYPE="airl_backward_kl"

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
PREF_RM="/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/HC_6500_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL + PrefRank-Disc + Adaptive + LfD  [disc reward ablation]"
echo "       ring buffer: teacher_buffer_size=1000"
echo "       adaptive_score_source: gt (default)"
echo "       pref_rank_disc: BT ranking loss, weight=0.1, batch=16"
echo "       disc fires every 500 env steps"
echo "       entcoeff=0.01"
echo "       disc_reward_type: ${DISC_REWARD_TYPE}"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="HC_6500_SAIL-PREF-D_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_6500_sail_prefd_discrew_${DISC_REWARD_TYPE}"
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
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "${PREF_RM}" \
  --entcoeff 0.01 \
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
