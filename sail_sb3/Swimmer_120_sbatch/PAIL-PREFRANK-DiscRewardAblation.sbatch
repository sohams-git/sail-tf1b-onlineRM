#!/bin/bash
#SBATCH --job-name=Swim_120_PAILPR_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/PAIL-PREFRANK-DiscRewardAblation_%A_%a.out
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
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_4.npz"
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/swimmer_pebble_1M/reward_model_1000000_0.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Swimmer_120_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  PAIL + PrefRank-Disc + Adaptive + LfD | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       adaptive_score_source: rm (RM-based promotion)"
echo "       pref_reweight_teacher: Boltzmann-weighted expert BCE"
echo "       pref_beta=50.0 (aggressive sharpening for tight Swimmer return dist)"
echo "       pref_rank_disc: BT ranking loss, weight=0.5, batch=32"
echo "       pref_expect_obs_dim=8 (Swimmer RM trained on native 8-dim obs)"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Swimmer"
export WANDB_NAME="Swimmer_120_PAIL-PREFRANK_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Swimmer-v2_120_pail_prefrank_discrew_${DISC_REWARD_TYPE}"
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
  --adaptive_score_source gt \
  --pref_reweight_teacher \
  --pref_beta 50.0 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "${PREF_RM}" \
  --pref_expect_obs_dim 8 \
  --pref_rank_disc \
  --pref_rank_weight 0.5 \
  --pref_rank_batch_size 32 \
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
