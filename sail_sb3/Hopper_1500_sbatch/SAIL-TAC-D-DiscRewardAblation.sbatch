#!/bin/bash
#SBATCH --job-name=Hop_1500_SAILTD_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/SAIL-TAC-D-DiscRewardAblation_%A_%a.out
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
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_10.npz"
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Hopper_1500_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL + Soft-TAC-Disc (separate pool) + Adaptive + LfD | Hopper-v2"
echo "       ring buffer: teacher_buffer_size=1000 (~2-3 episodes)"
echo "       adaptive_score_source: gt (GT-based promotion)"
echo "       soft_tac: weight=0.03  temp=1.0  tie_eps=0.0  max_student=200"
echo "       pref_expect_obs_dim=11 (Hopper RM trained on native 11-dim obs)"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Hopper"
export WANDB_NAME="Hopper_1500_SAIL-TAC-D_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Hopper-v2_1500_sail_tacd_discrew_${DISC_REWARD_TYPE}"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env Hopper-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --pref_rm "${PREF_RM}" \
  --pref_expect_obs_dim 11 \
  --soft_tac \
  --soft_tac_weight 0.03 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.0 \
  --soft_tac_max_student_trajs 200 \
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
