#!/bin/bash
#SBATCH --job-name=W2d_1500_ORM_PAILPR_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
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
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Walker2d_scores_1500_episodes_10.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3_online/Walker2d_1500_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  Online RM + PAIL + PrefRank-Disc + Adaptive SAIL + LfD | Walker2d-v2"
echo "       ring buffer: teacher_buffer_size=1000 (~2-3 episodes)"
echo "       pref_reweight_teacher: Boltzmann-weighted expert BCE (beta=1.0)"
echo "       pref_rank_disc: BT ranking loss, weight=0.1, batch=32"
echo "       Online RM obs_dim=18 (TFW-wrapped) — no pref_expect_obs_dim needed"
echo "       RM activation: segs>=500 | updates>=50 | held_out_acc>=0.60"
echo "       RM rescore: every 20000 env steps"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_OnlineRM_Walker2d"
export WANDB_NAME="Walker2d_1500_ORM_PAIL-PREFRANK_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Walker2d-v2_1500_online_pail_prefrank_discrew_${DISC_REWARD_TYPE}"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env Walker2d-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_reweight_teacher \
  --pref_beta 1.0 \
  --pref_max_teacher_trajs 500 \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 32 \
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
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --seed "${SEED}" \
  --device cpu \
  --debug

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
