#!/bin/bash
#SBATCH --job-name=HC_ORM_PrefTAC_Smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:45:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/logs/HC_OnlineRM_PrefRank_SoftTAC_Smoke30k_%j.out
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
SEED=1

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3_online/logs

echo "========================================"
echo "START: $(date)"
echo "SMOKE TEST: Online RM + pref_rank_disc + soft_tac, 30k steps, seed=${SEED}"
echo ""
echo "Checks:"
echo "  1. pool_size=4 at startup (GT J, pre-truncation)"
echo "  2. pref_rank_disc: pairs sampled, pref_rank/j_diff_mean > 0"
echo "  3. soft_tac: labels non-trivial (n_pos + n_neg > 0)"
echo "  4. No crashes, no shape/NaN errors"
echo "  5. After RM activation (~26k): src=rm on promoted episodes"
echo ""
echo "Expected: is_active=0 until ~26k, then flips to 1"
echo "Seed: ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_DISABLED=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 30000 \
  --learning_starts 5000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_reweight_teacher \
  --pref_beta 1.0 \
  --pref_max_teacher_trajs 500 \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --soft_tac \
  --soft_tac_weight 0.5 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.1 \
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

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
