#!/bin/bash
#SBATCH --job-name=HC_ORM_STAC_RMJ
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/logs/HC_OnlineRM_SoftTAC_RMJ_%A_%a.out
#SBATCH --array=1-2
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3_online/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  Online RM + Soft-TAC [RM-only J] + Adaptive SAIL + LfD"
echo "       Soft-TAC J source: online RM only (no GT labels)"
echo "         Expert J: GT placeholder at init, overwritten by RM on first rescore"
echo "         Student J: GT placeholder at insertion, overwritten by RM on first rescore"
echo "         Soft-TAC gate: is_active AND _soft_tac_pool_rescored"
echo "         Rescore: every 20000 env steps (same trigger as pref_episodes)"
echo "         First activation: ~300-420k steps (after RM gates open + first rescore)"
echo "       soft_tac_pool = 4 expert eps + all students (recency-200)"
echo "       NO pref_rank_disc"
echo "       Activation gates: segs>=500 | updates>=50 | held_out_acc>=0.60"
echo "       soft_tac: weight=0.5  temp=1.0  tie_eps=0.1  max_student_trajs=200"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_OnlineRM_HalfCheetah"
export WANDB_NAME="OnlineRM_SoftTAC_RMJ_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_online_rm_softtac_rmj_v1"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_max_teacher_trajs 500 \
  --soft_tac \
  --soft_tac_weight 0.5 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.1 \
  --soft_tac_max_student_trajs 200 \
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
