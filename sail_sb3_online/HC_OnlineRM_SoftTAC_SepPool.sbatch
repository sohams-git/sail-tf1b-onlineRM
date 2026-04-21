#!/bin/bash
#SBATCH --job-name=HC_ORM_STAC_SP
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/logs/HC_OnlineRM_SoftTAC_SepPool_%A_%a.out
#SBATCH --array=1-3
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
echo "Mode:  Online RM + Soft-TAC [SepPool fix] + Adaptive SAIL + LfD"
echo "       Base: HC_OnlineRM_SoftTAC_Adapt_LfD (identical augmentation)"
echo "       Fix: Soft-TAC now uses dedicated soft_tac_pool instead of pref_episodes"
echo "         soft_tac_pool = 4 expert eps GT-scored (permanent) + all students (recency-200)"
echo "         Pool is diverse from step ~1k — no 300k wait for first promotion"
echo "       NO pref_rank_disc"
echo "       y = sign(J_pos - J_neg): GT J before activation, RM J after rescore"
echo "       Activation: Gate1 >=500 segs | Gate2 >=50 updates | Gate3 acc>=0.60"
echo "       Rescore: every 20000 env steps"
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
export WANDB_NAME="OnlineRM_SoftTAC_SepPool_v2_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_online_rm_softtac_seppool_v2"
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
