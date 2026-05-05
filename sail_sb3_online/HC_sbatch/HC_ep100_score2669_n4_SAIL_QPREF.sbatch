#!/bin/bash
#SBATCH --job-name=HC_ep100s2669_SAIL_QPREF
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/HC_sbatch/logs/HC_ep100_score2669_n4_SAIL_QPREF_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_2669_episodes_4_ep100.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${REPO_ROOT}/sail_sb3_online/HC_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Mode: Online RM + SAIL_QPREF + Adaptive + LfD | HalfCheetah-v2"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_OnlineRM_HalfCheetah"
export WANDB_NAME="HC_ep100_score2669_n4_SAIL_QPREF_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_ep100_sailqpref"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"${PYTHON_BIN}" sail_sb3_online/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_max_teacher_trajs 500 \
  --qpref \
  --qpref_source student \
  --qpref_weight 0.1 \
  --qpref_temp 1.0 \
  --qpref_batch_size 4 \
  --qpref_grad_interval 10 \
  --qpref_guard_threshold -0.3 \
  --qpref_guard_confirm 3 \
  --qpref_guard_window 10 \
  --qpref_guard_positive_threshold 1.0 \
  --pref_max_student_trajs 500 \
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
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
