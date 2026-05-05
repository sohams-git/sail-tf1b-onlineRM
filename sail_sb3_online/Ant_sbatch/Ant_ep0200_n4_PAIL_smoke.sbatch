#!/bin/bash
#SBATCH --job-name=Ant_ep0200_PAIL_smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Ant_sbatch/logs/Ant_ep0200_n4_PAIL_smoke_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Ant_sbatch/logs/Ant_ep0200_n4_PAIL_smoke_%j.err
#SBATCH --account=sabymath0

# SMOKE TEST — 10k steps, seed 0 only. Do NOT use for real runs.
# Dataset: teacher_dataset_Ant/expert_data_no_img_Ant_scores_2878_episodes_4_ep200.npz

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Ant/expert_data_no_img_Ant_scores_2878_episodes_4_ep200.npz"

mkdir -p "${REPO_ROOT}/sail_sb3_online/Ant_sbatch/logs"

echo "========================================"
echo "SMOKE TEST: Ant ep0200 n4 PAIL (sail_sb3_online)"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"${PYTHON_BIN}" sail_sb3_online/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 10000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_reweight_teacher \
  --pref_beta 5.0 \
  --pref_max_teacher_trajs 500 \
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
  --seed 0 \
  --device cpu \
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
