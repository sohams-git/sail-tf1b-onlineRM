#!/bin/bash
#SBATCH --job-name=W2d_ep0100s2768_SAIL_TACD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Walker2d_sbatch/logs/Walker2d_ep0100_score2768_n10_SAIL_TACD_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_2768_episodes_10_ep100.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Walker2d_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Mode: SAIL + Soft-TAC-Disc + Adaptive + LfD | Walker2d-v2"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Walker2d"
export WANDB_NAME="Walker2d_ep0100_score2768_n10_SAIL_TACD_s${SEED}"
export WANDB_GROUP="Walker2d-v2_ep0100_s2768_sail_tacd"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" \
"/home/sohams/miniconda3/envs/sail_sb3_env/bin/python" sail_sb3/scripts/train_sail.py \
  --env Walker2d-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --soft_tac \
  --soft_tac_weight 0.1 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.0 \
  --soft_tac_max_student_trajs 200 \
  --pref_rm "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/walker2d_pebble_1M/reward_model_1000000_0.pt" \
  --pref_expect_obs_dim 17 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
