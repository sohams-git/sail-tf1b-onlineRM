#!/bin/bash
#SBATCH --job-name=Ant_ep0200s2878_SAIL_QPREF
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_sbatch/logs/Ant_ep0200_score2878_n4_SAIL_QPREF_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Ant/expert_data_no_img_Ant_scores_2878_episodes_4_ep200.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Mode: SAIL + QPREF (student source) + Adaptive + LfD | Ant-v2"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_ep0200_score2878_n4_SAIL_QPREF_s${SEED}"
export WANDB_GROUP="Ant-v2_ep0200_s2878_sail_qpref"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" \
"/home/sohams/miniconda3/envs/sail_sb3_env/bin/python" sail_sb3/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --qpref \
  --qpref_weight 0.05 \
  --qpref_source student \
  --qpref_grad_interval 50 \
  --qpref_guard_threshold -0.3 \
  --qpref_guard_confirm 3 \
  --qpref_guard_window 10 \
  --qpref_guard_positive_threshold 1.0 \
  --pref_max_student_trajs 500 \
  --pref_rm "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt" \
  --pref_expect_obs_dim 111 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
