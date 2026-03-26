#!/bin/bash
#SBATCH --job-name=SAIL_PREF_RANK_HC
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/slurm_pref_rank_hc_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/slurm_pref_rank_hc_%A_%a.err
#SBATCH --partition=standard
#SBATCH --account=sabymath0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0

echo "========================================"
echo "JOB ID: $SLURM_JOB_ID"
echo "ARRAY ID: $SLURM_ARRAY_TASK_ID"
echo "NODE: $(hostname)"
echo "START: $(date)"
echo "========================================"

REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
LOG_DIR=${REPO_ROOT}/stable-baselines/run/logs
PYTHON_BIN=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python

mkdir -p "${LOG_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}

EXPERT_DATA=${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

cd "${REPO_ROOT}" || exit 1

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Repo: $REPO_ROOT"
echo "Expert data: $EXPERT_DATA"
echo "Pref RM: $PREF_RM"
echo "Seed: $SEED"

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --pref_rank_disc \
  --pref_rm "${PREF_RM}" \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --debug

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE