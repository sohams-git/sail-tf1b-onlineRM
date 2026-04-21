#!/bin/bash
#SBATCH --job-name=HC_QPREF_Guard
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=sail_sb3/logs/HC_QPREF_Guard_smoke_%A_%a.out
#SBATCH --array=1-3
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
PREF_RM="/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

echo "========================================"
echo "START: $(date)"
echo "Mode:  QPREF guard SMOKE TEST (60k steps)"
echo "       Verifies: code runs, new metrics log, no crashes"
echo "       FIXED: guard only fires when deque is full (50 entries = 50k steps)"
echo "       FIXED: qpref_delta_rolling_mean persists after disable"
echo "       FIXED: guard uses recent 10-entry window (not diluted 50-entry mean)"
echo "       FIXED: guard requires peak_guard_mean>=1.0 before triggering (positive phase gate)"
echo "       qpref_guard_threshold=-0.3  qpref_guard_confirm=3  qpref_guard_window=10  qpref_guard_positive_threshold=1.0"
echo "       qpref_source=student  qpref_weight=0.05  qpref_grad_interval=50"
echo "       adaptive=gt  lfd_mixing=True  entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_QPREF_Guard_smoke_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_qpref_guard_smoke"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --pref_rm "${PREF_RM}" \
  --qpref \
  --qpref_weight 0.05 \
  --qpref_temp 1.0 \
  --qpref_batch_size 4 \
  --qpref_start_step 0 \
  --qpref_source student \
  --qpref_grad_interval 50 \
  --qpref_guard_threshold -0.3 \
  --qpref_guard_confirm 3 \
  --qpref_guard_window 10 \
  --qpref_guard_positive_threshold 1.0 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
