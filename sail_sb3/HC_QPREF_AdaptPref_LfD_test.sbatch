#!/bin/bash
#SBATCH --job-name=HC_QPREF_AP
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=sail_sb3/logs/HC_QPREF_AdaptPref_LfD_%A_%a.out
#SBATCH --array=1-2
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
echo "Mode:  QPREF + GT-Adaptive SAIL + PrefRank + LfD mixing [SMOKE TEST]"
echo "       qpref: trajectory-mean-Q Bradley-Terry loss on critic"
echo "       qpref_source=teacher: uses pref_episodes (all 4 expert eps)"
echo "       qpref_weight=0.1  qpref_temp=1.0  qpref_batch_size=4"
echo "       qpref_grad_interval=10: fires every 10th gradient step"
echo "       adaptive_score_source=gt (GT-based promotion)"
echo "       pref_rank_disc: preference ranking loss on discriminator"
echo "       ring buffer: teacher_buffer_size=1000 (TF parity)"
echo "       pref pool: ALL 4 expert episodes (SepPool, built before ring truncation)"
echo "       entcoeff=0.05 (confirmed working; avoids saturation)"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_QPREF_AdaptPref_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_qpref_adaptpref_smoke"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 100000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --pref_rm "${PREF_RM}" \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --qpref \
  --qpref_weight 0.1 \
  --qpref_temp 1.0 \
  --qpref_batch_size 4 \
  --qpref_start_step 0 \
  --qpref_source teacher \
  --qpref_grad_interval 10 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
