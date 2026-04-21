#!/bin/bash
#SBATCH --job-name=HC_RMRw_test
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=sail_sb3/logs/HC_RMReweight_Adapt_LfD_test_%A_%a.out
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
echo "Mode:  RM-Adaptive SAIL + PrefReweight + LfD [SMOKE TEST 100k]"
echo "       adaptive_score_source=rm: promotes based on RM score, no GT leakage"
echo "       ring buffer: teacher_buffer_size=1000 (TF parity)"
echo "       pref pool: ALL 4 expert episodes (built before ring truncation)"
echo "       pref_reweight_teacher: Boltzmann-weighted disc expert batch (beta=1.0)"
echo "       disc fires every 500 env steps"
echo "       entcoeff=0.01"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_RMReweight_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_rmreweight_adapt_test"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source rm \
  --pref_reweight_teacher \
  --pref_beta 1.0 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "${PREF_RM}" \
  --entcoeff 0.01 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
