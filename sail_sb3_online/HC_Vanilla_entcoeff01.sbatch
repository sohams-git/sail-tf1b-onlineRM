#!/bin/bash
#SBATCH --job-name=HC_V_ec01
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=sail_sb3/logs/HC_Vanilla_ec01_%A_%a.out
#SBATCH --array=1-3
#SBATCH --account=sabymath0

# Vanilla SAIL + entcoeff=0.1
# Hypothesis: entcoeff=0.05 succeeds 2/3 seeds but fails 1/3 via early disc saturation
# entcoeff=0.1 should make entropy regularization strong enough for 3/3 seeds
# Expected: disc_loss stabilizes > 0.4 across all seeds, ep_rew > 5000 by 300k

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL Vanilla | entcoeff=0.1"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_HalfCheetah"
export WANDB_NAME="SAIL_Vanilla_ec01_HC_s${SEED}"
export WANDB_GROUP="HalfCheetah-v2_vanilla_entcoeff01"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --entcoeff 0.1 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
