#!/bin/bash
#SBATCH --job-name=sail_sb3_hc_v2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=sabymath0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/logs/slurm_%A.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/logs/slurm_%A.err

set -euo pipefail

echo "======================================="
echo "SAIL SB3 PyTorch Run - HalfCheetah-v2"
date
hostname
echo "JOB_ID = $SLURM_JOB_ID"
echo "======================================="

# --------------------------------------------------
# Paths
# --------------------------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
SB3_DIR=${REPO_ROOT}/sail_sb3

# Ensure we have a logs folder for SLURM output
mkdir -p "${SB3_DIR}/logs"

# --------------------------------------------------
# MuJoCo Headless Environment Variables
# --------------------------------------------------
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# --------------------------------------------------
# Execution Config
# --------------------------------------------------
ENV_ID="HalfCheetah-v2"
STEPS=1000000
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"

echo "Using Dataset: ${EXPERT_DATA}"

# --------------------------------------------------
# Activate Conda Env & Execute
# --------------------------------------------------
# Safely source conda using conda.sh pattern standard on Great Lakes
source $HOME/miniconda3/etc/profile.d/conda.sh || true
conda activate sail_sb3_env

# pyenv can shadow 'python' even inside a conda env; use the explicit path
PYTHON="${CONDA_PREFIX}/bin/python"
echo "Using Python: ${PYTHON}"
"${PYTHON}" --version

cd "${SB3_DIR}"

export PYTHONPATH="${REPO_ROOT}"

set -x
"${PYTHON}" scripts/train_sail.py \
  --env "${ENV_ID}" \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps "${STEPS}"
set +x

echo "======================================="
echo "Training Finished for SB3 SAIL on ${ENV_ID}!"
date
echo "======================================="
