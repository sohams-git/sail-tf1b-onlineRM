#!/bin/bash
#SBATCH --job-name=gen_datasets
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=1-6
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/gen_datasets_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/gen_datasets_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail

REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
SCRIPT="${REPO}/stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO}"

case $SLURM_ARRAY_TASK_ID in
  1)
    CKPT="teacher_training/ant/teacher_checkpoints/ep0200_return2858"
    ENV="Ant-v2"; N=4; SAVE="teacher_dataset_Ant"
    ;;
  2)
    CKPT="teacher_training/ant/teacher_checkpoints/ep1000_return3653"
    ENV="Ant-v2"; N=4; SAVE="teacher_dataset_Ant"
    ;;
  3)
    CKPT="teacher_training/hopper/teacher_checkpoints/ep0900_return3061"
    ENV="Hopper-v2"; N=10; SAVE="teacher_dataset_Hopper"
    ;;
  4)
    CKPT="teacher_training/halfcheetah/teacher_checkpoints/ep0200_return3335"
    ENV="HalfCheetah-v2"; N=4; SAVE="teacher_dataset_HC"
    ;;
  5)
    CKPT="teacher_training/walker2d/teacher_checkpoints/ep0100_return2598"
    ENV="Walker2d-v2"; N=10; SAVE="teacher_dataset_Walker2d"
    ;;
  6)
    CKPT="teacher_training/walker2d/teacher_checkpoints/ep0300_return3891"
    ENV="Walker2d-v2"; N=10; SAVE="teacher_dataset_Walker2d"
    ;;
esac

echo "TASK ${SLURM_ARRAY_TASK_ID}: ${ENV} | ${CKPT} | n_trajs=${N} -> ${SAVE}"
date

"${PYTHON}" -u "${SCRIPT}" \
  --checkpoint_dir "${CKPT}" \
  --env_id "${ENV}" \
  --n_trajs "${N}" \
  --save_dir "${SAVE}"

echo "DONE: task ${SLURM_ARRAY_TASK_ID}"
date
