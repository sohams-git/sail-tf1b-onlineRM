#!/bin/bash
#SBATCH --job-name=gen_ant_ep40
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/gen_ant_ep40_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/stable-baselines/run/logs/gen_ant_ep40_%j.err
#SBATCH --account=sabymath0

set -euo pipefail
REPO=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python

export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO"
"$PYTHON" -u stable-baselines/scripts/generate_sail_dataset_from_checkpoint.py \
  --checkpoint_dir teacher_training/ant_100ep_ckpt20/ant/teacher_checkpoints/ep0040_return1089 \
  --env_id Ant-v2 \
  --n_trajs 4 \
  --save_dir teacher_dataset_Ant

echo "DONE"
