#!/bin/bash
#SBATCH --job-name=HC_OnlineRM_Smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/logs/HC_OnlineRM_Smoke30k_%j.out
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
SEED=1

cd "${REPO_ROOT}" || exit 1

echo "========================================"
echo "START: $(date)"
echo "SMOKE TEST: Online RM pipeline, 30k steps, seed=${SEED}"
echo "Checks: segmentation, RM training loop, gate status, no crashes"
echo "Expected: is_active=0 throughout (gates need ~50k+ to open)"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_DISABLED=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 30000 \
  --learning_starts 5000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_reweight_teacher \
  --pref_beta 1.0 \
  --pref_max_teacher_trajs 500 \
  --online_rm \
  --rm_segment_len 50 \
  --rm_train_freq 1000 \
  --rm_gradient_steps 10 \
  --rm_batch_size 256 \
  --rm_lr 3e-4 \
  --rm_max_segments 10000 \
  --rm_min_segments 500 \
  --rm_min_updates 50 \
  --rm_min_acc 0.60 \
  --rm_tie_margin 0.5 \
  --rm_rescore_freq 20000 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu \
  --debug

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
