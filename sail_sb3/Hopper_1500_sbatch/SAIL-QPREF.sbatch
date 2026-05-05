#!/bin/bash
#SBATCH --job-name=Hop_1500_SAILQP
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/SAIL-QPREF_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_10.npz"
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Hopper_1500_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL + QPREF (student source) + Adaptive + LfD | Hopper-v2"
echo "       ring buffer: teacher_buffer_size=1000 (~2-3 episodes)"
echo "       adaptive_score_source: gt (GT-based promotion)"
echo "       qpref: source=student  weight=0.03  grad_interval=50"
echo "       qpref guard: threshold=-0.3  confirm=3  window=10  pos_threshold=1.0"
echo "       student pool: pref_max_student_trajs=500"
echo "       pref_expect_obs_dim=11 (Hopper RM trained on native 11-dim obs)"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Hopper"
export WANDB_NAME="Hopper_1500_SAIL-QPREF_s${SEED}"
export WANDB_GROUP="Hopper-v2_1500_sail_qpref"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env Hopper-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --qpref \
  --qpref_weight 0.03 \
  --qpref_source student \
  --qpref_grad_interval 50 \
  --qpref_guard_threshold -0.3 \
  --qpref_guard_confirm 3 \
  --qpref_guard_window 10 \
  --qpref_guard_positive_threshold 1.0 \
  --pref_max_student_trajs 500 \
  --pref_rm "${PREF_RM}" \
  --pref_expect_obs_dim 11 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
