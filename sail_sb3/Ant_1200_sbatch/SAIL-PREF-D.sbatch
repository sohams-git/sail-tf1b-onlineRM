#!/bin/bash
#SBATCH --job-name=Ant_1200_SAILPD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_1200_sbatch/logs/SAIL-PREF-D_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_4.npz"
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/Ant_1200_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  SAIL + PrefRank-Disc + Adaptive + LfD | Ant-v2"
echo "       ring buffer: teacher_buffer_size=1000"
echo "       adaptive_score_source: gt (default)"
echo "       pref_rank_disc: BT ranking loss, weight=0.1, batch=32"
echo "       pref_expect_obs_dim=111 (Ant RM trained on native 111-dim obs)"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_1200_SAIL-PREF-D_s${SEED}"
export WANDB_GROUP="Ant-v2_1200_sail_prefd"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 32 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "${PREF_RM}" \
  --pref_expect_obs_dim 111 \
  --entcoeff 0.05 \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
