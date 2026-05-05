#!/bin/bash
#SBATCH --job-name=Ant_ep0200s2878_SAILPD_DiscRew
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_sbatch/logs/Ant_ep0200_score2878_n4_SAIL_PREFD_DiscRewAbl_%A_%a.out
#SBATCH --array=1-3
#SBATCH --account=sabymath0

# Discriminator reward ablation — SAIL + PREF-D-Disc (Ant-v2)
# DISC_REWARD_TYPE options: gail_js | airl_backward_kl | fairl_forward_kl | gail_heuristic
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset_Ant/expert_data_no_img_Ant_scores_2878_episodes_4_ep200.npz"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${SLURM_JOB_ID}  SEED: ${SEED}"
echo "Mode: SAIL + PREF-D-Disc + Adaptive + LfD  [disc reward ablation] | Ant-v2"
echo "disc_reward_type: ${DISC_REWARD_TYPE}"
echo "Dataset: ${EXPERT_DATA}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_ep0200_score2878_n4_SAIL_PREFD_discrew_${DISC_REWARD_TYPE}_s${SEED}"
export WANDB_GROUP="Ant-v2_ep0200_s2878_sail_prefd_discrew_${DISC_REWARD_TYPE}"
export WANDB_SILENT=true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" \
"/home/sohams/miniconda3/envs/sail_sb3_env/bin/python" sail_sb3/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --pref_max_teacher_trajs 500 \
  --pref_rm "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt" \
  --pref_expect_obs_dim 111 \
  --entcoeff 0.05 \
  --disc_reward_type "${DISC_REWARD_TYPE}" \
  --seed "${SEED}" \
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${EXIT_CODE}"; echo "========================================"
exit $EXIT_CODE
