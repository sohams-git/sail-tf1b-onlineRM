#!/bin/bash
#SBATCH --job-name=Swim_120_ORM_SAILPR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Swimmer_120_sbatch/logs/SAIL-PREFRANK_%A_%a.out
#SBATCH --array=1-5
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3_online/Swimmer_120_sbatch/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  Online RM + SAIL + PrefRank-Disc + Adaptive SAIL + LfD | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       pref_rank_disc: BT ranking loss, weight=0.05, batch=32"
echo "       Online RM obs_dim=9 (TFW-wrapped) — no pref_expect_obs_dim needed"
echo "       RM activation: segs>=500 | updates>=50 | held_out_acc>=0.60"
echo "       RM rescore: every 20000 env steps"
echo "       entcoeff=0.05"
echo "Seed:  ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_PROJECT="SAIL_SB3_OnlineRM_Swimmer"
export WANDB_NAME="Swimmer_120_ORM_SAIL-PREFRANK_s${SEED}"
export WANDB_GROUP="Swimmer-v2_120_online_sail_prefrank"
export WANDB_SILENT=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env Swimmer-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 1000000 \
  --learning_starts 10000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_rank_disc \
  --pref_rank_weight 0.05 \
  --pref_rank_batch_size 32 \
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
