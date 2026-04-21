#!/bin/bash
#SBATCH --job-name=Ant_ORM_PD_smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:45:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Ant_smoke/logs/SAIL-PREF-D_smoke_%A_%a.out
#SBATCH --array=1-2
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_4.npz"
SEED=$SLURM_ARRAY_TASK_ID

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3_online/Ant_smoke/logs

echo "========================================"
echo "START: $(date)"
echo "SMOKE TEST: Ant-v2 online RM + SAIL-PREF-D [50k steps]"
echo ""
echo "What this verifies:"
echo "  1. expert_loader reads Ant npz correctly (obs=112, act=8, 4 episodes)"
echo "  2. expert return stats printed: mean ~1148"
echo "  3. obs_dim=112 printed (TFW applied)"
echo "  4. online RM segment collection: rm/n_segs_total grows each ~1k steps"
echo "  5. online RM trains: rm/held_out_acc rises from 0.5 baseline"
echo "  6. rm/is_active transitions 0 -> 1 around step ~25k"
echo "     (gates: segs>=500 | updates>=50 | held_out_acc>=0.60)"
echo "  7. pref_rank_disc: pref_loss non-NaN after RM activation + first promotion"
echo "  8. disc_loss in 0.3–1.5, surrogate_reward > 0.1"
echo "  9. No shape/dimension errors"
echo ""
echo "Key difference from offline smoke: no --pref_rm, no --pref_expect_obs_dim."
echo "Online RM trains on obs_dim=112 (TFW-wrapped) — no slicing needed."
echo "Seed: ${SEED}"
echo "========================================"

# Safety check
if [[ ! -f "${EXPERT_DATA}" ]]; then
  echo "[ERROR] Missing Ant dataset: ${EXPERT_DATA}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export WANDB_DISABLED=true

PYTHONPATH="${REPO_ROOT}/sail_sb3_online" \
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \
  --env Ant-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 50000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --pref_rank_disc \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 32 \
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
