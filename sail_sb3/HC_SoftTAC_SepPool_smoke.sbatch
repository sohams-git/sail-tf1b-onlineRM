#!/bin/bash
#SBATCH --job-name=HC_STAC_SP_smoke
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=sail_sb3/logs/HC_SoftTAC_SepPool_smoke_%j.out
#SBATCH --account=sabymath0

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
EXPERT_DATA="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz"
PREF_RM="/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt"
SEED=1

cd "${REPO_ROOT}" || exit 1

mkdir -p sail_sb3/logs

echo "========================================"
echo "START: $(date)"
echo "SMOKE TEST: Soft-TAC separate pool (offline RM), 10k steps, seed=${SEED}"
echo ""
echo "What this verifies:"
echo "  1. soft_tac_pool starts with 4 expert eps at step 0 (all pre-truncation)"
echo "  2. soft_tac/student_pool_size grows every ~1k steps (one episode ~1k steps)"
echo "  3. soft_tac/pool_size = 4 + student_pool_size"
echo "  4. soft_tac_loss is non-zero from early in training (no 300k wait)"
echo "  5. soft_tac/y_pos_frac + y_neg_frac > 0 (diverse J labels, not all ties)"
echo "  6. pref_episodes unchanged (still expert-only at 10k, no promotions yet)"
echo "  7. No crashes, no shape/NaN errors"
echo ""
echo "Key change: Soft-TAC now uses dedicated soft_tac_pool (expert + ALL students)"
echo "  instead of pref_episodes (expert + promoted-only). Pool is diverse from step 1."
echo "  pref_rank_disc is OFF — pref_episodes stays expert-only for clean isolation."
echo ""
echo "Flags: soft_tac_weight=0.5  soft_tac_temp=1.0  soft_tac_max_student_trajs=200"
echo "       adaptive gt-source  lfd_mixing  teacher_buffer_size=1000"
echo "Seed: ${SEED}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_DISABLED=true

PYTHONPATH="${REPO_ROOT}" \
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 10000 \
  --learning_starts 1000 \
  --adaptive \
  --lfd_mixing \
  --teacher_buffer_size 1000 \
  --adaptive_score_source gt \
  --pref_rm "${PREF_RM}" \
  --soft_tac \
  --soft_tac_weight 0.5 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.0 \
  --soft_tac_max_student_trajs 200 \
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
