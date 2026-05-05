#!/bin/bash
#SBATCH --job-name=MW_Teacher_pp_v2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/teacher_training/metaworld/pick-place-v2/retrain_v2/slurm_logs/teacher_pick_place_v2_retrain_%j.out
#SBATCH --account=sabymath0

# Retraining v2 for pick-place-v2.
# v1 (sac_seed0, 1M steps) failed: SR=0 throughout all 1M steps.
# Diagnosis: task requires multi-stage exploration (reach→grasp→lift→place);
#            1M steps insufficient; auto ent_coef collapsed exploration too early.
# Fix: 3M steps, fixed ent_coef=0.1, higher learning_starts for broader exploration.

set -e

REPO_ROOT="/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN="/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
OUT_ROOT="${REPO_ROOT}/teacher_training/metaworld/pick-place-v2/retrain_v2"
SEED=0

mkdir -p "${OUT_ROOT}/slurm_logs"

cd "${REPO_ROOT}" || exit 1

echo "========================================================"
echo "START : $(date)"
echo "JOB   : ${SLURM_JOB_ID}"
echo "TASK  : pick-place-v2  [retrain_v2]"
echo "SEED  : ${SEED}"
echo "OUT   : ${OUT_ROOT}"
echo "CHANGES vs v1: 3M steps, ent_coef=0.1 (fixed), learning_starts=5000"
echo "========================================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHONPATH="${REPO_ROOT}" \
"${PYTHON_BIN}" mw_pipeline/teacher_training/train_mw_teacher.py \
    --task pick-place-v2 \
    --seed "${SEED}" \
    --algo sac \
    --total_timesteps 3000000 \
    --out_root "${OUT_ROOT}" \
    --checkpoint_freq 100000 \
    --eval_freq 50000 \
    --n_eval_episodes 20 \
    --target_success_rate 0.90 \
    --learning_rate 3e-4 \
    --buffer_size 1000000 \
    --batch_size 256 \
    --learning_starts 5000 \
    --ent_coef 0.1 \
    --device cpu \
    --log_to_file

EXIT_CODE=$?

echo "========================================================"
echo "END  : $(date)"
echo "EXIT : ${EXIT_CODE}"
echo "========================================================"

exit "${EXIT_CODE}"
