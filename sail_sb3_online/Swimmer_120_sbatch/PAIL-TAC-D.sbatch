#!/bin/bash
#SBATCH --job-name=Swim_120_ORM_PAILTD
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM/sail_sb3_online/Swimmer_120_sbatch/logs/PAIL-TAC-D_%A_%a.out
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
echo "Mode:  Online RM + PAIL + Soft-TAC-Disc (separate pool) + Adaptive SAIL + LfD | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       adaptive_score_source: gt (GT-based promotion)"
echo "       pref_reweight_teacher: Boltzmann-weighted expert BCE"
echo "       pref_beta=50.0 (aggressive sharpening for tight Swimmer return dist)"
echo "       soft_tac: weight=0.5  temp=1.0  tie_eps=0.0  max_student=200"
echo "       Online RM drives both reweighting and TAC student pool J labels"
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
export WANDB_NAME="Swimmer_120_ORM_PAIL-TAC-D_s${SEED}"
export WANDB_GROUP="Swimmer-v2_120_online_pail_tacd"
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
  --adaptive_score_source gt \
  --pref_reweight_teacher \
  --pref_beta 50.0 \
  --pref_max_teacher_trajs 500 \
  --soft_tac \
  --soft_tac_weight 0.5 \
  --soft_tac_temp 1.0 \
  --tac_tie_eps 0.0 \
  --soft_tac_max_student_trajs 200 \
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
