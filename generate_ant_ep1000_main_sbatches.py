"""
generate_ant_ep1000_main_sbatches.py
=====================================
Generates all sail_sb3 (offline) and sail_sb3_online sbatches for the
Ant-v2 ep1000_return3653 checkpoint dataset.

  sail_sb3/Ant_sbatch/          (9 offline variants)
  sail_sb3_online/Ant_sbatch/   (7 online variants)

Skips any sbatch file that already exists.
Errors if the NPZ glob matches no file.
"""

import os
import glob
import sys

REPO_ROOT  = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"

ANT_PREF_RM = (
    "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/"
    "ant_pebble_1M/reward_model_1000000_0.pt"
)

# ep label used in sbatch filenames (zero-padded) and glob (unpadded)
EP_LABEL = "ep1000"   # zero-padded for filename
EP_TAG   = "ep1000"   # matches generate_sail_dataset_from_checkpoint.py output
N_TRAJS  = 4

NPZ_GLOB = (
    "teacher_dataset_Ant/"
    "expert_data_no_img_Ant_scores_*_episodes_4_ep1000.npz"
)

# Scores already covered by existing sbatches — skip any NPZ whose score
# is in this set so we don't try to create duplicate SAIL sbatches.
EXISTING_SCORES = {3633, 3638}

ONLINE_RM_ARGS = """\
  --online_rm \\
  --rm_segment_len 50 \\
  --rm_train_freq 1000 \\
  --rm_gradient_steps 10 \\
  --rm_batch_size 256 \\
  --rm_lr 3e-4 \\
  --rm_max_segments 10000 \\
  --rm_min_segments 500 \\
  --rm_min_updates 50 \\
  --rm_min_acc 0.60 \\
  --rm_tie_margin 0.5 \\
  --rm_rescore_freq 20000 \\"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env_exports():
    return """\
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"""

def _header(job_name, mem, time_limit, log_path, array="1-5"):
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}_%A_%a.out
#SBATCH --array={array}
#SBATCH --account=sabymath0
"""

def _write_if_new(path, content):
    if os.path.exists(path):
        print(f"  [SKIP already exists] {os.path.basename(path)}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [CREATED] {os.path.basename(path)}")
    return True

def _score_from_fname(fname):
    return int(os.path.basename(fname).split("_scores_")[1].split("_")[0])


# ── Offline builders ──────────────────────────────────────────────────────────

def _ant_offline_sail(ep, score, n, npz_rel):
    tag = f"Ant_{ep}_n{n}_SAIL"
    log = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    body = f"""\
{_header(f"Ant_{ep}_n{n}_SAIL", "12G", "24:00:00", log)}
# FULL RUN — SAIL + Adaptive + LfD | Ant-v2
# Dataset: {npz_rel}

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_{ep}_sail"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _ant_offline_pref(ep, score, n, npz_rel, variant):
    pref_rm   = ANT_PREF_RM
    obs_dim   = 111
    pref_beta = 5.0

    if variant == "PAIL":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "SAIL_PREFD":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_PREFD"
        extra = f"""\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_PREFD":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_PREFD"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_TACD"
        extra = f"""\
  --adaptive_score_source rm \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_TACD"
        extra = f"""\
  --adaptive_score_source rm \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_QPREF"
        extra = f"""\
  --adaptive_score_source gt \\
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\"""
    elif variant == "PAIL_QPREF":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_QPREF"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim {obs_dim} \\
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log   = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"Ant_{ep}s{score}_{variant}", "12G", "24:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {variant} + Adaptive + LfD | Ant-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_{ep}_s{score}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


def _ant_offline_discrewabl(ep, score, n, npz_rel):
    tag     = f"Ant_{ep}_score{score}_n{n}_SAIL_PREFD_DiscRewAbl"
    log     = f"{REPO_ROOT}/sail_sb3/Ant_sbatch/logs/{tag}"
    pref_rm = ANT_PREF_RM
    body    = f"""\
{_header(f"Ant_{ep}s{score}_SAILPD_DiscRew", "12G", "24:00:00", log, array="1-3")}
# Discriminator reward ablation — SAIL + PREF-D-Disc (Ant-v2)
DISC_REWARD_TYPE="airl_backward_kl"

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: SAIL + PREF-D-Disc + Adaptive + LfD  [disc reward ablation] | Ant-v2"
echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_Ant"
export WANDB_NAME="Ant_{ep}_score{score}_n{n}_SAIL_PREFD_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_{ep}_s{score}_sail_prefd_discrew_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}" \\
"${{PYTHON_BIN}}" sail_sb3/scripts/train_sail.py \\
  --env Ant-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\
  --pref_rm "{pref_rm}" \\
  --pref_expect_obs_dim 111 \\
  --entcoeff 0.05 \\
  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\
  --seed "${{SEED}}" \\
  --device cpu

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


# ── Online builders ───────────────────────────────────────────────────────────

def _ant_online(ep, score, n, npz_rel, variant):
    pref_rm   = ANT_PREF_RM
    pref_beta = 5.0

    if variant == "PAIL":
        tag   = f"Ant_{ep}_n{n}_PAIL"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "SAIL_PREFD":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_PREFD"
        mem   = "12G"
        extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 16 \\
  --pref_max_teacher_trajs 500 \\"""
    elif variant == "PAIL_PREFD":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_PREFD"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --pref_rank_disc \\
  --pref_rank_weight 0.1 \\
  --pref_rank_batch_size 32 \\"""
    elif variant == "SAIL_TACD":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_TACD"
        mem   = "12G"
        extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "PAIL_TACD":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_TACD"
        mem   = "12G"
        extra = f"""\
  --adaptive_score_source gt \\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\"""
    elif variant == "SAIL_QPREF":
        tag   = f"Ant_{ep}_score{score}_n{n}_SAIL_QPREF"
        mem   = "12G"
        extra = """\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    elif variant == "PAIL_QPREF":
        tag   = f"Ant_{ep}_score{score}_n{n}_PAIL_QPREF"
        mem   = "12G"
        extra = f"""\
  --pref_reweight_teacher \\
  --pref_beta {pref_beta} \\
  --pref_max_teacher_trajs 500 \\
  --qpref \\
  --qpref_source student \\
  --qpref_weight 0.1 \\
  --qpref_temp 1.0 \\
  --qpref_batch_size 4 \\
  --qpref_grad_interval 10 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\"""
    else:
        raise ValueError(variant)

    log   = f"{REPO_ROOT}/sail_sb3_online/Ant_sbatch/logs/{tag}"
    short = variant.lower().replace("_", "")
    body  = f"""\
{_header(f"Ant_{ep}s{score}_{variant}", mem, "12:00:00", log)}
set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz_rel}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "${{REPO_ROOT}}/sail_sb3_online/Ant_sbatch/logs"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: Online RM + {variant} + Adaptive + LfD | Ant-v2"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

{_env_exports()}
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Ant"
export WANDB_NAME="{tag}_s${{SEED}}"
export WANDB_GROUP="Ant-v2_{ep}_{short}"
export WANDB_SILENT=true

cd "${{REPO_ROOT}}"
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"${{PYTHON_BIN}}" sail_sb3_online/scripts/train_sail.py \\
  --env Ant-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra}
{ONLINE_RM_ARGS}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug

EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""
    return tag + ".sbatch", body


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Find the new NPZ — must NOT be one of the existing scores
    all_matches = glob.glob(os.path.join(REPO_ROOT, NPZ_GLOB))
    new_matches = [p for p in all_matches
                   if _score_from_fname(p) not in EXISTING_SCORES]

    if not new_matches:
        print(f"ERROR: No new ep1000 NPZ found (glob: {NPZ_GLOB})")
        print(f"  All matches: {all_matches}")
        sys.exit(1)

    if len(new_matches) > 1:
        print(f"WARNING: Multiple new ep1000 NPZs found, using first: {new_matches}")

    npz_path = new_matches[0]
    npz_rel  = os.path.relpath(npz_path, REPO_ROOT)
    score    = _score_from_fname(npz_path)
    ep       = EP_LABEL
    n        = N_TRAJS

    print(f"\nAnt ep1000 dataset: {os.path.basename(npz_path)}")
    print(f"  score={score}, ep={ep}, n={n}")

    offline_dir = os.path.join(REPO_ROOT, "sail_sb3/Ant_sbatch")
    online_dir  = os.path.join(REPO_ROOT, "sail_sb3_online/Ant_sbatch")
    created     = []

    # ── Offline: SAIL + 7 pref variants + DiscRewAbl = 9 ──────────────────────
    fname, body = _ant_offline_sail(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _ant_offline_pref(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(offline_dir, fname), body):
            created.append(fname)

    fname, body = _ant_offline_discrewabl(ep, score, n, npz_rel)
    if _write_if_new(os.path.join(offline_dir, fname), body):
        created.append(fname)

    # ── Online: 7 variants ────────────────────────────────────────────────────
    for v in ["PAIL", "SAIL_PREFD", "PAIL_PREFD",
              "SAIL_TACD", "PAIL_TACD", "SAIL_QPREF", "PAIL_QPREF"]:
        fname, body = _ant_online(ep, score, n, npz_rel, v)
        if _write_if_new(os.path.join(online_dir, fname), body):
            created.append(fname)

    print(f"\n=== DONE: {len(created)} sbatches created ===")

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n--- Validation ---")
    assert os.path.exists(npz_path), f"NPZ not found: {npz_path}"
    assert os.path.exists(ANT_PREF_RM), f"Pref RM not found: {ANT_PREF_RM}"

    for fname in created:
        # Find the full path
        if "sail_sb3_online" in fname:
            full = os.path.join(online_dir, fname)
        else:
            full = os.path.join(offline_dir, fname)
        assert os.path.exists(full), f"Missing: {full}"
        txt = open(full).read()
        assert npz_rel in txt, f"{fname}: NPZ path not found in sbatch"
        if "PAIL" in fname or "PREFD" in fname or "TACD" in fname or "QPREF" in fname:
            assert ANT_PREF_RM in txt, f"{fname}: pref_rm missing"
        if "sail_sb3_online" in fname or "online" in fname.lower():
            assert "PYTHONPATH=" in txt and "sail_sb3_online" in txt, \
                f"{fname}: wrong PYTHONPATH for online"
            assert "--online_rm" in txt, f"{fname}: --online_rm missing"

    print("  All checks passed.")


if __name__ == "__main__":
    main()
