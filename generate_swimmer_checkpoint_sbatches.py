"""
generate_swimmer_checkpoint_sbatches.py
========================================
Generates all sail_sb3 (offline) and sail_sb3_online sbatches for the three new
Swimmer checkpoint-based datasets (ep0800, ep2100, ep3000), mirroring exactly
the Swimmer_120_sbatch and sail_sb3_online/Swimmer_120_sbatch reference folders.

Reference: sail_sb3/Swimmer_120_sbatch/  (16 variants)
           sail_sb3_online/Swimmer_120_sbatch/  (14 variants)

Output:  sail_sb3/Swimmer_sbatch/
         sail_sb3_online/Swimmer_sbatch/

Swimmer-specific constants (extracted from reference sbatches):
  - n_trajs = 4
  - pref_expect_obs_dim = 8  (native Swimmer-v2 obs dim)
  - pref_beta = 50.0         (tight return distribution)
  - pref_rank_weight (SAIL-PREFRANK) = 0.05
  - pref_rank_weight (PAIL-PREFRANK) = 0.5
  - pref_rank_batch_size = 32
  - soft_tac_weight (SAIL-TAC-D) = 0.05
  - soft_tac_weight (PAIL-TAC-D) = 0.5
  - soft_tac_temp = 1.0, tac_tie_eps = 0.0, soft_tac_max_student_trajs = 200
  - qpref_weight = 0.05 (offline)
  - qpref_weight = 0.1 (online)
  - qpref_source = student, qpref_grad_interval = 50 (offline) / 10 (online)
  - qpref_batch_size = 4 (online only)
  - teacher_buffer_size = 1000, learning_starts = 10000, entcoeff = 0.05
  - offline RM: /nfs/.../swimmer_pebble_1M/reward_model_1000000_0.pt
  - DiscRewAbl arrays: SAIL-PREFRANK-offline=1-3/airl_backward_kl; all others=1-5/gail_js
"""

import os

REPO_ROOT = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PYTHON_BIN = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"
PREF_RM = "/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/swimmer_pebble_1M/reward_model_1000000_0.pt"

OFFLINE_DIR = os.path.join(REPO_ROOT, "sail_sb3/Swimmer_sbatch")
ONLINE_DIR  = os.path.join(REPO_ROOT, "sail_sb3_online/Swimmer_sbatch")

# ── Datasets to generate sbatches for ──────────────────────────────────────
DATASETS = [
    {
        "ep": "0300", "score": 55, "n": 4,
        "npz": "teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_55_episodes_4_ep300.npz",
    },
    {
        "ep": "0800", "score": 83, "n": 4,
        "npz": "teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_83_episodes_4_ep800.npz",
    },
    {
        "ep": "2100", "score": 142, "n": 4,
        "npz": "teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_142_episodes_4_ep2100.npz",
    },
    {
        "ep": "3000", "score": 148, "n": 4,
        "npz": "teacher_dataset_Swimmer/expert_data_no_img_Swimmer_scores_148_episodes_4_ep3000.npz",
    },
]


# ── Common RM / Online-RM args ──────────────────────────────────────────────
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

PREF_RM_ARGS = f"""\
  --pref_rm "{PREF_RM}" \\
  --pref_expect_obs_dim 8 \\"""

PAIL_REWEIGHT_ARGS = """\
  --pref_reweight_teacher \\
  --pref_beta 50.0 \\
  --pref_max_teacher_trajs 500 \\"""


def write_sbatch(path, content):
    if os.path.exists(path):
        print(f"  [SKIP] {os.path.basename(path)} (exists)")
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [OK]   {os.path.basename(path)}")
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE  sail_sb3/Swimmer_sbatch/
# ══════════════════════════════════════════════════════════════════════════════

def make_offline_header(ep, score, n, variant, array_spec="1-5",
                        full_log_path=False, sbatch_dir=None, log_name=None):
    if full_log_path and sbatch_dir and log_name:
        output_line = f"#SBATCH --output={sbatch_dir}/logs/{log_name}_%A_%a.out"
    else:
        output_line = f"#SBATCH --output=logs/{log_name}_%A_%a.out"
    return f"""\
#!/bin/bash
#SBATCH --job-name=Swim_ep{ep}_sc{score}_{variant}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
{output_line}
#SBATCH --array={array_spec}
#SBATCH --account=sabymath0
"""


def make_offline_common(ep, score, n, npz, sbatch_dir, mode_echo):
    return f"""\
set -e

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz}"
SEED=$SLURM_ARRAY_TASK_ID

cd "${{REPO_ROOT}}" || exit 1

mkdir -p {os.path.relpath(sbatch_dir, REPO_ROOT)}/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  {mode_echo} | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       dataset: ep{ep} score{score} n{n}"
echo "       entcoeff=0.05"
echo "Seed:  ${{SEED}}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"

"""


def make_offline_wandb(ep, score, n, group_suffix, run_suffix=None):
    rs = run_suffix or group_suffix.replace("_", "-").upper()
    return f"""\
export WANDB_PROJECT="SAIL_SB3_Swimmer"
export WANDB_NAME="Swimmer_ep{ep}sc{score}_{rs}_s${{SEED}}"
export WANDB_GROUP="Swimmer-v2_ep{ep}_{group_suffix}"
export WANDB_SILENT=true

"""


def make_offline_footer():
    return """\

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
"""


def offline_base_cmd(ep, score, n, npz, extra_args=""):
    return f"""\
PYTHONPATH="${{REPO_ROOT}}" \\
"$PYTHON_BIN" sail_sb3/scripts/train_sail.py \\
  --env Swimmer-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra_args}  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu
"""


def gen_offline_sail(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL.sbatch"
    hdr = make_offline_header(ep, score, n, "SAIL",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="SAIL")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "SAIL (base) + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "sail", "SAIL")
    cmd = offline_base_cmd(ep, score, n, npz)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_pail(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL.sbatch"
    hdr = make_offline_header(ep, score, n, "PAIL",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="PAIL")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "PAIL (PrefReweight) + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "pail", "PAIL")
    extra = f"""\
  --adaptive_score_source gt \\
{PAIL_REWEIGHT_ARGS}
{PREF_RM_ARGS}
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_sail_prefrank(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_PREFRANK.sbatch"
    hdr = make_offline_header(ep, score, n, "SAILPR",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="SAIL-PREFRANK")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "SAIL + PrefRank-Disc + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "sail_prefrank", "SAIL-PREFRANK")
    extra = f"""\
  --adaptive_score_source gt \\
{PREF_RM_ARGS}
  --pref_rank_disc \\
  --pref_rank_weight 0.05 \\
  --pref_rank_batch_size 32 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_pail_prefrank(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_PREFRANK.sbatch"
    hdr = make_offline_header(ep, score, n, "PAILPR",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="PAIL-PREFRANK")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "PAIL + PrefRank-Disc + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "pail_prefrank", "PAIL-PREFRANK")
    extra = f"""\
  --adaptive_score_source gt \\
{PAIL_REWEIGHT_ARGS}
{PREF_RM_ARGS}
  --pref_rank_disc \\
  --pref_rank_weight 0.5 \\
  --pref_rank_batch_size 32 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_sail_tacd(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_TACD.sbatch"
    hdr = make_offline_header(ep, score, n, "SAILTD",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="SAIL-TAC-D")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "SAIL + Soft-TAC-Disc + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "sail_tacd", "SAIL-TAC-D")
    extra = f"""\
  --adaptive_score_source gt \\
{PREF_RM_ARGS}
  --soft_tac \\
  --soft_tac_weight 0.05 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_pail_tacd(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_TACD.sbatch"
    hdr = make_offline_header(ep, score, n, "PAILTD",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="PAIL-TAC-D")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "PAIL + Soft-TAC-Disc + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "pail_tacd", "PAIL-TAC-D")
    extra = f"""\
  --adaptive_score_source gt \\
{PAIL_REWEIGHT_ARGS}
{PREF_RM_ARGS}
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_sail_qpref(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_QPREF.sbatch"
    hdr = make_offline_header(ep, score, n, "SAILQP",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="SAIL-QPREF")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "SAIL + QPREF (student source) + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "sail_qpref", "SAIL-QPREF")
    extra = f"""\
  --adaptive_score_source gt \\
{PREF_RM_ARGS}
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


def gen_offline_pail_qpref(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_QPREF.sbatch"
    hdr = make_offline_header(ep, score, n, "PAILQP",
                              full_log_path=True, sbatch_dir=sbatch_dir, log_name="PAIL-QPREF")
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 "PAIL + QPREF (student source) + Adaptive + LfD")
    wandb = make_offline_wandb(ep, score, n, "pail_qpref", "PAIL-QPREF")
    extra = f"""\
  --adaptive_score_source gt \\
{PAIL_REWEIGHT_ARGS}
{PREF_RM_ARGS}
  --qpref \\
  --qpref_weight 0.05 \\
  --qpref_source student \\
  --qpref_grad_interval 50 \\
  --qpref_guard_threshold -0.3 \\
  --qpref_guard_confirm 3 \\
  --qpref_guard_window 10 \\
  --qpref_guard_positive_threshold 1.0 \\
  --pref_max_student_trajs 500 \\
"""
    cmd = offline_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


# ── DiscRewardAblation variants (offline) ────────────────────────────────────

def make_disc_rew_header(ep, score, n, variant_tag, array_spec, sbatch_dir, log_name, disc_type):
    return f"""\
#!/bin/bash
#SBATCH --job-name=Swim_ep{ep}_{variant_tag}_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output={sbatch_dir}/logs/{log_name}_%A_%a.out
#SBATCH --array={array_spec}
#SBATCH --account=sabymath0

# ============================================================
# Discriminator reward ablation: {disc_type}
# ============================================================
DISC_REWARD_TYPE="{disc_type}"
"""


def gen_offline_discrew(ds, sbatch_dir, base_variant, base_extra="",
                        array_spec="1-5", disc_type="gail_js",
                        variant_tag=None, log_name=None, group_suffix=None, run_suffix=None):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    variant_tag = variant_tag or base_variant
    log_name = log_name or f"{base_variant}-DiscRewardAblation"
    group_suffix = group_suffix or base_variant.lower().replace("-", "_") + "_discrew"
    run_suffix = run_suffix or f"{base_variant.upper()}-DR"
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_{base_variant.replace('-','_')}_DiscRewardAblation.sbatch"

    hdr = make_disc_rew_header(ep, score, n, variant_tag, array_spec, sbatch_dir, log_name, disc_type)
    common = make_offline_common(ep, score, n, npz, sbatch_dir,
                                 f"{base_variant} + DiscRewardAblation + Adaptive + LfD")
    wandb = f"""\
export WANDB_PROJECT="SAIL_SB3_Swimmer"
export WANDB_NAME="Swimmer_ep{ep}sc{score}_{base_variant.replace('-','')}_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Swimmer-v2_ep{ep}_{group_suffix}_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

"""
    extra_full = base_extra + "  --disc_reward_type \"${DISC_REWARD_TYPE}\" \\\n"
    cmd = offline_base_cmd(ep, score, n, npz, extra_full)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_offline_footer())


# ══════════════════════════════════════════════════════════════════════════════
# ONLINE  sail_sb3_online/Swimmer_sbatch/
# ══════════════════════════════════════════════════════════════════════════════

def make_online_header(ep, score, n, variant, array_spec="1-5",
                       sbatch_dir=None, log_name=None):
    return f"""\
#!/bin/bash
#SBATCH --job-name=Swim_ep{ep}_{variant}_ORM
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output={sbatch_dir}/logs/{log_name}_%A_%a.out
#SBATCH --array={array_spec}
#SBATCH --account=sabymath0
"""


def make_online_common(ep, score, n, npz, sbatch_dir, mode_echo):
    return f"""\
set -e

REPO_ROOT="{REPO_ROOT}"
PYTHON_BIN="{PYTHON_BIN}"
EXPERT_DATA="${{REPO_ROOT}}/{npz}"
SEED=$SLURM_ARRAY_TASK_ID

cd "${{REPO_ROOT}}" || exit 1

mkdir -p {os.path.relpath(sbatch_dir, REPO_ROOT)}/logs

echo "========================================"
echo "START: $(date)"
echo "Mode:  Online RM + {mode_echo} | Swimmer-v2"
echo "       ring buffer: teacher_buffer_size=1000 (1 episode, ep_len=1000)"
echo "       dataset: ep{ep} score{score} n{n}"
echo "       RM activation: segs>=500 | updates>=50 | held_out_acc>=0.60"
echo "       RM rescore: every 20000 env steps"
echo "       entcoeff=0.05"
echo "Seed:  ${{SEED}}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"

"""


def make_online_wandb(ep, score, n, group_suffix, run_suffix):
    return f"""\
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Swimmer"
export WANDB_NAME="Swimmer_ep{ep}sc{score}_ORM_{run_suffix}_s${{SEED}}"
export WANDB_GROUP="Swimmer-v2_ep{ep}_online_{group_suffix}"
export WANDB_SILENT=true

"""


def online_base_cmd(ep, score, n, npz, extra_args=""):
    return f"""\
PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\
"$PYTHON_BIN" sail_sb3_online/scripts/train_sail.py \\
  --env Swimmer-v2 \\
  --expert_data "${{EXPERT_DATA}}" \\
  --total_timesteps 1000000 \\
  --learning_starts 10000 \\
  --adaptive \\
  --lfd_mixing \\
  --teacher_buffer_size 1000 \\
{extra_args}{ONLINE_RM_ARGS}
  --entcoeff 0.05 \\
  --seed "${{SEED}}" \\
  --device cpu \\
  --debug
"""


def make_online_footer():
    return """\

EXIT_CODE=$?

echo "========================================"
echo "END: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
"""


def gen_online_pail(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "PAIL", sbatch_dir=sbatch_dir, log_name="PAIL")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "PAIL (PrefReweight) + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "pail", "PAIL")
    extra = f"""\
{PAIL_REWEIGHT_ARGS}
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_sail_prefrank(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_PREFRANK_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "SAILPR", sbatch_dir=sbatch_dir, log_name="SAIL-PREFRANK")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "SAIL + PrefRank-Disc + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "sail_prefrank", "SAIL-PREFRANK")
    extra = """\
  --pref_rank_disc \\
  --pref_rank_weight 0.05 \\
  --pref_rank_batch_size 32 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_pail_prefrank(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_PREFRANK_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "PAILPR", sbatch_dir=sbatch_dir, log_name="PAIL-PREFRANK")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "PAIL + PrefRank-Disc + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "pail_prefrank", "PAIL-PREFRANK")
    extra = f"""\
{PAIL_REWEIGHT_ARGS}
  --pref_rank_disc \\
  --pref_rank_weight 0.5 \\
  --pref_rank_batch_size 32 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_sail_tacd(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_TACD_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "SAILTD", sbatch_dir=sbatch_dir, log_name="SAIL-TAC-D")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "SAIL + Soft-TAC-Disc + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "sail_tacd", "SAIL-TAC-D")
    extra = """\
  --adaptive_score_source gt \\
  --soft_tac \\
  --soft_tac_weight 0.05 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_pail_tacd(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_TACD_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "PAILTD", sbatch_dir=sbatch_dir, log_name="PAIL-TAC-D")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "PAIL + Soft-TAC-Disc + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "pail_tacd", "PAIL-TAC-D")
    extra = f"""\
  --adaptive_score_source gt \\
{PAIL_REWEIGHT_ARGS}
  --soft_tac \\
  --soft_tac_weight 0.5 \\
  --soft_tac_temp 1.0 \\
  --tac_tie_eps 0.0 \\
  --soft_tac_max_student_trajs 200 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_sail_qpref(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_SAIL_QPREF_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "SAILQP", sbatch_dir=sbatch_dir, log_name="SAIL-QPREF")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "SAIL + QPREF (student source) + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "sail_qpref", "SAIL-QPREF")
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
  --pref_max_student_trajs 500 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def gen_online_pail_qpref(ds, sbatch_dir):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_PAIL_QPREF_ONLINE.sbatch"
    hdr = make_online_header(ep, score, n, "PAILQP", sbatch_dir=sbatch_dir, log_name="PAIL-QPREF")
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                "PAIL + QPREF (student source) + Adaptive SAIL + LfD")
    wandb = make_online_wandb(ep, score, n, "pail_qpref", "PAIL-QPREF")
    extra = f"""\
{PAIL_REWEIGHT_ARGS}
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
  --pref_max_student_trajs 500 \\
"""
    cmd = online_base_cmd(ep, score, n, npz, extra)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


def make_online_disc_rew_header(ep, score, n, variant_tag, array_spec, sbatch_dir, log_name, disc_type):
    return f"""\
#!/bin/bash
#SBATCH --job-name=Swim_ep{ep}_{variant_tag}_DR
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output={sbatch_dir}/logs/{log_name}_%A_%a.out
#SBATCH --array={array_spec}
#SBATCH --account=sabymath0

# ============================================================
# Discriminator reward ablation: {disc_type}
# ============================================================
DISC_REWARD_TYPE="{disc_type}"
"""


def gen_online_discrew(ds, sbatch_dir, base_variant, base_extra="",
                       disc_type="gail_js", variant_tag=None, log_name=None,
                       group_suffix=None, run_suffix=None):
    ep, score, n, npz = ds["ep"], ds["score"], ds["n"], ds["npz"]
    variant_tag = variant_tag or base_variant
    log_name = log_name or f"{base_variant}-DiscRewardAblation"
    group_suffix = group_suffix or base_variant.lower().replace("-", "_") + "_discrew"
    run_suffix = run_suffix or f"{base_variant.upper()}-DR"
    safe_name = base_variant.replace("-", "_")
    fname = f"Swimmer_ep{ep}_score{score}_n{n}_{safe_name}_DiscRewardAblation_ONLINE.sbatch"

    hdr = make_online_disc_rew_header(ep, score, n, variant_tag, "1-5", sbatch_dir, log_name, disc_type)
    common = make_online_common(ep, score, n, npz, sbatch_dir,
                                f"{base_variant} + DiscRewardAblation + Adaptive SAIL + LfD")
    wandb = f"""\
export WANDB_PROJECT="SAIL_SB3_OnlineRM_Swimmer"
export WANDB_NAME="Swimmer_ep{ep}sc{score}_ORM_{base_variant.replace('-','')}_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"
export WANDB_GROUP="Swimmer-v2_ep{ep}_online_{group_suffix}_${{DISC_REWARD_TYPE}}"
export WANDB_SILENT=true

"""
    extra_full = base_extra + "  --disc_reward_type \"${DISC_REWARD_TYPE}\" \\\n"
    cmd = online_base_cmd(ep, score, n, npz, extra_full)
    return write_sbatch(os.path.join(sbatch_dir, fname),
                        hdr + "\n" + common + wandb + cmd + make_online_footer())


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    created = skipped = 0

    for ds in DATASETS:
        ep, score, n = ds["ep"], ds["score"], ds["n"]
        npz_full = os.path.join(REPO_ROOT, ds["npz"])
        if not os.path.exists(npz_full):
            print(f"\n[WARNING] Dataset missing: {npz_full}  — skipping ep{ep}")
            continue

        print(f"\n=== ep{ep} score{score} n{n} ===")

        # ── Offline (sail_sb3) ──────────────────────────────────────────────
        print("  -- offline --")
        c  = gen_offline_sail(ds, OFFLINE_DIR)
        c += gen_offline_pail(ds, OFFLINE_DIR)
        c += gen_offline_sail_prefrank(ds, OFFLINE_DIR)
        c += gen_offline_pail_prefrank(ds, OFFLINE_DIR)
        c += gen_offline_sail_tacd(ds, OFFLINE_DIR)
        c += gen_offline_pail_tacd(ds, OFFLINE_DIR)
        c += gen_offline_sail_qpref(ds, OFFLINE_DIR)
        c += gen_offline_pail_qpref(ds, OFFLINE_DIR)

        # DiscRewardAblation offline
        pail_extra = (f"  --adaptive_score_source gt \\\n"
                      f"{PAIL_REWEIGHT_ARGS}\n{PREF_RM_ARGS}\n")
        sail_rm_extra = f"{PREF_RM_ARGS}\n  --adaptive_score_source gt \\\n"
        sail_prefrank_extra = (sail_rm_extra +
                               "  --pref_rank_disc \\\n"
                               "  --pref_rank_weight 0.05 \\\n"
                               "  --pref_rank_batch_size 32 \\\n")
        pail_prefrank_extra = (pail_extra +
                               "  --pref_rank_disc \\\n"
                               "  --pref_rank_weight 0.5 \\\n"
                               "  --pref_rank_batch_size 32 \\\n")
        sail_tacd_extra = (sail_rm_extra +
                           "  --soft_tac \\\n"
                           "  --soft_tac_weight 0.05 \\\n"
                           "  --soft_tac_temp 1.0 \\\n"
                           "  --tac_tie_eps 0.0 \\\n"
                           "  --soft_tac_max_student_trajs 200 \\\n")
        pail_tacd_extra = (pail_extra +
                           "  --soft_tac \\\n"
                           "  --soft_tac_weight 0.5 \\\n"
                           "  --soft_tac_temp 1.0 \\\n"
                           "  --tac_tie_eps 0.0 \\\n"
                           "  --soft_tac_max_student_trajs 200 \\\n")
        sail_qpref_extra = (sail_rm_extra +
                            "  --qpref \\\n"
                            "  --qpref_weight 0.05 \\\n"
                            "  --qpref_source student \\\n"
                            "  --qpref_grad_interval 50 \\\n"
                            "  --qpref_guard_threshold -0.3 \\\n"
                            "  --qpref_guard_confirm 3 \\\n"
                            "  --qpref_guard_window 10 \\\n"
                            "  --qpref_guard_positive_threshold 1.0 \\\n"
                            "  --pref_max_student_trajs 500 \\\n")
        pail_qpref_extra = (pail_extra +
                            "  --qpref \\\n"
                            "  --qpref_weight 0.05 \\\n"
                            "  --qpref_source student \\\n"
                            "  --qpref_grad_interval 50 \\\n"
                            "  --qpref_guard_threshold -0.3 \\\n"
                            "  --qpref_guard_confirm 3 \\\n"
                            "  --qpref_guard_window 10 \\\n"
                            "  --qpref_guard_positive_threshold 1.0 \\\n"
                            "  --pref_max_student_trajs 500 \\\n")

        c += gen_offline_discrew(ds, OFFLINE_DIR, "SAIL", base_extra="",
                                 group_suffix="sail_discrew", run_suffix="SAIL-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "PAIL", base_extra=pail_extra,
                                 group_suffix="pail_discrew", run_suffix="PAIL-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "SAIL-PREFRANK",
                                 base_extra=sail_prefrank_extra,
                                 array_spec="1-3", disc_type="airl_backward_kl",
                                 variant_tag="SAILPR", log_name="SAIL-PREFRANK-DiscRewardAblation",
                                 group_suffix="sail_prefrank_discrew", run_suffix="SAILPR-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "PAIL-PREFRANK",
                                 base_extra=pail_prefrank_extra, variant_tag="PAILPR",
                                 log_name="PAIL-PREFRANK-DiscRewardAblation",
                                 group_suffix="pail_prefrank_discrew", run_suffix="PAILPR-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "SAIL-TAC-D",
                                 base_extra=sail_tacd_extra, variant_tag="SAILTD",
                                 log_name="SAIL-TAC-D-DiscRewardAblation",
                                 group_suffix="sail_tacd_discrew", run_suffix="SAILTD-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "PAIL-TAC-D",
                                 base_extra=pail_tacd_extra, variant_tag="PAILTD",
                                 log_name="PAIL-TAC-D-DiscRewardAblation",
                                 group_suffix="pail_tacd_discrew", run_suffix="PAILTD-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "SAIL-QPREF",
                                 base_extra=sail_qpref_extra, variant_tag="SAILQP",
                                 log_name="SAIL-QPREF-DiscRewardAblation",
                                 group_suffix="sail_qpref_discrew", run_suffix="SAILQP-DR")
        c += gen_offline_discrew(ds, OFFLINE_DIR, "PAIL-QPREF",
                                 base_extra=pail_qpref_extra, variant_tag="PAILQP",
                                 log_name="PAIL-QPREF-DiscRewardAblation",
                                 group_suffix="pail_qpref_discrew", run_suffix="PAILQP-DR")

        # ── Online (sail_sb3_online) ────────────────────────────────────────
        print("  -- online --")
        c += gen_online_pail(ds, ONLINE_DIR)
        c += gen_online_sail_prefrank(ds, ONLINE_DIR)
        c += gen_online_pail_prefrank(ds, ONLINE_DIR)
        c += gen_online_sail_tacd(ds, ONLINE_DIR)
        c += gen_online_pail_tacd(ds, ONLINE_DIR)
        c += gen_online_sail_qpref(ds, ONLINE_DIR)
        c += gen_online_pail_qpref(ds, ONLINE_DIR)

        # Online DiscRewardAblation
        pail_online_extra = f"{PAIL_REWEIGHT_ARGS}\n"
        sail_pr_online_extra = ("  --pref_rank_disc \\\n"
                                "  --pref_rank_weight 0.05 \\\n"
                                "  --pref_rank_batch_size 32 \\\n")
        pail_pr_online_extra = (pail_online_extra +
                                "  --pref_rank_disc \\\n"
                                "  --pref_rank_weight 0.5 \\\n"
                                "  --pref_rank_batch_size 32 \\\n")
        sail_tacd_online_extra = ("  --adaptive_score_source gt \\\n"
                                  "  --soft_tac \\\n"
                                  "  --soft_tac_weight 0.05 \\\n"
                                  "  --soft_tac_temp 1.0 \\\n"
                                  "  --tac_tie_eps 0.0 \\\n"
                                  "  --soft_tac_max_student_trajs 200 \\\n")
        pail_tacd_online_extra = ("  --adaptive_score_source gt \\\n" +
                                  pail_online_extra +
                                  "  --soft_tac \\\n"
                                  "  --soft_tac_weight 0.5 \\\n"
                                  "  --soft_tac_temp 1.0 \\\n"
                                  "  --tac_tie_eps 0.0 \\\n"
                                  "  --soft_tac_max_student_trajs 200 \\\n")
        sail_qpref_online_extra = ("  --pref_max_teacher_trajs 500 \\\n"
                                   "  --qpref \\\n"
                                   "  --qpref_source student \\\n"
                                   "  --qpref_weight 0.1 \\\n"
                                   "  --qpref_temp 1.0 \\\n"
                                   "  --qpref_batch_size 4 \\\n"
                                   "  --qpref_grad_interval 10 \\\n"
                                   "  --qpref_guard_threshold -0.3 \\\n"
                                   "  --qpref_guard_confirm 3 \\\n"
                                   "  --qpref_guard_window 10 \\\n"
                                   "  --qpref_guard_positive_threshold 1.0 \\\n"
                                   "  --pref_max_student_trajs 500 \\\n")
        pail_qpref_online_extra = (pail_online_extra + sail_qpref_online_extra)

        c += gen_online_discrew(ds, ONLINE_DIR, "PAIL", base_extra=pail_online_extra,
                                log_name="PAIL-DiscRewardAblation",
                                group_suffix="pail_discrew", run_suffix="PAIL-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "SAIL-PREFRANK",
                                base_extra=sail_pr_online_extra,
                                log_name="SAIL-PREFRANK-DiscRewardAblation",
                                group_suffix="sail_prefrank_discrew", run_suffix="SAILPR-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "PAIL-PREFRANK",
                                base_extra=pail_pr_online_extra,
                                log_name="PAIL-PREFRANK-DiscRewardAblation",
                                group_suffix="pail_prefrank_discrew", run_suffix="PAILPR-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "SAIL-TAC-D",
                                base_extra=sail_tacd_online_extra,
                                log_name="SAIL-TAC-D-DiscRewardAblation",
                                group_suffix="sail_tacd_discrew", run_suffix="SAILTD-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "PAIL-TAC-D",
                                base_extra=pail_tacd_online_extra,
                                log_name="PAIL-TAC-D-DiscRewardAblation",
                                group_suffix="pail_tacd_discrew", run_suffix="PAILTD-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "SAIL-QPREF",
                                base_extra=sail_qpref_online_extra,
                                log_name="SAIL-QPREF-DiscRewardAblation",
                                group_suffix="sail_qpref_discrew", run_suffix="SAILQP-DR")
        c += gen_online_discrew(ds, ONLINE_DIR, "PAIL-QPREF",
                                base_extra=pail_qpref_online_extra,
                                log_name="PAIL-QPREF-DiscRewardAblation",
                                group_suffix="pail_qpref_discrew", run_suffix="PAILQP-DR")

        created += c
        skipped += (30 - c)

    print(f"\n=== DONE: {created} created, {skipped} skipped (already exist) ===")


if __name__ == "__main__":
    main()
