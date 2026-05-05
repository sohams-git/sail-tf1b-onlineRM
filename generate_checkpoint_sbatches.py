#!/usr/bin/env python3
"""
Generate all variant sbatches for the 7 checkpoint-based datasets.
Run from REPO_ROOT:
  python generate_checkpoint_sbatches.py
"""
import os

REPO_ROOT = "/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM"
PY = "/home/sohams/miniconda3/envs/sail_sb3_env/bin/python"

# ── offline RM paths ──────────────────────────────────────────────────────────
RM = {
    "Ant":         f"{REPO_ROOT}/../BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt",
    "HalfCheetah": "/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt",
    "Hopper":      f"{REPO_ROOT}/../BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt",
    "Walker2d":    f"{REPO_ROOT}/../BPref/exported_models/walker2d_pebble_1M/reward_model_1000000_0.pt",
}
# resolve relative paths
RM = {k: os.path.normpath(v) for k, v in RM.items()}

# ── dataset specs ─────────────────────────────────────────────────────────────
DATASETS = [
    dict(env="Ant",         env_id="Ant-v2",         ep="ep0200", score=2878, n=4,
         npz="teacher_dataset_Ant/expert_data_no_img_Ant_scores_2878_episodes_4_ep200.npz",
         short="Ant", tag="Ant_ep0200", wandb_proj="SAIL_SB3_Ant",
         wandb_proj_online="SAIL_SB3_OnlineRM_Ant",
         pref_beta=5.0, pref_expect_obs_dim=111, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.1, pref_rank_batch_size=32,
         sail_tac_weight=0.5, pail_tac_weight=0.5,
         sail_tac_src_offline="rm", pail_tac_src_offline="gt",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.05, qpref_weight_pail=0.05,
         prefd_name="PREF-D", discrew_batch=16),
    dict(env="Ant",         env_id="Ant-v2",         ep="ep1000", score=3638, n=4,
         npz="teacher_dataset_Ant/expert_data_no_img_Ant_scores_3638_episodes_4_ep1000.npz",
         short="Ant", tag="Ant_ep1000", wandb_proj="SAIL_SB3_Ant",
         wandb_proj_online="SAIL_SB3_OnlineRM_Ant",
         pref_beta=5.0, pref_expect_obs_dim=111, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.1, pref_rank_batch_size=32,
         sail_tac_weight=0.5, pail_tac_weight=0.5,
         sail_tac_src_offline="rm", pail_tac_src_offline="gt",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.05, qpref_weight_pail=0.05,
         prefd_name="PREF-D", discrew_batch=16),
    dict(env="Hopper",      env_id="Hopper-v2",      ep="ep0200", score=847,  n=10,
         npz="teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_847_episodes_10_ep200.npz",
         short="Hop", tag="Hopper_ep0200", wandb_proj="SAIL_SB3_Hopper",
         wandb_proj_online="SAIL_SB3_OnlineRM_Hopper",
         pref_beta=1.0, pref_expect_obs_dim=11, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.03, pref_rank_batch_size=32,
         sail_tac_weight=0.03, pail_tac_weight=0.03,
         sail_tac_src_offline="gt", pail_tac_src_offline="rm",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.03, qpref_weight_pail=0.05,
         prefd_name="PREFRANK", discrew_batch=32),
    dict(env="Hopper",      env_id="Hopper-v2",      ep="ep0900", score=3083, n=10,
         npz="teacher_dataset_Hopper/expert_data_no_img_Hopper_scores_3083_episodes_10_ep900.npz",
         short="Hop", tag="Hopper_ep0900", wandb_proj="SAIL_SB3_Hopper",
         wandb_proj_online="SAIL_SB3_OnlineRM_Hopper",
         pref_beta=1.0, pref_expect_obs_dim=11, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.03, pref_rank_batch_size=32,
         sail_tac_weight=0.03, pail_tac_weight=0.03,
         sail_tac_src_offline="gt", pail_tac_src_offline="rm",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.03, qpref_weight_pail=0.05,
         prefd_name="PREFRANK", discrew_batch=32),
    dict(env="HalfCheetah", env_id="HalfCheetah-v2", ep="ep0200", score=3617, n=4,
         npz="teacher_dataset_HC/expert_data_no_img_HalfCheetah_scores_3617_episodes_4_ep200.npz",
         short="HC", tag="HC_ep0200", wandb_proj="SAIL_SB3_HalfCheetah",
         wandb_proj_online="SAIL_SB3_OnlineRM_HalfCheetah",
         pref_beta=1.0, pref_expect_obs_dim=None, mem="8G",
         entcoeff_base=0.01,
         pref_rank_weight=0.1, pref_rank_batch_size=16,
         sail_tac_weight=0.5, pail_tac_weight=0.5,
         sail_tac_src_offline="rm", pail_tac_src_offline="rm",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.05, qpref_weight_pail=0.05,
         prefd_name="PREF-D", discrew_batch=16),
    dict(env="Walker2d",    env_id="Walker2d-v2",    ep="ep0100", score=2768, n=10,
         npz="teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_2768_episodes_10_ep100.npz",
         short="W2d", tag="Walker2d_ep0100", wandb_proj="SAIL_SB3_Walker2d",
         wandb_proj_online="SAIL_SB3_OnlineRM_Walker2d",
         pref_beta=1.0, pref_expect_obs_dim=17, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.1, pref_rank_batch_size=32,
         sail_tac_weight=0.1, pail_tac_weight=0.1,
         sail_tac_src_offline="gt", pail_tac_src_offline="rm",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.05, qpref_weight_pail=0.05,
         prefd_name="PREFRANK", discrew_batch=32),
    dict(env="Walker2d",    env_id="Walker2d-v2",    ep="ep0300", score=3881, n=10,
         npz="teacher_dataset_Walker2d/expert_data_no_img_Walker2d_scores_3881_episodes_10_ep300.npz",
         short="W2d", tag="Walker2d_ep0300", wandb_proj="SAIL_SB3_Walker2d",
         wandb_proj_online="SAIL_SB3_OnlineRM_Walker2d",
         pref_beta=1.0, pref_expect_obs_dim=17, mem="12G",
         entcoeff_base=0.05,
         pref_rank_weight=0.1, pref_rank_batch_size=32,
         sail_tac_weight=0.1, pail_tac_weight=0.1,
         sail_tac_src_offline="gt", pail_tac_src_offline="rm",
         sail_tac_src_online="gt", pail_tac_src_online="gt",
         qpref_weight_sail=0.05, qpref_weight_pail=0.05,
         prefd_name="PREFRANK", discrew_batch=32),
]

HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out_path}
#SBATCH --array={array}
#SBATCH --account=sabymath0
"""

BODY_COMMON = """\
set -euo pipefail

REPO_ROOT="{repo_root}"
PYTHON_BIN="{py}"
EXPERT_DATA="${{REPO_ROOT}}/{npz}"
SEED=$SLURM_ARRAY_TASK_ID

mkdir -p "{log_dir}"

echo "========================================"
echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"
echo "Mode: {mode_desc}"
echo "Dataset: ${{EXPERT_DATA}}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"

export WANDB_PROJECT="{wandb_proj}"
export WANDB_NAME="{wandb_name}"
export WANDB_GROUP="{wandb_group}"
export WANDB_SILENT=true

"""

FOOTER = """\
EXIT_CODE=$?
echo "========================================"; echo "END: $(date)"; echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"
exit $EXIT_CODE
"""

ONLINE_RM_FLAGS = """\
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


def base_flags(d, ec):
    return (
        f'  --env {d["env_id"]} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --entcoeff {ec} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )


def pref_rm_flag(d):
    rm = RM[d["env"]]
    s = f'  --pref_rm "{rm}" \\\n'
    if d["pref_expect_obs_dim"] is not None:
        s += f'  --pref_expect_obs_dim {d["pref_expect_obs_dim"]} \\\n'
    return s


def write_sbatch(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  SKIP (exists): {os.path.relpath(path, REPO_ROOT)}")
        return False
    with open(path, "w") as f:
        f.write(content)
    print(f"  WROTE: {os.path.relpath(path, REPO_ROOT)}")
    return True


def make_header(d, variant_tag, time, array="1-5", memo=None):
    env_abbr = {"Ant": "Ant", "HalfCheetah": "HC", "Hopper": "Hop", "Walker2d": "W2d"}[d["env"]]
    job = f'{env_abbr}_{d["ep"]}s{d["score"]}_{variant_tag}'[:32]
    return HEADER.format(
        job_name=job,
        mem=memo or d["mem"],
        time=time,
        out_path="{log_dir}/{variant}_%A_%a.out",  # placeholder replaced below
        array=array,
    )


# ── main loop ─────────────────────────────────────────────────────────────────
created = 0
skipped = 0

for d in DATASETS:
    ep = d["ep"]
    sc = d["score"]
    n  = d["n"]
    env = d["env"]
    env_id = d["env_id"]
    tag = d["tag"]           # e.g. "Ant_ep0200"
    slug = f'{tag}_score{sc}_n{n}'  # e.g. "Ant_ep0200_score2878_n4"
    pd_name = d["prefd_name"]      # "PREF-D" or "PREFRANK"
    pd_slug = pd_name.replace("-", "")  # "PREFD" or "PREFRANK"

    pref_rm_str = pref_rm_flag(d)
    ec_base = d["entcoeff_base"]   # 0.05 for most, 0.01 for HC

    # ── helpers ──────────────────────────────────────────────────────────────
    def make_sbatch_offline(variant_tag, mode_desc, flags_str, time="24:00:00", array="1-5",
                            mem=None, wandb_name=None, wandb_group=None):
        folder = f"sail_sb3/{env}_sbatch"
        if env == "HalfCheetah":
            folder = "sail_sb3/HC_sbatch"
        elif env == "Walker2d":
            folder = "sail_sb3/Walker2d_sbatch"
        elif env == "Hopper":
            folder = "sail_sb3/Hopper_sbatch"
        elif env == "Ant":
            folder = "sail_sb3/Ant_sbatch"
        log_dir = f"{REPO_ROOT}/{folder}/logs"
        out = f"{log_dir}/{slug}_{variant_tag}_%A_%a.out"
        env_abbr = {"Ant": "Ant", "HalfCheetah": "HC", "Hopper": "Hop", "Walker2d": "W2d"}[env]
        job = f'{env_abbr}_{ep}s{sc}_{variant_tag}'[:32]
        hdr = (
            f'#!/bin/bash\n'
            f'#SBATCH --job-name={job}\n'
            f'#SBATCH --partition=standard\n'
            f'#SBATCH --nodes=1\n'
            f'#SBATCH --ntasks=1\n'
            f'#SBATCH --cpus-per-task=4\n'
            f'#SBATCH --mem={mem or d["mem"]}\n'
            f'#SBATCH --time={time}\n'
            f'#SBATCH --output={out}\n'
            f'#SBATCH --array={array}\n'
            f'#SBATCH --account=sabymath0\n'
        )
        wname = wandb_name or f'{slug}_{variant_tag}_s${{SEED}}'
        wgroup = wandb_group or f'{env_id}_{ep}_s{sc}_{variant_tag.lower()}'
        body = (
            f'\nset -euo pipefail\n\n'
            f'REPO_ROOT="{REPO_ROOT}"\n'
            f'PYTHON_BIN="{PY}"\n'
            f'EXPERT_DATA="${{REPO_ROOT}}/{d["npz"]}"\n'
            f'SEED=$SLURM_ARRAY_TASK_ID\n\n'
            f'mkdir -p "{log_dir}"\n\n'
            f'echo "========================================"\n'
            f'echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"\n'
            f'echo "Mode: {mode_desc}"\n'
            f'echo "Dataset: ${{EXPERT_DATA}}"\n'
            f'echo "========================================"\n\n'
            f'export CUDA_VISIBLE_DEVICES=""\n'
            f'export MUJOCO_GL=osmesa\n'
            f'export DISPLAY=\n'
            f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n'
            f'export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n\n'
            f'export WANDB_PROJECT="{d["wandb_proj"]}"\n'
            f'export WANDB_NAME="{wname}"\n'
            f'export WANDB_GROUP="{wgroup}"\n'
            f'export WANDB_SILENT=true\n\n'
            f'cd "${{REPO_ROOT}}"\n'
            f'PYTHONPATH="${{REPO_ROOT}}" \\\n'
            f'"{PY}" sail_sb3/scripts/train_sail.py \\\n'
            f'{flags_str}\n\n'
        )
        footer = (
            f'EXIT_CODE=$?\n'
            f'echo "========================================"; echo "END: $(date)"; '
            f'echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"\n'
            f'exit $EXIT_CODE\n'
        )
        fname = f"{REPO_ROOT}/{folder}/{slug}_{variant_tag}.sbatch"
        return fname, hdr + body + footer

    def make_sbatch_online(variant_tag, mode_desc, flags_str, time="12:00:00", array="1-5",
                           mem=None, wandb_name=None, wandb_group=None):
        folder = f"sail_sb3_online/{env}_sbatch"
        if env == "HalfCheetah":
            folder = "sail_sb3_online/HC_sbatch"
        elif env == "Walker2d":
            folder = "sail_sb3_online/Walker2d_sbatch"
        elif env == "Hopper":
            folder = "sail_sb3_online/Hopper_sbatch"
        elif env == "Ant":
            folder = "sail_sb3_online/Ant_sbatch"
        log_dir = f"{REPO_ROOT}/{folder}/logs"
        out = f"{log_dir}/{slug}_{variant_tag}_%A_%a.out"
        env_abbr = {"Ant": "Ant", "HalfCheetah": "HC", "Hopper": "Hop", "Walker2d": "W2d"}[env]
        job = f'{env_abbr}_{ep}s{sc}_{variant_tag}'[:32]
        hdr = (
            f'#!/bin/bash\n'
            f'#SBATCH --job-name={job}\n'
            f'#SBATCH --partition=standard\n'
            f'#SBATCH --nodes=1\n'
            f'#SBATCH --ntasks=1\n'
            f'#SBATCH --cpus-per-task=4\n'
            f'#SBATCH --mem={mem or "12G"}\n'
            f'#SBATCH --time={time}\n'
            f'#SBATCH --output={out}\n'
            f'#SBATCH --array={array}\n'
            f'#SBATCH --account=sabymath0\n'
        )
        wname = wandb_name or f'{slug}_{variant_tag}_s${{SEED}}'
        wgroup = wandb_group or f'{env_id}_{ep}_s{sc}_{variant_tag.lower()}'
        body = (
            f'\nset -euo pipefail\n\n'
            f'REPO_ROOT="{REPO_ROOT}"\n'
            f'PYTHON_BIN="{PY}"\n'
            f'EXPERT_DATA="${{REPO_ROOT}}/{d["npz"]}"\n'
            f'SEED=$SLURM_ARRAY_TASK_ID\n\n'
            f'mkdir -p "{log_dir}"\n\n'
            f'echo "========================================"\n'
            f'echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"\n'
            f'echo "Mode: {mode_desc}"\n'
            f'echo "Dataset: ${{EXPERT_DATA}}"\n'
            f'echo "========================================"\n\n'
            f'export CUDA_VISIBLE_DEVICES=""\n'
            f'export MUJOCO_GL=osmesa\n'
            f'export DISPLAY=\n'
            f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n'
            f'export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n\n'
            f'export WANDB_PROJECT="{d["wandb_proj_online"]}"\n'
            f'export WANDB_NAME="{wname}"\n'
            f'export WANDB_GROUP="{wgroup}"\n'
            f'export WANDB_SILENT=true\n\n'
            f'cd "${{REPO_ROOT}}"\n'
            f'PYTHONPATH="${{REPO_ROOT}}/sail_sb3_online" \\\n'
            f'"{PY}" sail_sb3_online/scripts/train_sail.py \\\n'
            f'{flags_str}\n\n'
        )
        footer = (
            f'EXIT_CODE=$?\n'
            f'echo "========================================"; echo "END: $(date)"; '
            f'echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"\n'
            f'exit $EXIT_CODE\n'
        )
        fname = f"{REPO_ROOT}/{folder}/{slug}_{variant_tag}.sbatch"
        return fname, hdr + body + footer

    def orm_block():
        return (
            f'  --online_rm \\\n'
            f'  --rm_segment_len 50 \\\n'
            f'  --rm_train_freq 1000 \\\n'
            f'  --rm_gradient_steps 10 \\\n'
            f'  --rm_batch_size 256 \\\n'
            f'  --rm_lr 3e-4 \\\n'
            f'  --rm_max_segments 10000 \\\n'
            f'  --rm_min_segments 500 \\\n'
            f'  --rm_min_updates 50 \\\n'
            f'  --rm_min_acc 0.60 \\\n'
            f'  --rm_tie_margin 0.5 \\\n'
            f'  --rm_rescore_freq 20000 \\\n'
        )

    rm = RM[env]
    pref_exp = f'  --pref_expect_obs_dim {d["pref_expect_obs_dim"]} \\\n' if d["pref_expect_obs_dim"] else ""

    # ═══════════════════════════════════════════════════════════════════════════
    # OFFLINE (sail_sb3) — 8 variants per dataset
    # ═══════════════════════════════════════════════════════════════════════════

    # 1. PAIL
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source gt \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --entcoeff {ec_base} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline("PAIL",
        f"PAIL (PrefReweight) + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 2. SAIL-PREFD / SAIL-PREFRANK
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_rank_disc \\\n'
        f'  --pref_rank_weight {d["pref_rank_weight"]} \\\n'
        f'  --pref_rank_batch_size {d["pref_rank_batch_size"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --entcoeff {ec_base} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline(f"SAIL_{pd_slug}",
        f"SAIL + {pd_name}-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 3. PAIL-PREFD / PAIL-PREFRANK
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source rm \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --pref_rank_disc \\\n'
        f'  --pref_rank_weight {d["pref_rank_weight"]} \\\n'
        f'  --pref_rank_batch_size {d["pref_rank_batch_size"]} \\\n'
        f'  --entcoeff {ec_base} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline(f"PAIL_{pd_slug}",
        f"PAIL + {pd_name}-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 4. SAIL-TAC-D
    ec_tac = 0.05  # TAC-D always uses 0.05 even for HC
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source {d["sail_tac_src_offline"]} \\\n'
        f'  --soft_tac \\\n'
        f'  --soft_tac_weight {d["sail_tac_weight"]} \\\n'
        f'  --soft_tac_temp 1.0 \\\n'
        f'  --tac_tie_eps 0.0 \\\n'
        f'  --soft_tac_max_student_trajs 200 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --entcoeff {ec_tac} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline("SAIL_TACD",
        f"SAIL + Soft-TAC-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 5. PAIL-TAC-D
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source {d["pail_tac_src_offline"]} \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --soft_tac \\\n'
        f'  --soft_tac_weight {d["pail_tac_weight"]} \\\n'
        f'  --soft_tac_temp 1.0 \\\n'
        f'  --tac_tie_eps 0.0 \\\n'
        f'  --soft_tac_max_student_trajs 200 \\\n'
        f'  --entcoeff {ec_tac} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline("PAIL_TACD",
        f"PAIL + Soft-TAC-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 6. SAIL-QPREF
    ec_qp = 0.05  # QPREF always 0.05
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source gt \\\n'
        f'  --qpref \\\n'
        f'  --qpref_weight {d["qpref_weight_sail"]} \\\n'
        f'  --qpref_source student \\\n'
        f'  --qpref_grad_interval 50 \\\n'
        f'  --qpref_guard_threshold -0.3 \\\n'
        f'  --qpref_guard_confirm 3 \\\n'
        f'  --qpref_guard_window 10 \\\n'
        f'  --qpref_guard_positive_threshold 1.0 \\\n'
        f'  --pref_max_student_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --entcoeff {ec_qp} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline("SAIL_QPREF",
        f"SAIL + QPREF (student source) + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 7. PAIL-QPREF
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source rm \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --qpref \\\n'
        f'  --qpref_weight {d["qpref_weight_pail"]} \\\n'
        f'  --qpref_source student \\\n'
        f'  --qpref_grad_interval 50 \\\n'
        f'  --qpref_guard_threshold -0.3 \\\n'
        f'  --qpref_guard_confirm 3 \\\n'
        f'  --qpref_guard_window 10 \\\n'
        f'  --qpref_guard_positive_threshold 1.0 \\\n'
        f'  --pref_max_student_trajs 500 \\\n'
        f'  --entcoeff {ec_qp} \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu'
    )
    p, c = make_sbatch_offline("PAIL_QPREF",
        f"PAIL + QPREF (student source) + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 8. SAIL-PREFD-DiscRewardAblation (array=1-3)
    flags = (
        f'DISC_REWARD_TYPE="airl_backward_kl"\n\n'
        f'REPO_ROOT="{REPO_ROOT}"\n'
        f'PYTHON_BIN="{PY}"\n'
        f'EXPERT_DATA="${{REPO_ROOT}}/{d["npz"]}"\n'
        f'SEED=$SLURM_ARRAY_TASK_ID\n'
    )  # special case — write manually
    # Build full content for DiscRewardAblation
    env_abbr = {"Ant": "Ant", "HalfCheetah": "HC", "Hopper": "Hop", "Walker2d": "W2d"}[env]
    job = f'{env_abbr}_{ep}s{sc}_SAILPD_DiscRew'[:32]
    folder_map = {"Ant": "sail_sb3/Ant_sbatch", "HalfCheetah": "sail_sb3/HC_sbatch",
                  "Hopper": "sail_sb3/Hopper_sbatch", "Walker2d": "sail_sb3/Walker2d_sbatch"}
    folder = folder_map[env]
    log_dir = f"{REPO_ROOT}/{folder}/logs"
    dra_content = (
        f'#!/bin/bash\n'
        f'#SBATCH --job-name={job}\n'
        f'#SBATCH --partition=standard\n'
        f'#SBATCH --nodes=1\n'
        f'#SBATCH --ntasks=1\n'
        f'#SBATCH --cpus-per-task=4\n'
        f'#SBATCH --mem={d["mem"]}\n'
        f'#SBATCH --time=24:00:00\n'
        f'#SBATCH --output={log_dir}/{slug}_SAIL_{pd_slug}_DiscRewAbl_%A_%a.out\n'
        f'#SBATCH --array=1-3\n'
        f'#SBATCH --account=sabymath0\n'
        f'\n'
        f'# Discriminator reward ablation — SAIL + {pd_name}-Disc ({env_id})\n'
        f'# DISC_REWARD_TYPE options: gail_js | airl_backward_kl | fairl_forward_kl | gail_heuristic\n'
        f'DISC_REWARD_TYPE="airl_backward_kl"\n\n'
        f'set -euo pipefail\n\n'
        f'REPO_ROOT="{REPO_ROOT}"\n'
        f'PYTHON_BIN="{PY}"\n'
        f'EXPERT_DATA="${{REPO_ROOT}}/{d["npz"]}"\n'
        f'SEED=$SLURM_ARRAY_TASK_ID\n\n'
        f'mkdir -p "{log_dir}"\n\n'
        f'echo "========================================"\n'
        f'echo "START: $(date)  JOB: ${{SLURM_JOB_ID}}  SEED: ${{SEED}}"\n'
        f'echo "Mode: SAIL + {pd_name}-Disc + Adaptive + LfD  [disc reward ablation] | {env_id}"\n'
        f'echo "disc_reward_type: ${{DISC_REWARD_TYPE}}"\n'
        f'echo "Dataset: ${{EXPERT_DATA}}"\n'
        f'echo "========================================"\n\n'
        f'export CUDA_VISIBLE_DEVICES=""\n'
        f'export MUJOCO_GL=osmesa\n'
        f'export DISPLAY=\n'
        f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n'
        f'export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"\n\n'
        f'export WANDB_PROJECT="{d["wandb_proj"]}"\n'
        f'export WANDB_NAME="{slug}_SAIL_{pd_slug}_discrew_${{DISC_REWARD_TYPE}}_s${{SEED}}"\n'
        f'export WANDB_GROUP="{env_id}_{ep}_s{sc}_sail_{pd_slug.lower()}_discrew_${{DISC_REWARD_TYPE}}"\n'
        f'export WANDB_SILENT=true\n\n'
        f'cd "${{REPO_ROOT}}"\n'
        f'PYTHONPATH="${{REPO_ROOT}}" \\\n'
        f'"{PY}" sail_sb3/scripts/train_sail.py \\\n'
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_rank_disc \\\n'
        f'  --pref_rank_weight {d["pref_rank_weight"]} \\\n'
        f'  --pref_rank_batch_size {d["discrew_batch"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rm "{rm}" \\\n'
        f'{pref_exp}'
        f'  --entcoeff {ec_base} \\\n'
        f'  --disc_reward_type "${{DISC_REWARD_TYPE}}" \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu\n\n'
        f'EXIT_CODE=$?\n'
        f'echo "========================================"; echo "END: $(date)"; '
        f'echo "EXIT CODE: ${{EXIT_CODE}}"; echo "========================================"\n'
        f'exit $EXIT_CODE\n'
    )
    dra_path = f"{REPO_ROOT}/{folder}/{slug}_SAIL_{pd_slug}_DiscRewAbl.sbatch"
    r = write_sbatch(dra_path, dra_content); created += r; skipped += (not r)

    # ═══════════════════════════════════════════════════════════════════════════
    # ONLINE (sail_sb3_online) — 6 variants per dataset (PAIL already exists)
    # ═══════════════════════════════════════════════════════════════════════════

    # 1. SAIL-PREFD / SAIL-PREFRANK (online)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_rank_disc \\\n'
        f'  --pref_rank_weight {d["pref_rank_weight"]} \\\n'
        f'  --pref_rank_batch_size {d["pref_rank_batch_size"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online(f"SAIL_{pd_slug}",
        f"Online RM + SAIL + {pd_name}-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 2. PAIL-PREFD / PAIL-PREFRANK (online)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --pref_rank_disc \\\n'
        f'  --pref_rank_weight {d["pref_rank_weight"]} \\\n'
        f'  --pref_rank_batch_size {d["pref_rank_batch_size"]} \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online(f"PAIL_{pd_slug}",
        f"Online RM + PAIL + {pd_name}-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 3. SAIL-TAC-D (online)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source {d["sail_tac_src_online"]} \\\n'
        f'  --soft_tac \\\n'
        f'  --soft_tac_weight {d["sail_tac_weight"]} \\\n'
        f'  --soft_tac_temp 1.0 \\\n'
        f'  --tac_tie_eps 0.0 \\\n'
        f'  --soft_tac_max_student_trajs 200 \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online("SAIL_TACD",
        f"Online RM + SAIL + Soft-TAC-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 4. PAIL-TAC-D (online)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --adaptive_score_source {d["pail_tac_src_online"]} \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --soft_tac \\\n'
        f'  --soft_tac_weight {d["pail_tac_weight"]} \\\n'
        f'  --soft_tac_temp 1.0 \\\n'
        f'  --tac_tie_eps 0.0 \\\n'
        f'  --soft_tac_max_student_trajs 200 \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online("PAIL_TACD",
        f"Online RM + PAIL + Soft-TAC-Disc + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 5. SAIL-QPREF (online — different qpref params: weight=0.1, batch=4, interval=10)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --qpref \\\n'
        f'  --qpref_source student \\\n'
        f'  --qpref_weight 0.1 \\\n'
        f'  --qpref_temp 1.0 \\\n'
        f'  --qpref_batch_size 4 \\\n'
        f'  --qpref_grad_interval 10 \\\n'
        f'  --qpref_guard_threshold -0.3 \\\n'
        f'  --qpref_guard_confirm 3 \\\n'
        f'  --qpref_guard_window 10 \\\n'
        f'  --qpref_guard_positive_threshold 1.0 \\\n'
        f'  --pref_max_student_trajs 500 \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online("SAIL_QPREF",
        f"Online RM + SAIL + QPREF (student source, RM-only J) + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

    # 6. PAIL-QPREF (online)
    flags = (
        f'  --env {env_id} \\\n'
        f'  --expert_data "${{EXPERT_DATA}}" \\\n'
        f'  --total_timesteps 1000000 \\\n'
        f'  --learning_starts 10000 \\\n'
        f'  --adaptive \\\n'
        f'  --lfd_mixing \\\n'
        f'  --teacher_buffer_size 1000 \\\n'
        f'  --pref_reweight_teacher \\\n'
        f'  --pref_beta {d["pref_beta"]} \\\n'
        f'  --pref_max_teacher_trajs 500 \\\n'
        f'  --qpref \\\n'
        f'  --qpref_source student \\\n'
        f'  --qpref_weight 0.1 \\\n'
        f'  --qpref_temp 1.0 \\\n'
        f'  --qpref_batch_size 4 \\\n'
        f'  --qpref_grad_interval 10 \\\n'
        f'  --qpref_guard_threshold -0.3 \\\n'
        f'  --qpref_guard_confirm 3 \\\n'
        f'  --qpref_guard_window 10 \\\n'
        f'  --qpref_guard_positive_threshold 1.0 \\\n'
        f'  --pref_max_student_trajs 500 \\\n'
        + orm_block() +
        f'  --entcoeff 0.05 \\\n'
        f'  --seed "${{SEED}}" \\\n'
        f'  --device cpu \\\n'
        f'  --debug'
    )
    p, c = make_sbatch_online("PAIL_QPREF",
        f"Online RM + PAIL + QPREF (student source, RM-only J) + Adaptive + LfD | {env_id}",
        flags)
    r = write_sbatch(p, c); created += r; skipped += (not r)

print(f"\n=== DONE: {created} created, {skipped} skipped (already exist) ===")
