"""
train_mw_teacher.py  —  Meta-World teacher training (Phase 1.1)
===============================================================
Standalone SAC teacher training for a single Meta-World MT1 task.
No dependencies on sail_sb3/ or sail_sb3_online/.

Outputs (all under --out_root)
-------------------------------
    checkpoints/
        step_{N:08d}/model.zip + meta.json
        best   -> symlink
        latest -> symlink
    training_log.csv
    run_config.json
    training_summary.json
    train.log   (mirror of stdout, written only when --log_to_file is set)

TimeFeatureWrapper
------------------
NOT applied here.  The teacher is trained on the raw 39-dim observation.
TimeFeatureWrapper is added at dataset-generation time (Phase 1.5) when
teacher rollouts are collected for SAIL.

Success rate
------------
Primary quality metric.  An episode is "successful" if info["success"] is
True at any point during the episode (Meta-World v2 convention).
Selection criterion for dataset generation: success_rate >= 0.90.

Algorithm
---------
SAC (SB3) — preferred over TD3 for Meta-World due to better sample
efficiency in short-horizon manipulation tasks.

Python
------
/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import numpy as np

# ---- Locate repo root so both mw_pipeline/ and SB3 imports resolve ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from mw_pipeline.utils.mw_env_utils import (
    make_mw_teacher_env,
    MW_EP_LEN,
    MW_OBS_DIM,
    MW_ACT_DIM,
)
from mw_pipeline.teacher_training.checkpoint_utils import (
    save_checkpoint,
    write_run_config,
    write_training_summary,
    print_checkpoint_table,
    list_checkpoints,
)

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(task_name: str, model, n_eval_episodes: int, seed: int) -> dict:
    """
    Roll out the model deterministically for n_eval_episodes episodes.

    Returns a dict with:
        mean_return, min_return, max_return  — undiscounted episodic return
        success_rate                          — fraction of episodes where
                                               info["success"] was True at
                                               any step (Meta-World v2 convention)
        n_eval_episodes
    """
    from mw_pipeline.utils.mw_env_utils import MetaWorldWrapper

    # Import metaworld lazily via the factory (error message is user-friendly)
    factory = make_mw_teacher_env(task_name, seed=seed)
    env = factory()  # wrapped: MetaWorldWrapper → Monitor

    returns    = []
    successes  = []

    for ep_idx in range(n_eval_episodes):
        obs   = env.reset()
        ep_ret   = 0.0
        ep_steps = 0
        done     = False
        ep_success = False

        while not done and ep_steps < MW_EP_LEN:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret    += float(reward)
            ep_steps  += 1
            if info.get("success", False):
                ep_success = True

        returns.append(ep_ret)
        successes.append(float(ep_success))

    env.close()
    arr = np.array(returns, dtype=np.float64)
    return {
        "mean_return":    float(arr.mean()),
        "min_return":     float(arr.min()),
        "max_return":     float(arr.max()),
        "success_rate":   float(np.mean(successes)),
        "n_eval_episodes": n_eval_episodes,
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Meta-World teacher policy with SAC (or TD3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Task / run identity ---
    p.add_argument("--task",     required=True,
                   help="Meta-World task name, e.g. reach-v2")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--algo",     choices=["sac", "td3"], default="sac",
                   help="RL algorithm (SAC recommended for Meta-World)")
    p.add_argument("--run_name", default=None,
                   help="Optional run name; defaults to {algo}_seed{seed}")

    # --- Training length ---
    p.add_argument("--total_timesteps", type=int, default=1_000_000,
                   help="Total environment steps")

    # --- Checkpointing ---
    p.add_argument("--out_root",        required=True,
                   help="Root output directory for this run")
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Save a checkpoint every N env steps")

    # --- Evaluation ---
    p.add_argument("--eval_freq",       type=int, default=10_000,
                   help="Evaluate policy every N env steps")
    p.add_argument("--n_eval_episodes", type=int, default=20,
                   help="Number of episodes per evaluation")

    # --- Milestone checkpoint ---
    p.add_argument("--target_success_rate", type=float, default=None,
                   help="Save an extra milestone checkpoint the first time "
                        "success_rate >= this value.  Disabled if not set.")

    # --- SAC hyperparameters ---
    p.add_argument("--learning_rate",   type=float, default=3e-4)
    p.add_argument("--buffer_size",     type=int,   default=1_000_000)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--learning_starts", type=int,   default=1_000)
    p.add_argument("--tau",             type=float, default=0.005)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--train_freq",      type=int,   default=1)
    p.add_argument("--gradient_steps",  type=int,   default=1)
    p.add_argument("--ent_coef",        default="auto",
                   help="SAC entropy coefficient ('auto' or float)")

    # --- Logging ---
    p.add_argument("--wandb",        action="store_true",
                   help="Enable W&B logging (requires wandb installed)")
    p.add_argument("--wandb_project",default="mw-teacher-training")
    p.add_argument("--log_to_file",  action="store_true",
                   help="Mirror stdout to out_root/train.log")
    p.add_argument("--device",       default="auto",
                   help="PyTorch device: 'auto', 'cpu', 'cuda'")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = build_parser().parse_args()
    t_start = time.time()

    if args.run_name is None:
        args.run_name = f"{args.algo}_seed{args.seed}"

    out_root = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_root, exist_ok=True)

    # ---- Logging setup ----
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_to_file:
        log_file = os.path.join(out_root, "train.log")
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger(__name__)

    # ---- Persist run config ----
    run_cfg = vars(args)
    run_cfg["out_root_resolved"] = out_root
    write_run_config(run_cfg, out_root)

    # ---- W&B ----
    if args.wandb:
        if not WANDB_AVAILABLE:
            log.warning("W&B requested but not installed — skipping.")
            args.wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.task}_{args.run_name}",
                config=run_cfg,
            )

    # ---- Seeding ----
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Build training environment ----
    log.info("=" * 68)
    log.info(f"Meta-World Teacher Training  [{args.algo.upper()}]")
    log.info(f"  Task            : {args.task}")
    log.info(f"  Seed            : {args.seed}")
    log.info(f"  Total steps     : {args.total_timesteps:,}")
    log.info(f"  Checkpoint freq : {args.checkpoint_freq:,} steps")
    log.info(f"  Eval freq       : {args.eval_freq:,} steps")
    log.info(f"  Eval episodes   : {args.n_eval_episodes}")
    log.info(f"  Target SR       : {args.target_success_rate}")
    log.info(f"  Obs dim         : {MW_OBS_DIM}  (raw, no TimeFeatureWrapper)")
    log.info(f"  Act dim         : {MW_ACT_DIM}")
    log.info(f"  Ep length       : {MW_EP_LEN}")
    log.info(f"  Output          : {out_root}")
    log.info("=" * 68)

    train_env = DummyVecEnv([make_mw_teacher_env(args.task, seed=args.seed)])

    # ---- Build model ----
    if args.algo == "sac":
        ent_coef = args.ent_coef
        # If the user passed a numeric string, convert to float so SB3 accepts it
        if ent_coef != "auto":
            try:
                ent_coef = float(ent_coef)
            except ValueError:
                pass
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=args.seed,
            device=args.device,
        )
    else:  # td3
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model = TD3(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=args.seed,
            device=args.device,
        )

    # ---- Chunk-based training loop ----
    best_success_rate  = -1.0
    best_meta          = {}
    target_triggered   = False
    last_ckpt_step     = 0
    last_eval_step     = 0

    # Determine chunk boundaries — evaluate and (optionally) checkpoint at each.
    # We use eval_freq as the inner loop step size, then checkpoint at every
    # checkpoint_freq (which must be a multiple of eval_freq, or we checkpoint
    # whenever step >= last_ckpt_step + checkpoint_freq at an eval point).
    chunk_size = min(args.eval_freq, args.checkpoint_freq)
    steps_remaining = args.total_timesteps
    total_trained   = 0
    chunk_idx       = 0

    while total_trained < args.total_timesteps:
        steps_this_chunk = min(chunk_size, args.total_timesteps - total_trained)
        reset_timesteps  = (total_trained == 0)

        model.learn(
            total_timesteps=steps_this_chunk,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )
        total_trained += steps_this_chunk
        chunk_idx     += 1

        # ---- Evaluate ----
        if total_trained - last_eval_step >= args.eval_freq or total_trained >= args.total_timesteps:
            last_eval_step = total_trained
            elapsed = (time.time() - t_start) / 3600.0

            stats = evaluate(args.task, model, args.n_eval_episodes, seed=args.seed)
            stats["wall_time_hours"] = elapsed

            is_best = stats["success_rate"] > best_success_rate
            if is_best:
                best_success_rate = stats["success_rate"]

            log.info(
                f"  step={total_trained:>8,}  "
                f"SR={stats['success_rate']:.3f}  "
                f"ret={stats['mean_return']:.1f} "
                f"[{stats['min_return']:.1f},{stats['max_return']:.1f}]  "
                f"{'[BEST] ' if is_best else ''}"
                f"elapsed={elapsed*60:.1f}min"
            )

            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "eval/success_rate":  stats["success_rate"],
                    "eval/mean_return":   stats["mean_return"],
                    "eval/min_return":    stats["min_return"],
                    "eval/max_return":    stats["max_return"],
                    "eval/wall_time_h":   elapsed,
                }, step=total_trained)

            # ---- Checkpoint ----
            should_ckpt = (
                total_trained - last_ckpt_step >= args.checkpoint_freq
                or total_trained >= args.total_timesteps
            )
            if should_ckpt:
                last_ckpt_step = total_trained
                ckpt_dir = save_checkpoint(
                    model=model,
                    step=total_trained,
                    eval_stats=stats,
                    run_cfg=vars(args),
                    out_root=out_root,
                    is_best=is_best,
                )
                log.info(f"  [ckpt] saved → {ckpt_dir}")
                if is_best:
                    best_meta = dict(stats)
                    best_meta["step"] = total_trained
                    best_meta["ckpt_dir"] = ckpt_dir
                    best_meta["model_path"] = os.path.join(ckpt_dir, "model.zip")

            # ---- Milestone checkpoint ----
            if (
                args.target_success_rate is not None
                and not target_triggered
                and stats["success_rate"] >= args.target_success_rate
            ):
                target_triggered = True
                milestone_name = (
                    f"target_sr{args.target_success_rate:.2f}"
                    f"_step{total_trained:08d}"
                )
                milestone_dir = os.path.join(out_root, "checkpoints", milestone_name)
                os.makedirs(milestone_dir, exist_ok=True)
                model.save(os.path.join(milestone_dir, "model"))
                milestone_meta = _build_milestone_meta(
                    total_trained, stats, vars(args), args.target_success_rate
                )
                with open(os.path.join(milestone_dir, "meta.json"), "w") as fh:
                    json.dump(milestone_meta, fh, indent=2)
                log.info(
                    f"  [MILESTONE] SR={stats['success_rate']:.3f} >= "
                    f"{args.target_success_rate:.2f} → {milestone_dir}"
                )

    train_env.close()

    # ---- Final summary ----
    total_wall = (time.time() - t_start) / 3600.0

    # Refresh best_meta from disk if we never explicitly set it
    # (can happen if checkpoint_freq > total_timesteps)
    if not best_meta:
        ckpts = list_checkpoints(out_root)
        if ckpts:
            best_meta = max(ckpts, key=lambda c: c.get("success_rate", -1.0))

    write_training_summary(out_root, best_meta, total_wall)

    log.info("\n" + "=" * 68)
    log.info("Training complete.")
    log.info(f"  Task            : {args.task}")
    log.info(f"  Best SR         : {best_success_rate:.4f}")
    log.info(f"  Best ckpt       : {out_root}/checkpoints/best")
    log.info(f"  Total time      : {total_wall:.2f}h")
    log.info(f"  Summary         : {out_root}/training_summary.json")
    if args.target_success_rate is not None:
        status = "reached" if target_triggered else "NOT reached"
        log.info(f"  Target SR={args.target_success_rate:.2f} : {status}")
    log.info("=" * 68)

    log.info("\nCheckpoint table (sorted by success_rate):")
    print_checkpoint_table(out_root, sort_by="success_rate")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


def _build_milestone_meta(step, stats, run_cfg, target_sr):
    """Build meta.json payload for the milestone (target success rate) checkpoint."""
    import subprocess as _sp, time as _t
    try:
        gh = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        gh = "unknown"
    return {
        "task":                run_cfg.get("task", "unknown"),
        "algo":                run_cfg.get("algo", "sac"),
        "seed":                run_cfg.get("seed", 0),
        "step":                step,
        "mean_return":         round(float(stats["mean_return"]),  4),
        "min_return":          round(float(stats["min_return"]),   4),
        "max_return":          round(float(stats["max_return"]),   4),
        "success_rate":        round(float(stats["success_rate"]), 4),
        "n_eval_episodes":     int(stats["n_eval_episodes"]),
        "wall_time_hours":     round(float(stats.get("wall_time_hours", 0.0)), 4),
        "model_file":          "model.zip",
        "milestone_target_sr": float(target_sr),
        "milestone_note":      "saved when success_rate first reached target_success_rate",
        "git_hash":            gh,
        "saved_at":            _t.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    main()
