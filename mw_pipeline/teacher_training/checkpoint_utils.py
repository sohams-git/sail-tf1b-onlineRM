"""
checkpoint_utils.py  —  Meta-World teacher checkpoint helpers
=============================================================
Standalone — no dependencies on sail_sb3/ or sail_sb3_online/.

Directory layout produced by save_checkpoint()
-----------------------------------------------
{out_root}/
    checkpoints/
        step_{N:08d}/
            model.zip       ← SB3 model.save() output (SAC or TD3)
            meta.json       ← checkpoint metadata (see _build_meta())
        step_{M:08d}/
            ...
        best   -> symlink to the checkpoint with highest success_rate
        latest -> symlink to the most recently saved checkpoint
    training_log.csv        ← one row per checkpoint event
    run_config.json         ← full argparse namespace saved at training start
    training_summary.json   ← written at end of training

Public API
----------
save_checkpoint(model, step, eval_stats, run_cfg, out_root, is_best)
    Save model.zip + meta.json under checkpoints/step_{step:08d}/.
    Also updates "best" / "latest" symlinks and appends to training_log.csv.

load_best_checkpoint(out_root, metric="success_rate") -> (model_path, meta)
    Scan all meta.json files under out_root/checkpoints/ and return the path
    to model.zip for the checkpoint with the highest value of `metric`.

list_checkpoints(out_root) -> list[dict]
    Return a list of dicts, each containing "step", "model_path", and the
    full contents of meta.json, sorted by step ascending.

write_run_config(run_cfg_dict, out_root)
    Persist the argparse namespace as run_config.json at training start.

write_training_summary(out_root, best_meta, total_wall_time_hours)
    Write training_summary.json at end of training.
"""

import os
import csv
import json
import time
import subprocess
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return short git hash of HEAD, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _ckpt_dir_name(step: int) -> str:
    return f"step_{step:08d}"


def _build_meta(step: int, eval_stats: dict, run_cfg: dict) -> dict:
    """
    Construct the meta.json payload for a checkpoint.

    eval_stats must contain:
        mean_return    (float)
        min_return     (float)
        max_return     (float)
        success_rate   (float, 0–1)
        n_eval_episodes (int)

    run_cfg must contain at minimum:
        task, algo, seed
    """
    return {
        "task":             run_cfg.get("task", "unknown"),
        "algo":             run_cfg.get("algo", "sac"),
        "seed":             run_cfg.get("seed", 0),
        "step":             step,
        "mean_return":      round(float(eval_stats["mean_return"]),   4),
        "min_return":       round(float(eval_stats["min_return"]),    4),
        "max_return":       round(float(eval_stats["max_return"]),    4),
        "success_rate":     round(float(eval_stats["success_rate"]),  4),
        "n_eval_episodes":  int(eval_stats["n_eval_episodes"]),
        "wall_time_hours":  round(float(eval_stats.get("wall_time_hours", 0.0)), 4),
        "model_file":       "model.zip",
        "git_hash":         _git_hash(),
        "saved_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _update_symlinks(ckpt_root: str, ckpt_dir: str, is_best: bool) -> None:
    """Update 'latest' symlink always; update 'best' symlink only when is_best."""
    for name, should_update in [("latest", True), ("best", is_best)]:
        if not should_update:
            continue
        link_path = os.path.join(ckpt_root, name)
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.symlink(os.path.abspath(ckpt_dir), link_path)


def _append_csv(log_path: str, step: int, eval_stats: dict, is_best: bool) -> None:
    header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow([
                "step", "mean_return", "min_return", "max_return",
                "success_rate", "n_eval_episodes", "is_best", "wall_time",
            ])
        w.writerow([
            step,
            f"{eval_stats['mean_return']:.4f}",
            f"{eval_stats['min_return']:.4f}",
            f"{eval_stats['max_return']:.4f}",
            f"{eval_stats['success_rate']:.4f}",
            int(eval_stats["n_eval_episodes"]),
            int(is_best),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    step: int,
    eval_stats: dict,
    run_cfg: dict,
    out_root: str,
    is_best: bool,
) -> str:
    """
    Save model.zip + meta.json for the given training step.

    Parameters
    ----------
    model      : SB3 SAC or TD3 instance (has .save() method)
    step       : current env step count
    eval_stats : dict with keys: mean_return, min_return, max_return,
                 success_rate, n_eval_episodes, wall_time_hours
    run_cfg    : dict representation of the argparse namespace (task, algo, seed, ...)
    out_root   : root output directory for this training run
    is_best    : True if this checkpoint has the highest success_rate seen so far

    Returns
    -------
    ckpt_dir   : absolute path to the saved checkpoint directory
    """
    ckpt_root = os.path.join(out_root, "checkpoints")
    ckpt_dir  = os.path.join(ckpt_root, _ckpt_dir_name(step))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save SB3 model (writes model.zip)
    model.save(os.path.join(ckpt_dir, "model"))

    # Save metadata
    meta = _build_meta(step, eval_stats, run_cfg)
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Symlinks + CSV
    _update_symlinks(ckpt_root, ckpt_dir, is_best)
    _append_csv(os.path.join(out_root, "training_log.csv"), step, eval_stats, is_best)

    return ckpt_dir


def load_best_checkpoint(out_root: str, metric: str = "success_rate"):
    """
    Scan all checkpoints under out_root/checkpoints/, return (model_path, meta)
    for the one with the highest value of `metric`.

    Parameters
    ----------
    out_root : training run root directory (contains checkpoints/ subdirectory)
    metric   : key in meta.json to rank by (default: "success_rate")

    Returns
    -------
    (model_path, meta)  where model_path is the absolute path to model.zip
    Raises FileNotFoundError if no checkpoints exist.
    """
    checkpoints = list_checkpoints(out_root)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {out_root}/checkpoints/")

    best = max(checkpoints, key=lambda c: c.get(metric, float("-inf")))
    return best["model_path"], best


def list_checkpoints(out_root: str) -> list:
    """
    Return a list of dicts for every checkpoint under out_root/checkpoints/,
    sorted by step ascending.  Each dict contains all keys from meta.json plus:
        "model_path" : absolute path to model.zip
        "ckpt_dir"   : absolute path to the checkpoint directory

    Silently skips directories that do not contain a valid meta.json.
    """
    ckpt_root = os.path.join(out_root, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return []

    results = []
    for entry in os.scandir(ckpt_root):
        if not entry.is_dir():
            continue
        meta_path  = os.path.join(entry.path, "meta.json")
        model_path = os.path.join(entry.path, "model.zip")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        meta["model_path"] = model_path
        meta["ckpt_dir"]   = entry.path
        results.append(meta)

    results.sort(key=lambda c: c.get("step", 0))
    return results


def write_run_config(run_cfg_dict: dict, out_root: str) -> None:
    """Persist the training configuration to run_config.json at run start."""
    os.makedirs(out_root, exist_ok=True)
    cfg = dict(run_cfg_dict)
    cfg["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg["git_hash"] = _git_hash()
    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_training_summary(
    out_root: str,
    best_meta: dict,
    total_wall_time_hours: float,
) -> None:
    """
    Write training_summary.json at the end of training.
    Includes the best checkpoint path and key quality metrics.
    """
    summary = {
        "best_checkpoint_dir":  best_meta.get("ckpt_dir", "unknown"),
        "best_model_path":      best_meta.get("model_path", "unknown"),
        "best_step":            best_meta.get("step", -1),
        "best_success_rate":    best_meta.get("success_rate", None),
        "best_mean_return":     best_meta.get("mean_return", None),
        "total_wall_time_hours": round(float(total_wall_time_hours), 4),
        "saved_at":             time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_root, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Pretty-print helper (used by select_checkpoint.py)
# ---------------------------------------------------------------------------

def print_checkpoint_table(out_root: str, sort_by: str = "success_rate") -> None:
    """
    Print a human-readable table of all checkpoints sorted by `sort_by`.
    Useful for manual inspection when selecting a checkpoint for dataset generation.
    """
    checkpoints = list_checkpoints(out_root)
    if not checkpoints:
        print(f"No checkpoints found in {out_root}")
        return

    checkpoints_sorted = sorted(
        checkpoints, key=lambda c: c.get(sort_by, float("-inf")), reverse=True
    )

    header = f"{'Step':>10}  {'SuccessRate':>12}  {'MeanReturn':>12}  {'MinReturn':>10}  {'MaxReturn':>10}  {'Saved At'}"
    print(header)
    print("-" * len(header))
    for c in checkpoints_sorted:
        sr   = c.get("success_rate", float("nan"))
        mr   = c.get("mean_return",  float("nan"))
        lo   = c.get("min_return",   float("nan"))
        hi   = c.get("max_return",   float("nan"))
        step = c.get("step", -1)
        ts   = c.get("saved_at", "")
        print(f"{step:>10}  {sr:>12.4f}  {mr:>12.2f}  {lo:>10.2f}  {hi:>10.2f}  {ts}")
