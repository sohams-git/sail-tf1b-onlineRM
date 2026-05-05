"""
eval_mw_teacher.py  —  Evaluate a saved Meta-World teacher checkpoint
======================================================================
Standalone script. No dependencies on sail_sb3/ or sail_sb3_online/.

Loads a saved SAC/TD3 checkpoint (model.zip) and rolls out the policy
deterministically on the corresponding Meta-World task.

Reports per-episode and aggregate:
  - mean / min / max episodic return
  - success rate  (episode successful if info["success"] True at any step)
  - mean episode length
  - checkpoint path / task / step

Saves results to --out_dir / eval_results.json (and prints to stdout).

Usage
-----
python mw_pipeline/teacher_training/eval_mw_teacher.py \\
    --task reach-v2 \\
    --checkpoint teacher_training/metaworld/reach-v2/sac_seed0/checkpoints/step_00250000 \\
    --n_eval_episodes 50 \\
    --out_dir teacher_training/metaworld/reach-v2/teacher_eval/step_00250000
"""

import os
import sys
import json
import argparse
import time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import SAC, TD3
from mw_pipeline.utils.mw_env_utils import make_mw_teacher_env, MW_EP_LEN


def evaluate(task: str, model, n_episodes: int, seed: int) -> dict:
    factory = make_mw_teacher_env(task, seed=seed)
    env = factory()

    returns, ep_lens, successes = [], [], []

    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret, ep_len, done, ep_success = 0.0, 0, False, False
        while not done and ep_len < MW_EP_LEN:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            if info.get("success", False):
                ep_success = True
        returns.append(ep_ret)
        ep_lens.append(ep_len)
        successes.append(float(ep_success))

    env.close()
    arr = np.array(returns)
    return {
        "mean_return":    float(arr.mean()),
        "min_return":     float(arr.min()),
        "max_return":     float(arr.max()),
        "std_return":     float(arr.std()),
        "success_rate":   float(np.mean(successes)),
        "mean_ep_length": float(np.mean(ep_lens)),
        "n_eval_episodes": n_episodes,
    }


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",           required=True)
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint directory (contains model.zip) or model.zip itself")
    p.add_argument("--algo",           choices=["sac", "td3"], default="sac")
    p.add_argument("--n_eval_episodes",type=int, default=50)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--out_dir",        default=None,
                   help="Directory to save eval_results.json. Defaults to <checkpoint>/eval/")
    p.add_argument("--device",         default="cpu")
    args = p.parse_args()

    # Resolve model path
    ckpt = args.checkpoint
    model_path = os.path.join(ckpt, "model.zip") if os.path.isdir(ckpt) else ckpt
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model.zip not found at {model_path}")

    # Infer step from directory name if possible
    ckpt_name = os.path.basename(ckpt.rstrip("/"))
    step_str = ckpt_name.replace("step_", "").split("_")[0]
    try:
        step = int(step_str)
    except ValueError:
        step = -1

    # Load model
    cls = SAC if args.algo == "sac" else TD3
    print(f"Loading {args.algo.upper()} from {model_path} ...")
    model = cls.load(model_path, device=args.device)

    # Evaluate
    print(f"Evaluating {args.task} for {args.n_eval_episodes} episodes ...")
    t0 = time.time()
    stats = evaluate(args.task, model, args.n_eval_episodes, args.seed)
    elapsed = time.time() - t0

    # Build result record
    result = {
        "task":            args.task,
        "checkpoint_path": model_path,
        "checkpoint_step": step,
        "algo":            args.algo,
        "seed":            args.seed,
        **stats,
        "eval_wall_time_s": round(elapsed, 2),
        "evaluated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Print
    print("\n" + "=" * 60)
    print(f"  Task             : {args.task}")
    print(f"  Checkpoint step  : {step:,}" if step >= 0 else f"  Checkpoint       : {model_path}")
    print(f"  Episodes         : {args.n_eval_episodes}")
    print(f"  Success rate     : {stats['success_rate']:.4f}")
    print(f"  Mean return      : {stats['mean_return']:.2f}  "
          f"[{stats['min_return']:.1f}, {stats['max_return']:.1f}]")
    print(f"  Mean ep length   : {stats['mean_ep_length']:.1f}")
    print(f"  Eval time        : {elapsed:.1f}s")
    print("=" * 60)

    # Save
    out_dir = args.out_dir or os.path.join(ckpt, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
