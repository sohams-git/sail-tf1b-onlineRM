"""
train_halfcheetah_teacher_checkpoints.py  (SB3 isolated version)
=================================================================
Fully isolated TD3 teacher training for HalfCheetah-v2 using
stable_baselines3 (PyTorch).  Zero dependencies on:
  - stable-baselines/ TF1 code or shared TD3 files
  - sail_sb3/ SAIL implementation
  - any shared replay buffers

Environment
-----------
HalfCheetah-v2 wrapped with TimeFeatureWrapper (17 → 18-dim obs).
This MUST match the existing SAIL teacher datasets which were all
collected with TimeFeatureWrapper applied (confirmed by inspecting
the live .npz files: obs shape (N, 18), native gym gives 17-dim).

TimeFeatureWrapper formula (identical to both TF1 and SB3 SAIL code):
  time_feature = 1.0 - (current_step / max_steps)   → appended to obs

Algorithm
---------
TD3 via stable_baselines3.TD3 with hyperparameters matching the
existing teacher datasets:
  policy_kwargs = net_arch=[400, 300]
  buffer_size   = 1_000_000
  batch_size    = 256
  learning_rate = 1e-3
  gamma         = 0.99
  train_freq    = 1
  gradient_steps= 1
  action_noise  = NormalActionNoise(sigma=0.1)

Checkpoints
-----------
  - Regular checkpoint every --checkpoint_every episodes (default 100)
  - Special milestone checkpoint when eval return first reaches or
    exceeds --target_return (default 6988).
    Rule: OPTION 2 — trigger once when mean_eval_return >= target_return

Target-return trigger rule (OPTION 2)
  Chosen because: in a continuous training loop the exact value may
  never be hit; "first crossing ≥ threshold" is well-defined and
  reproducible for a given seed.  Saved exactly once.

Outputs (all under --save_root/halfcheetah/)
  teacher_checkpoints/
    ep{NNNN}_return{RRR}/  model.zip + meta.json
    target{T}_reached_ep{NNNN}_return{RRR}/  model.zip + meta.json
    best    -> symlink
    latest  -> symlink
  teacher_logs/
    training_log.csv

Python
------
/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
"""

import os
import json
import csv
import random
import argparse
import time
import numpy as np
import gym

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

# HalfCheetah-v2 episode horizon
EP_LEN = 1000


# ---------------------------------------------------------------------------
# TimeFeatureWrapper — self-contained, no imports from sail_sb3 or TF1 code
# Formula identical to both stable-baselines/utils/wrappers.py and
# sail_sb3/scripts/train_sail.py TimeFeatureWrapper.
# ---------------------------------------------------------------------------

class TimeFeatureWrapper(gym.Wrapper):
    """
    Appends remaining-time feature to observations:
        obs_new = concat(obs, [1.0 - step/max_steps])
    HalfCheetah-v2: 17-dim native → 18-dim with wrapper.
    The existing SAIL teacher datasets have 18-dim obs, confirming they
    were collected with this wrapper applied.
    """
    def __init__(self, env, max_steps: int = EP_LEN):
        super().__init__(env)
        low  = np.concatenate([env.observation_space.low,  [0.0]])
        high = np.concatenate([env.observation_space.high, [1.0]])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)
        self._max_steps    = max_steps
        self._current_step = 0

    def reset(self):
        self._current_step = 0
        obs = self.env.reset()
        # gym ≥0.26 returns (obs, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._add_time(obs)

    def step(self, action):
        self._current_step += 1
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        return self._add_time(obs), reward, done, info

    def _add_time(self, obs):
        t = 1.0 - self._current_step / self._max_steps
        return np.concatenate([obs, [t]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Gym compatibility helper (raw env, pre-TimeFeatureWrapper)
# ---------------------------------------------------------------------------

def _raw_reset(env):
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret


# ---------------------------------------------------------------------------
# Evaluation using TimeFeatureWrapper
# ---------------------------------------------------------------------------

def evaluate(model, env_id, n_eval_episodes):
    """
    Run n_eval_episodes deterministic episodes with TimeFeatureWrapper.
    Returns (mean_return, min_return, max_return).
    """
    returns = []
    for _ in range(n_eval_episodes):
        env = TimeFeatureWrapper(gym.make(env_id))
        obs = env.reset()
        ep_ret, ep_len, done = 0.0, 0, False
        while not done and ep_len < EP_LEN:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
        env.close()
        returns.append(ep_ret)
    arr = np.array(returns, dtype=np.float64)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_name(episode, mean_return):
    return f"ep{episode:04d}_return{mean_return:.0f}"


def _target_name(target, episode, mean_return):
    return f"target{target:.0f}_reached_ep{episode:04d}_return{mean_return:.0f}"


def _save_checkpoint(model, ckpt_dir, episode, mean_ret, min_ret, max_ret,
                     seed, env_id, is_best, extra_meta=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "model"))   # → model.zip
    meta = {
        "episode":       episode,
        "timestep":      episode * EP_LEN,
        "mean_return":   round(float(mean_ret), 4),
        "min_return":    round(float(min_ret),  4),
        "max_return":    round(float(max_ret),  4),
        "seed":          seed,
        "env_id":        env_id,
        "is_best":       bool(is_best),
        "obs_dim":       18,
        "note":          "TimeFeatureWrapper applied (17+1=18 dim)",
        "checkpoint_dir": os.path.basename(ckpt_dir),
        "saved_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _update_symlinks(ckpt_root, ckpt_dir, is_best):
    for name, update in [("latest", True), ("best", is_best)]:
        link = os.path.join(ckpt_root, name)
        if os.path.islink(link):
            os.unlink(link)
        if update:
            os.symlink(os.path.abspath(ckpt_dir), link)


def _append_csv(log_dir, episode, mean_ret, min_ret, max_ret, is_best):
    path = os.path.join(log_dir, "training_log.csv")
    header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["episode", "timestep", "mean_return",
                        "min_return", "max_return", "is_best", "wall_time"])
        w.writerow([episode, episode * EP_LEN,
                    f"{mean_ret:.4f}", f"{min_ret:.4f}", f"{max_ret:.4f}",
                    int(is_best), time.strftime("%Y-%m-%d %H:%M:%S")])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root", required=True,
                    help="All outputs go under {save_root}/halfcheetah/")
    ap.add_argument("--env_id",   default="HalfCheetah-v2")
    ap.add_argument("--total_episodes",    type=int,   default=2000)
    ap.add_argument("--checkpoint_every",  type=int,   default=100)
    ap.add_argument("--n_eval_episodes",   type=int,   default=10)
    ap.add_argument("--target_return",     type=float, default=6988.0,
                    help="Trigger rule OPTION 2: save once when mean eval return "
                         "first reaches or exceeds this value.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- Directories ----
    run_root  = os.path.join(args.save_root, "halfcheetah")
    ckpt_root = os.path.join(run_root, "teacher_checkpoints")
    log_dir   = os.path.join(run_root, "teacher_logs")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save run config
    cfg = dict(vars(args))
    cfg["wrapper"]      = "TimeFeatureWrapper (17->18 dim, 1-step/max_steps)"
    cfg["algorithm"]    = "TD3 via stable_baselines3 (isolated, no TF1/sail_sb3 deps)"
    cfg["target_rule"]  = "OPTION 2: first time mean_eval_return >= target_return"
    cfg["started_at"]   = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run_root, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 68)
    print("HalfCheetah-v2  Teacher Training  (SB3 TD3, isolated)")
    print(f"  Total episodes   : {args.total_episodes}  ({args.total_episodes * EP_LEN:,} steps)")
    print(f"  Checkpoint every : {args.checkpoint_every} episodes")
    print(f"  Eval episodes    : {args.n_eval_episodes}")
    print(f"  Target return    : {args.target_return}  (OPTION 2: first >= threshold)")
    print(f"  Seed             : {args.seed}")
    print(f"  Obs dim          : 18 (TimeFeatureWrapper applied)")
    print(f"  Checkpoints      : {ckpt_root}")
    print(f"  Logs             : {log_dir}")
    print("=" * 68)

    # ---- Reproducibility ----
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Build SB3 TD3 model ----
    # Training env: TimeFeatureWrapper on top of raw HalfCheetah-v2
    train_env = TimeFeatureWrapper(gym.make(args.env_id))
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )
    model = TD3(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[400, 300]),
        buffer_size=1_000_000,
        batch_size=256,
        learning_rate=1e-3,
        action_noise=action_noise,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=args.seed,
    )

    # ---- Training loop ----
    best_mean_return = -np.inf
    target_triggered = False
    t_start = time.time()

    chunks = list(range(0, args.total_episodes, args.checkpoint_every))
    n_chunks = len(chunks)

    for idx, chunk_start in enumerate(chunks):
        chunk_end   = min(chunk_start + args.checkpoint_every, args.total_episodes)
        chunk_steps = (chunk_end - chunk_start) * EP_LEN

        print(f"\n[Chunk {idx+1}/{n_chunks}]  ep {chunk_start+1}–{chunk_end}"
              f"  ({chunk_steps:,} steps) ...", flush=True)

        model.learn(
            chunk_steps,
            reset_num_timesteps=(idx == 0),
            progress_bar=False,
        )
        ep = chunk_end

        # ---- Eval ----
        mean_r, min_r, max_r = evaluate(model, args.env_id, args.n_eval_episodes)
        is_best = mean_r > best_mean_return
        if is_best:
            best_mean_return = mean_r

        elapsed = time.time() - t_start
        print(f"  ep={ep:4d}  mean/min/max = {mean_r:.1f}/{min_r:.1f}/{max_r:.1f}"
              f"  {'[BEST] ' if is_best else ''}"
              f"elapsed={elapsed:.0f}s", flush=True)

        # ---- Regular checkpoint ----
        name = _ckpt_name(ep, mean_r)
        ckpt_dir = os.path.join(ckpt_root, name)
        _save_checkpoint(model, ckpt_dir, ep, mean_r, min_r, max_r,
                         args.seed, args.env_id, is_best)
        _update_symlinks(ckpt_root, ckpt_dir, is_best)
        _append_csv(log_dir, ep, mean_r, min_r, max_r, is_best)
        print(f"  [ckpt] {name}", flush=True)

        # ---- Target-return checkpoint (OPTION 2: first time >= target) ----
        if not target_triggered and mean_r >= args.target_return:
            target_triggered = True
            tgt_name = _target_name(args.target_return, ep, mean_r)
            tgt_dir  = os.path.join(ckpt_root, tgt_name)
            extra = {
                "target_return":    float(args.target_return),
                "actual_return":    round(float(mean_r), 4),
                "trigger_rule":     "OPTION 2: first time mean eval return >= target_return",
                "trigger_episode":  ep,
                "trigger_timestep": ep * EP_LEN,
            }
            _save_checkpoint(model, tgt_dir, ep, mean_r, min_r, max_r,
                             args.seed, args.env_id, is_best, extra_meta=extra)
            print(f"  [TARGET] *** {mean_r:.1f} >= {args.target_return:.0f} ***", flush=True)
            print(f"  [TARGET] Saved: {tgt_name}", flush=True)

    train_env.close()

    print("\n" + "=" * 68)
    print("Training complete.")
    print(f"  Best mean return : {best_mean_return:.1f}")
    print(f"  Best checkpoint  : {ckpt_root}/best")
    print(f"  Latest checkpoint: {ckpt_root}/latest")
    print(f"  Training log     : {log_dir}/training_log.csv")
    if target_triggered:
        print(f"  Target {args.target_return:.0f} ckpt : "
              f"{ckpt_root}/target{args.target_return:.0f}_reached_*")
    else:
        print(f"  Target {args.target_return:.0f}      : NOT reached in "
              f"{args.total_episodes} episodes")
    print(f"  Total time       : {(time.time() - t_start)/3600:.2f}h")
    print("=" * 68)


if __name__ == "__main__":
    main()
