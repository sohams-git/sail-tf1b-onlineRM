"""
train_teacher_checkpoints.py  (SB3 isolated, all MuJoCo envs)
==============================================================
Generalized TD3 teacher training for any MuJoCo gym environment.
Fully isolated — no dependencies on TF1 code or sail_sb3/.

Extends train_halfcheetah_teacher_checkpoints.py to all envs by
deriving the output subdirectory from --env_id instead of hardcoding
"halfcheetah".

Outputs (all under --save_root/{env_short}/)
  teacher_checkpoints/
    ep{NNNN}_return{RRR}/   model.zip + meta.json
    target{T}_reached_ep{NNNN}_return{RRR}/  model.zip + meta.json
    best    -> symlink
    latest  -> symlink
  teacher_logs/
    training_log.csv
  run_config.json

TimeFeatureWrapper
------------------
Appends 1.0 - step/max_steps to obs. Applied to both train and eval envs.
Obs dim = native_dim + 1.

Target-return checkpoint (OPTION 2)
------------------------------------
Saved exactly once when mean_eval_return first reaches or exceeds
--target_return. Named target{T}_reached_ep{NNNN}_return{RRR}/.

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

EP_LEN = 1000  # TimeFeatureWrapper horizon; standard for all MuJoCo-v2 envs


# ---------------------------------------------------------------------------
# TimeFeatureWrapper — self-contained, no external imports
# ---------------------------------------------------------------------------

class TimeFeatureWrapper(gym.Wrapper):
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, env_id, n_eval_episodes):
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
                     seed, env_id, obs_dim, is_best, extra_meta=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "model"))
    meta = {
        "episode":        episode,
        "timestep":       episode * EP_LEN,
        "mean_return":    round(float(mean_ret), 4),
        "min_return":     round(float(min_ret),  4),
        "max_return":     round(float(max_ret),  4),
        "seed":           seed,
        "env_id":         env_id,
        "is_best":        bool(is_best),
        "obs_dim":        obs_dim,
        "note":           f"TimeFeatureWrapper applied ({obs_dim-1}+1={obs_dim} dim)",
        "checkpoint_dir": os.path.basename(ckpt_dir),
        "saved_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
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
                    help="All outputs go under {save_root}/{env_short}/")
    ap.add_argument("--env_id",           default="HalfCheetah-v2")
    ap.add_argument("--total_episodes",   type=int,   default=2000)
    ap.add_argument("--checkpoint_every", type=int,   default=100)
    ap.add_argument("--n_eval_episodes",  type=int,   default=10)
    ap.add_argument("--target_return",    type=float, default=None,
                    help="Save milestone ckpt first time mean eval >= this value. "
                         "Skip if not provided.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Derive env short name for directory (e.g. "Ant-v2" -> "ant")
    env_short = args.env_id.split("-")[0].lower()

    # ---- Directories ----
    run_root  = os.path.join(args.save_root, env_short)
    ckpt_root = os.path.join(run_root, "teacher_checkpoints")
    log_dir   = os.path.join(run_root, "teacher_logs")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Infer obs dim from gym (before wrapping)
    _tmp = gym.make(args.env_id)
    native_obs_dim = _tmp.observation_space.shape[0]
    obs_dim = native_obs_dim + 1   # +1 for TimeFeature
    _tmp.close()

    cfg = dict(vars(args))
    cfg["env_short"]   = env_short
    cfg["obs_dim"]     = obs_dim
    cfg["wrapper"]     = f"TimeFeatureWrapper ({native_obs_dim}+1={obs_dim} dim)"
    cfg["algorithm"]   = "TD3 via stable_baselines3 (isolated)"
    cfg["target_rule"] = "OPTION 2: first time mean_eval_return >= target_return"
    cfg["started_at"]  = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run_root, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 68)
    print(f"{args.env_id}  Teacher Training  (SB3 TD3, isolated)")
    print(f"  Total episodes   : {args.total_episodes}  ({args.total_episodes * EP_LEN:,} steps)")
    print(f"  Checkpoint every : {args.checkpoint_every} episodes")
    print(f"  Eval episodes    : {args.n_eval_episodes}")
    print(f"  Target return    : {args.target_return}  (OPTION 2)" if args.target_return else
          f"  Target return    : None (milestone ckpt disabled)")
    print(f"  Seed             : {args.seed}")
    print(f"  Obs dim          : {obs_dim} (TimeFeatureWrapper applied)")
    print(f"  Checkpoints      : {ckpt_root}")
    print(f"  Logs             : {log_dir}")
    print("=" * 68)

    np.random.seed(args.seed)
    random.seed(args.seed)

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

    best_mean_return  = -np.inf
    target_triggered  = False
    t_start = time.time()

    chunks   = list(range(0, args.total_episodes, args.checkpoint_every))
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

        mean_r, min_r, max_r = evaluate(model, args.env_id, args.n_eval_episodes)
        is_best = mean_r > best_mean_return
        if is_best:
            best_mean_return = mean_r

        elapsed = time.time() - t_start
        print(f"  ep={ep:4d}  mean/min/max = {mean_r:.1f}/{min_r:.1f}/{max_r:.1f}"
              f"  {'[BEST] ' if is_best else ''}"
              f"elapsed={elapsed:.0f}s", flush=True)

        name     = _ckpt_name(ep, mean_r)
        ckpt_dir = os.path.join(ckpt_root, name)
        _save_checkpoint(model, ckpt_dir, ep, mean_r, min_r, max_r,
                         args.seed, args.env_id, obs_dim, is_best)
        _update_symlinks(ckpt_root, ckpt_dir, is_best)
        _append_csv(log_dir, ep, mean_r, min_r, max_r, is_best)
        print(f"  [ckpt] {name}", flush=True)

        if (args.target_return is not None
                and not target_triggered
                and mean_r >= args.target_return):
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
                             args.seed, args.env_id, obs_dim, is_best, extra_meta=extra)
            print(f"  [TARGET] *** {mean_r:.1f} >= {args.target_return:.0f} ***", flush=True)
            print(f"  [TARGET] Saved: {tgt_name}", flush=True)

    train_env.close()

    print("\n" + "=" * 68)
    print("Training complete.")
    print(f"  Env              : {args.env_id}")
    print(f"  Best mean return : {best_mean_return:.1f}")
    print(f"  Best checkpoint  : {ckpt_root}/best")
    print(f"  Latest checkpoint: {ckpt_root}/latest")
    print(f"  Training log     : {log_dir}/training_log.csv")
    if args.target_return is not None:
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
