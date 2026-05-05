"""
retrain_swimmer_teacher.py — Swimmer-v2 TD3 teacher retraining
==============================================================
Fixes the failed original run (ep0100–ep2000 all stuck at -17 to -21).

Root cause of original failure
-------------------------------
- Swimmer-v2 reward per step is very small (~0.009 for zero-action policy).
- Original run used learning_starts=100 (SB3 default): only 100 random
  transitions before Q-network updates begin.  With such sparse signal the
  critic got poor initial estimates; the policy converged to a backward-swimming
  local minimum and NEVER escaped over 2M steps.
- All other envs (HC, Ant, Hopper, Walker) have much larger per-step rewards
  and so tolerate learning_starts=100.

Fixes applied
--------------
- learning_starts=10000  (fill replay buffer with diverse random experience first)
- action_noise_sigma=0.3 (stronger exploration to escape local optima)
- learning_rate=3e-4     (slower Q-network updates → less sensitive to early noise)
- New output directory   (old checkpoints preserved untouched)
- target_return=345      (aligned with expected TD3 Swimmer-v2 benchmark)

All other structure is identical to train_teacher_checkpoints.py:
TimeFeatureWrapper, EP_LEN=1000, checkpoint_every=100, n_eval_episodes=10.
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

EP_LEN = 1000


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root",          required=True)
    ap.add_argument("--env_id",             default="Swimmer-v2")
    ap.add_argument("--total_episodes",     type=int,   default=3000)
    ap.add_argument("--checkpoint_every",   type=int,   default=100)
    ap.add_argument("--n_eval_episodes",    type=int,   default=10)
    ap.add_argument("--target_return",      type=float, default=345.0)
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--learning_starts",    type=int,   default=10000,
                    help="Steps of random exploration before Q-updates begin.")
    ap.add_argument("--action_noise_sigma", type=float, default=0.3,
                    help="Std of Gaussian action noise for exploration.")
    ap.add_argument("--learning_rate",      type=float, default=3e-4,
                    help="TD3 learning rate for both actor and critic.")
    args = ap.parse_args()

    env_short = args.env_id.split("-")[0].lower()

    run_root  = os.path.join(args.save_root, env_short)
    ckpt_root = os.path.join(run_root, "teacher_checkpoints")
    log_dir   = os.path.join(run_root, "teacher_logs")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    _tmp = gym.make(args.env_id)
    native_obs_dim = _tmp.observation_space.shape[0]
    obs_dim = native_obs_dim + 1
    _tmp.close()

    cfg = dict(vars(args))
    cfg["env_short"]          = env_short
    cfg["obs_dim"]            = obs_dim
    cfg["wrapper"]            = f"TimeFeatureWrapper ({native_obs_dim}+1={obs_dim} dim)"
    cfg["algorithm"]          = "TD3 via stable_baselines3 (retrain)"
    cfg["target_rule"]        = "OPTION 2: first time mean_eval_return >= target_return"
    cfg["fix_notes"]          = (
        "Fixes failed original run (all scores -17 to -21 over 2M steps). "
        "Root cause: learning_starts=100 too low for Swimmer-v2's tiny reward scale. "
        "Fix: learning_starts=10000, action_noise_sigma=0.3, lr=3e-4."
    )
    cfg["started_at"]         = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run_root, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 68)
    print(f"{args.env_id}  Teacher RETRAIN  (SB3 TD3, fixed hyperparams)")
    print(f"  Total episodes   : {args.total_episodes}  ({args.total_episodes * EP_LEN:,} steps)")
    print(f"  Checkpoint every : {args.checkpoint_every} episodes")
    print(f"  Eval episodes    : {args.n_eval_episodes}")
    print(f"  Target return    : {args.target_return}")
    print(f"  Seed             : {args.seed}")
    print(f"  Obs dim          : {obs_dim} (TimeFeatureWrapper applied)")
    print(f"  learning_starts  : {args.learning_starts}")
    print(f"  action_noise σ   : {args.action_noise_sigma}")
    print(f"  learning_rate    : {args.learning_rate}")
    print(f"  Checkpoints      : {ckpt_root}")
    print(f"  Logs             : {log_dir}")
    print("=" * 68)
    print("FIX: learning_starts=10000 addresses root cause of original failure.")
    print("=" * 68)

    np.random.seed(args.seed)
    random.seed(args.seed)

    train_env = TimeFeatureWrapper(gym.make(args.env_id))
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.action_noise_sigma * np.ones(n_actions),
    )
    model = TD3(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[400, 300]),
        buffer_size=1_000_000,
        batch_size=256,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        action_noise=action_noise,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=args.seed,
    )

    best_mean_return = -np.inf
    target_triggered = False
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
    print("=" * 68)


if __name__ == "__main__":
    main()
