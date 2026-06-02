"""
generate_sail_dataset_fast.py  —  optimized dataset generation
==============================================================
Drop-in replacement for generate_sail_dataset_from_checkpoint.py.
Produces identical SAIL-compatible .npz output.

Root cause of original slowness (3+ hours for 100 trajs)
---------------------------------------------------------
PyTorch defaults to using ALL available CPU cores (often 64+ on HPC).
For a tiny 2-layer TD3 actor, dispatching 1000 matrix-multiply calls
per episode through a 64-thread pool has catastrophic thread-dispatch
and synchronisation overhead — FAR exceeding the actual compute.
Fix: torch.set_num_threads(1) → single-threaded BLAS for small nets.

Key changes vs original
-----------------------
1. torch.set_num_threads(1) + torch.set_num_interop_threads(1) called
   immediately after SB3 import (before any model.predict call).
2. numpy arrays pre-allocated to avoid 100k dynamic list appends.
3. Per-episode progress logging: episode#, return, steps, elapsed time.
4. Stdout flushed after every episode so sbatch logs are live.
5. OMP/MKL/OpenBLAS thread env-vars should also be set in the launcher
   (sbatch script) to prevent numpy/scipy thread explosion.

Output format
-------------
Identical to generate_sail_dataset_from_checkpoint.py:
  obs             (N, obs_dim) float32
  actions         (N, act_dim) float32
  rewards         (N, 1)       float32
  episode_returns (n_trajs,)   float64
  episode_starts  (N,)         object  — True at ep start, np.array([False]) elsewhere

Usage
-----
  python generate_sail_dataset_fast.py \\
      --checkpoint_dir  /path/to/ep0400_return6718 \\
      --env_id          HalfCheetah-v2 \\
      --n_trajs         100 \\
      --save_dir        teacher_dataset_HC/

Python binary
-------------
/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import gym

# ---------------------------------------------------------------------------
# Thread limits — import SB3/torch, then immediately cap threads.
# Must happen before any model.predict() call.
# ---------------------------------------------------------------------------
from stable_baselines3 import TD3

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EP_LEN = 1000
SAIL_OPTIMAL_SCORE = 5600


# ---------------------------------------------------------------------------
# TimeFeatureWrapper — identical to original, self-contained
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
# Rollout collection — optimised
# ---------------------------------------------------------------------------

def collect_rollouts(model, env_id, n_trajs, max_ep_len=EP_LEN):
    """
    Collect n_trajs deterministic episodes.

    Changes vs original:
    - Pre-allocate obs/actions/rewards arrays (avoids 100k list appends).
    - Per-episode progress printed + flushed immediately.
    - Exact same SAIL-compatible output format.
    """
    # Probe observation/action dims from a throwaway env
    _probe = TimeFeatureWrapper(gym.make(env_id))
    obs_dim = _probe.observation_space.shape[0]
    act_dim = _probe.action_space.shape[0]
    _probe.close()

    max_steps = n_trajs * max_ep_len

    # Pre-allocate
    obs_buf  = np.zeros((max_steps, obs_dim),  dtype=np.float32)
    act_buf  = np.zeros((max_steps, act_dim),  dtype=np.float32)
    rew_buf  = np.zeros((max_steps,),          dtype=np.float32)
    ep_rets  = np.zeros((n_trajs,),            dtype=np.float64)
    ep_lens  = np.zeros((n_trajs,),            dtype=np.int32)

    ptr = 0  # write pointer into buffers
    t_start = time.time()

    for i in range(n_trajs):
        t_ep = time.time()
        env  = TimeFeatureWrapper(gym.make(env_id))
        obs  = env.reset()
        ep_ret, ep_len, done = 0.0, 0, False

        while not done and ep_len < max_ep_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            obs_buf[ptr] = obs
            act_buf[ptr] = action
            rew_buf[ptr] = reward
            ep_ret += float(reward)
            ep_len += 1
            ptr += 1

        env.close()
        ep_rets[i] = ep_ret
        ep_lens[i] = ep_len

        elapsed = time.time() - t_start
        ep_time = time.time() - t_ep
        print(
            f"  ep {i+1:3d}/{n_trajs}  ret={ep_ret:8.1f}  steps={ep_len}  "
            f"ep_time={ep_time:.1f}s  total={elapsed:.1f}s",
            flush=True,
        )

    # Slice to actual collected steps
    observations = obs_buf[:ptr]
    actions      = act_buf[:ptr]
    rewards_2d   = rew_buf[:ptr].reshape(-1, 1)

    # Build episode_starts in SAIL object-dtype format
    # True at episode start, np.array([False]) at every other step
    _false = np.array([False])
    episode_starts = np.empty(ptr, dtype=object)
    episode_starts[:] = _false
    ep_start_idx = 0
    for ep_len in ep_lens:
        episode_starts[ep_start_idx] = True
        ep_start_idx += ep_len

    total_time = time.time() - t_start
    print(f"\n  Rollout summary:")
    print(f"    Total steps : {ptr}")
    print(f"    Total time  : {total_time:.1f}s  ({total_time/n_trajs:.1f}s/ep avg)")
    print(f"    Returns     : mean={ep_rets.mean():.1f}  std={ep_rets.std():.1f}  "
          f"min={ep_rets.min():.1f}  max={ep_rets.max():.1f}", flush=True)

    return dict(
        obs=observations,
        actions=actions,
        rewards=rewards_2d,
        episode_returns=ep_rets,
        episode_starts=episode_starts,
        _episode_lengths=ep_lens,
    )


# ---------------------------------------------------------------------------
# Format verification — identical to original
# ---------------------------------------------------------------------------

def _verify(path):
    d = np.load(path, allow_pickle=True)
    required = {"obs", "actions", "rewards", "episode_returns", "episode_starts"}
    missing  = required - set(d.keys())
    assert not missing, f"Missing keys: {missing}"

    obs   = d["obs"];    acts = d["actions"]
    rews  = d["rewards"]; epret = d["episode_returns"]
    epst  = d["episode_starts"]

    assert obs.ndim  == 2, f"obs must be 2D, got {obs.shape}"
    assert acts.ndim == 2, f"actions must be 2D, got {acts.shape}"
    assert rews.ndim == 2 and rews.shape[1] == 1, \
        f"rewards must be (N, 1), got {rews.shape}"
    assert len(obs) == len(acts) == len(rews) == len(epst)
    assert epret.ndim == 1
    assert epst.dtype == object

    n_ep = sum(1 for s in epst
               if bool(s if not hasattr(s, "__len__") else s[0]))
    assert n_ep == len(epret), \
        f"episode_starts episodes={n_ep} != episode_returns={len(epret)}"

    demo_dones = np.concatenate((epst[1:], np.array([1])))
    done_idx   = np.where(demo_dones == 1)[0]
    assert len(done_idx) == len(epret), \
        f"demo_dones episode count {len(done_idx)} != {len(epret)}"

    print(f"  [VERIFY OK] obs={obs.shape} acts={acts.shape} "
          f"rews={rews.shape} n_ep={len(epret)}", flush=True)
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--env_id",         default="HalfCheetah-v2")
    ap.add_argument("--n_trajs",        type=int, default=4)
    ap.add_argument("--save_dir",       required=True)
    ap.add_argument("--max_ep_len",     type=int, default=EP_LEN)
    args = ap.parse_args()

    ckpt_dir  = os.path.abspath(args.checkpoint_dir)
    model_zip = os.path.join(ckpt_dir, "model.zip")
    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"model.zip not found in {ckpt_dir}")

    meta_path = os.path.join(ckpt_dir, "meta.json")
    ckpt_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ckpt_meta = json.load(f)

    training_ep  = ckpt_meta.get("episode", "unknown")
    training_ret = ckpt_meta.get("mean_return", None)

    print("=" * 68, flush=True)
    print("SAIL dataset generation (fast version)")
    print(f"  Checkpoint    : {ckpt_dir}")
    print(f"  Training ep   : {training_ep}")
    print(f"  Checkpoint ret: {training_ret}")
    print(f"  n_trajs       : {args.n_trajs}")
    print(f"  env_id        : {args.env_id}")
    print(f"  max_ep_len    : {args.max_ep_len}")
    print(f"  save_dir      : {args.save_dir}")
    print(f"  torch threads : {torch.get_num_threads()} (forced to 1)")
    print("=" * 68, flush=True)

    print("Loading SB3 TD3 model ...", flush=True)
    tmp_env = TimeFeatureWrapper(gym.make(args.env_id))
    model   = TD3.load(model_zip, env=tmp_env)
    tmp_env.close()
    print("  Model loaded.", flush=True)

    print(f"\nCollecting {args.n_trajs} deterministic rollouts ...", flush=True)
    data = collect_rollouts(model, args.env_id, args.n_trajs, args.max_ep_len)

    ep_ret   = data["episode_returns"]
    mean_r   = float(ep_ret.mean())
    env_name = args.env_id.split("-")[0]
    score_tag = int(round(mean_r))
    ep_tag    = str(training_ep)

    info_name = (
        f"expert_data_no_img_{env_name}"
        f"_scores_{score_tag}"
        f"_episodes_{args.n_trajs}"
        f"_ep{ep_tag}.npz"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_data = {k: v for k, v in data.items() if not k.startswith("_")}

    info_path = os.path.join(args.save_dir, info_name)
    print(f"\nSaving → {info_path}", flush=True)
    np.savez_compressed(info_path, **save_data)
    print(f"[SAVED] {info_path}", flush=True)
    _verify(info_path)

    gen_meta = {
        "checkpoint_dir":     ckpt_dir,
        "training_episode":   training_ep,
        "checkpoint_return":  training_ret,
        "actual_mean_return": round(mean_r, 4),
        "n_trajs":            args.n_trajs,
        "env_id":             args.env_id,
        "obs_dim":            int(data["obs"].shape[1]),
        "wrapper":            "TimeFeatureWrapper (+1 time feature)",
        "steps":              int(data["obs"].shape[0]),
        "episode_returns":    [round(float(r), 4) for r in ep_ret],
        "saved_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "info_name":          info_name,
        "generation_script":  "generate_sail_dataset_fast.py",
        "torch_threads":      torch.get_num_threads(),
    }
    with open(info_path.replace(".npz", "_gen_meta.json"), "w") as f:
        json.dump(gen_meta, f, indent=2)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
