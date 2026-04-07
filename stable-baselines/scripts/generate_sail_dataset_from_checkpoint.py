"""
generate_sail_dataset_from_checkpoint.py  (SB3 isolated version)
=================================================================
Load a saved SB3 TD3 checkpoint and generate a SAIL-compatible .npz dataset.

Fully isolated — no dependencies on:
  - stable-baselines/ TF1 code
  - sail_sb3/ implementation
  - any shared SAIL algorithm files

Output format exactly matches the existing teacher datasets:
  expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
  Keys: obs (N,18) float32 | actions (N,6) float32 | rewards (N,1) float32
        episode_returns (n_traj,) float64 | episode_starts (N,) object

Observation dimension
---------------------
Rollouts are collected WITH TimeFeatureWrapper (17→18 dim) to match SAIL's
expected input. The SAIL ExpertDataset loader receives 18-dim obs.

Usage
-----
  python generate_sail_dataset_from_checkpoint.py \\
      --checkpoint_dir  /path/to/ep0500_return6012 \\
      --env_id          HalfCheetah-v2 \\
      --n_trajs         4 \\
      --save_dir        /path/to/teacher_dataset/ \\
      [--sail_compat_name]   # also save as scores_5600_episodes_N.npz

Python
------
/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
"""

import os
import json
import argparse
import time
import numpy as np
import gym

from stable_baselines3 import TD3

EP_LEN = 1000
SAIL_OPTIMAL_SCORE = 5600   # reference constant from settings.py


# ---------------------------------------------------------------------------
# TimeFeatureWrapper — self-contained copy, no external imports
# Formula: 1.0 - (step / max_steps), same as TF1 wrappers.py and sail_sb3
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
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(model, env_id, n_trajs, max_ep_len=EP_LEN):
    """
    Collect n_trajs deterministic episodes using TimeFeatureWrapper.
    Returns a dict with the exact SAIL-compatible arrays.
    """
    obs_list, act_list, rew_list, ep_rets, ep_lens = [], [], [], [], []

    for _ in range(n_trajs):
        env  = TimeFeatureWrapper(gym.make(env_id))
        obs  = env.reset()
        ep_ret, ep_len, done = 0.0, 0, False
        while not done and ep_len < max_ep_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs.astype(np.float32))
            act_list.append(action.astype(np.float32))
            rew_list.append(float(reward))
            ep_ret += float(reward)
            ep_len += 1
        env.close()
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)

    N = len(obs_list)
    observations = np.array(obs_list, dtype=np.float32)
    actions      = np.array(act_list, dtype=np.float32)
    rewards_2d   = np.array(rew_list, dtype=np.float32).reshape(-1, 1)   # (N, 1)
    ep_returns   = np.array(ep_rets,  dtype=np.float64)
    ep_lens_arr  = np.array(ep_lens,  dtype=np.int32)

    # Build episode_starts in SAIL-compatible object-dtype format:
    # Python True at every episode start, np.array([False]) elsewhere.
    # Matches format of expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz.
    starts = []
    for ep_len in ep_lens_arr:
        starts.append(True)
        for _ in range(1, ep_len):
            starts.append(np.array([False]))
    episode_starts = np.array(starts, dtype=object)

    return dict(
        obs=observations,
        actions=actions,
        rewards=rewards_2d,
        episode_returns=ep_returns,
        episode_starts=episode_starts,
        _episode_lengths=ep_lens_arr,
    )


# ---------------------------------------------------------------------------
# Format verification
# ---------------------------------------------------------------------------

def _verify(path):
    """Assert all SAIL ExpertDataset required keys/shapes are correct."""
    d = np.load(path, allow_pickle=True)
    required = {"obs", "actions", "rewards", "episode_returns", "episode_starts"}
    missing  = required - set(d.keys())
    assert not missing, f"Missing keys: {missing}"

    obs   = d["obs"];    acts = d["actions"]
    rews  = d["rewards"]; epret = d["episode_returns"]
    epst  = d["episode_starts"]

    assert obs.ndim == 2 and obs.shape[1] == 18, \
        f"obs must be (N, 18), got {obs.shape}"
    assert acts.ndim == 2 and acts.shape[1] == 6, \
        f"actions must be (N, 6), got {acts.shape}"
    assert rews.ndim == 2 and rews.shape[1] == 1, \
        f"rewards must be (N, 1), got {rews.shape}"
    assert len(obs) == len(acts) == len(rews) == len(epst)
    assert epret.ndim == 1
    assert epst.dtype == object

    # Verify episode_starts episode count matches episode_returns count
    n_ep = sum(1 for s in epst
               if bool(s if not hasattr(s, "__len__") else s[0]))
    assert n_ep == len(epret), \
        f"episode_starts episodes={n_ep} != episode_returns={len(epret)}"

    # Verify demo_dones derivation (used by ExpertDataset)
    demo_dones = np.concatenate((epst[1:], np.array([1])))
    done_idx   = np.where(demo_dones == 1)[0]
    assert len(done_idx) == len(epret), \
        f"demo_dones episode count {len(done_idx)} != {len(epret)}"

    print(f"  [VERIFY OK] obs={obs.shape} acts={acts.shape} "
          f"rews={rews.shape} n_ep={len(epret)}")
    print(f"  [VERIFY OK] episode_returns={epret}")
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                    help="Checkpoint dir containing model.zip (e.g. ep0500_return6012)")
    ap.add_argument("--env_id",   default="HalfCheetah-v2")
    ap.add_argument("--n_trajs",  type=int, default=4)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--sail_compat_name", action="store_true",
                    help="Also save as expert_data_no_img_HalfCheetah_scores_5600_"
                         "episodes_N.npz so train_sail.py can load it unchanged.")
    ap.add_argument("--max_ep_len", type=int, default=EP_LEN)
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

    print("=" * 68)
    print("SAIL dataset generation from SB3 TD3 checkpoint")
    print(f"  Checkpoint    : {ckpt_dir}")
    print(f"  Training ep   : {training_ep}")
    print(f"  Checkpoint ret: {training_ret}")
    print(f"  n_trajs       : {args.n_trajs}")
    print(f"  env_id        : {args.env_id}")
    print(f"  Obs dim       : 18 (TimeFeatureWrapper applied)")
    print(f"  save_dir      : {args.save_dir}")
    print("=" * 68)

    # Load SB3 TD3 model
    print("Loading SB3 TD3 model ...", flush=True)
    tmp_env = TimeFeatureWrapper(gym.make(args.env_id))
    model   = TD3.load(model_zip, env=tmp_env)
    tmp_env.close()

    # Collect rollouts
    print(f"Collecting {args.n_trajs} deterministic rollouts ...", flush=True)
    t0   = time.time()
    data = collect_rollouts(model, args.env_id, args.n_trajs, args.max_ep_len)
    print(f"  Done in {time.time()-t0:.1f}s")

    ep_ret = data["episode_returns"]
    mean_r = float(ep_ret.mean())
    print(f"  Returns: mean={mean_r:.1f} min={ep_ret.min():.1f} "
          f"max={ep_ret.max():.1f}")
    print(f"  Per episode: {ep_ret.tolist()}")

    # Informative filename using actual rollout returns
    env_name  = args.env_id.split("-")[0]    # "HalfCheetah"
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
    np.savez_compressed(info_path, **save_data)
    print(f"\n[SAVED] {info_path}")
    _verify(info_path)

    # Optional SAIL-canonical filename
    if args.sail_compat_name:
        sail_name = (
            f"expert_data_no_img_{env_name}"
            f"_scores_{SAIL_OPTIMAL_SCORE}"
            f"_episodes_{args.n_trajs}.npz"
        )
        sail_path = os.path.join(args.save_dir, sail_name)
        np.savez_compressed(sail_path, **save_data)
        _verify(sail_path)
        print(f"[SAIL COMPAT] {sail_path}")
        print(f"  → place in teacher_dataset/ and run train_sail.py unchanged")

    # Generation metadata
    gen_meta = {
        "checkpoint_dir":     ckpt_dir,
        "training_episode":   training_ep,
        "checkpoint_return":  training_ret,
        "actual_mean_return": round(mean_r, 4),
        "n_trajs":            args.n_trajs,
        "env_id":             args.env_id,
        "obs_dim":            18,
        "wrapper":            "TimeFeatureWrapper (17->18 dim)",
        "steps":              int(data["obs"].shape[0]),
        "episode_returns":    [round(float(r), 4) for r in ep_ret],
        "saved_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "info_name":          info_name,
    }
    with open(info_path.replace(".npz", "_gen_meta.json"), "w") as f:
        json.dump(gen_meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
