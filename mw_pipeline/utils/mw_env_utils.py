"""
mw_env_utils.py  —  Meta-World environment utilities
=====================================================
Standalone helper module for constructing Meta-World environments.
No imports from sail_sb3/ or sail_sb3_online/.

Provides
--------
MetaWorldWrapper       Thin gym.Wrapper that adapts Meta-World to SB3 1.8.0 / gym 0.21.0.
TimeFeatureWrapper     Self-contained copy (matches sail_sb3_online version); used only
                       by make_mw_sail_env (Phase 4, not needed for teacher training).
make_mw_teacher_env    Factory for teacher training: raw obs, no time feature.
make_mw_sail_env       Factory for SAIL training: +time feature (Phase 4, define now).
is_metaworld_task      Checks whether an env_id string refers to a Meta-World task.
MW_TASKS               List of tested/supported task names.
MW_EXPERT_RETURNS      Reference success rates populated after teacher training.
MW_OBS_DIM / MW_ACT_DIM / MW_EP_LEN  Environment constants for v2 tasks.

Meta-World compatibility notes
-------------------------------
- Meta-World does NOT use gym.make(). Envs are constructed via metaworld.MT1.
- Newer Meta-World releases follow the Gymnasium API: reset() returns (obs, info)
  and step() returns (obs, rew, terminated, truncated, info).
  MetaWorldWrapper normalises both back to the gym 0.21.0 4-tuple used by SB3 1.8.0.
- Every episode, set_task() must be called to re-sample the goal before reset().
  MetaWorldWrapper handles this automatically in reset().
- Episode length for all v2 tasks is 200 steps (MW_EP_LEN).
- Observation dimension is 39 for all v2 tasks (MW_OBS_DIM).
- Action dimension is 4 for all v2 tasks (MW_ACT_DIM).
"""

import random
import numpy as np
import gym


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MW_OBS_DIM = 39   # native obs dim for all Meta-World v2 tasks
MW_ACT_DIM = 4    # (Δx, Δy, Δz, gripper) for all v2 tasks
MW_EP_LEN  = 200  # max_episode_steps for all v2 tasks

# Supported / tested tasks.  Extend this list as new tasks are validated.
MW_TASKS = [
    "reach-v2",
    "push-v2",
    "pick-place-v2",
    "button-press-v2",
    "window-open-v2",
    "door-open-v2",
]

# Populated incrementally after teacher training and evaluation.
# Keys are bare task names (without the "mw:" routing prefix).
# Fill in mean episodic return from the chosen teacher checkpoint's meta.json.
MW_EXPERT_RETURNS = {
    "reach-v2":      None,   # fill after teacher training
    "push-v2":       None,
    "pick-place-v2": None,
}


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def is_metaworld_task(env_id: str) -> bool:
    """Return True if env_id uses the "mw:" prefix convention (e.g. "mw:reach-v2")."""
    return isinstance(env_id, str) and env_id.startswith("mw:")


# ---------------------------------------------------------------------------
# Lazy Meta-World import
# ---------------------------------------------------------------------------

def _import_metaworld():
    """
    Import metaworld lazily so that modules that import mw_env_utils do not
    break when metaworld is not installed, as long as no MW factory is called.
    """
    try:
        import metaworld as _mw
        return _mw
    except ImportError:
        raise ImportError(
            "metaworld is not installed.\n"
            "Install it with:\n"
            "  /home/sohams/miniconda3/envs/sail_sb3_env/bin/pip install metaworld\n"
            "See metaworld/requirements.txt for details."
        )


# ---------------------------------------------------------------------------
# MetaWorldWrapper
# ---------------------------------------------------------------------------

class MetaWorldWrapper(gym.Wrapper):
    """
    Adapts a Meta-World environment to be compatible with SB3 1.8.0 / gym 0.21.0.

    Handles three incompatibilities:
    1. reset() in newer Meta-World returns (obs, info) — unwrapped to obs only.
    2. step()  in newer Meta-World returns 5-tuple  — collapsed to 4-tuple.
    3. set_task() must be called before each episode — done automatically in reset().

    info["success"] is always present as a Python bool after every step().
    """

    def __init__(self, env: gym.Env, tasks: list):
        super().__init__(env)
        if not tasks:
            raise ValueError("tasks list must be non-empty")
        self._tasks = tasks
        # Set an initial task so the env is in a valid state before first reset().
        self.env.set_task(random.choice(self._tasks))

    def reset(self):
        self.env.set_task(random.choice(self._tasks))
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        obs = np.array(obs, dtype=np.float32)
        info["success"] = bool(info.get("success", False))
        return obs, float(reward), done, info


# ---------------------------------------------------------------------------
# TimeFeatureWrapper (self-contained; only used by make_mw_sail_env)
# ---------------------------------------------------------------------------

class TimeFeatureWrapper(gym.Wrapper):
    """
    Appends a time-remaining feature [1.0 → 0.0] to every observation.
    Self-contained copy so metaworld/ has no dependency on sail_sb3_online/.
    For Meta-World use max_steps=MW_EP_LEN (200).
    """

    def __init__(self, env: gym.Env, max_steps: int = MW_EP_LEN):
        super().__init__(env)
        low  = np.concatenate([env.observation_space.low,  [0.0]])
        high = np.concatenate([env.observation_space.high, [1.0]])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._max_steps    = max_steps
        self._current_step = 0

    def reset(self):
        self._current_step = 0
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._append_time(obs)

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._append_time(obs), reward, done, info

    def _append_time(self, obs):
        t = 1.0 - self._current_step / self._max_steps
        return np.concatenate([obs, [t]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_mw_teacher_env(task_name: str, seed: int = 0):
    """
    Factory for Meta-World teacher training (Phase 1).
    No TimeFeatureWrapper — teacher is trained on the raw 39-dim observation.

    Wrapping order (inside-out):
        raw Meta-World env → MetaWorldWrapper → TimeLimit(200) → Monitor

    TimeLimit is required: Meta-World v2.0.0 raises ValueError if the env
    is stepped beyond its internal max_path_length without a reset.  Since
    done=True only fires on task success, an untrained policy would exceed
    the limit on every episode without this wrapper.

    Returns a zero-argument callable compatible with SB3's DummyVecEnv.
    """
    from stable_baselines3.common.monitor import Monitor
    from gym.wrappers import TimeLimit

    def _init():
        mw = _import_metaworld()
        ml1 = mw.MT1(task_name, seed=seed)
        env_cls = ml1.train_classes[task_name]
        env = env_cls()
        tasks = ml1.train_tasks
        env = MetaWorldWrapper(env, tasks)
        env = TimeLimit(env, max_episode_steps=MW_EP_LEN)
        env = Monitor(env)
        return env

    return _init


def make_mw_sail_env(task_name: str, seed: int = 0):
    """
    Factory for SAIL training on Meta-World (Phase 4 — defined here, not used yet).
    Adds TimeFeatureWrapper so the policy and discriminator see the 40-dim obs
    that will be stored in the teacher dataset.

    Wrapping order (inside-out):
        raw Meta-World env → MetaWorldWrapper → TimeLimit(200) → Monitor → TimeFeatureWrapper

    Returns a zero-argument callable compatible with SB3's DummyVecEnv.
    """
    from stable_baselines3.common.monitor import Monitor
    from gym.wrappers import TimeLimit

    def _init():
        mw = _import_metaworld()
        ml1 = mw.MT1(task_name, seed=seed)
        env_cls = ml1.train_classes[task_name]
        env = env_cls()
        tasks = ml1.train_tasks
        env = MetaWorldWrapper(env, tasks)
        env = TimeLimit(env, max_episode_steps=MW_EP_LEN)
        env = Monitor(env)
        env = TimeFeatureWrapper(env, max_steps=MW_EP_LEN)
        return env

    return _init


# ---------------------------------------------------------------------------
# Quick verification  (python -m metaworld.utils.mw_env_utils)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying Meta-World environment interface ...")
    print("  Importing metaworld ...")
    mw = _import_metaworld()
    print(f"  metaworld imported OK")

    TASK = "reach-v2"
    print(f"  Constructing {TASK} via MT1 ...")
    factory = make_mw_teacher_env(TASK, seed=0)
    env = factory()

    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")

    obs = env.reset()
    print(f"  reset() obs shape : {obs.shape}  (expect ({MW_OBS_DIM},))")
    assert obs.shape == (MW_OBS_DIM,), f"Expected ({MW_OBS_DIM},), got {obs.shape}"

    action = env.action_space.sample()
    obs2, rew, done, info = env.step(action)
    print(f"  step()  obs shape : {obs2.shape}")
    print(f"          reward    : {rew:.4f}")
    print(f"          done      : {done}")
    print(f"          success   : {info['success']}  (key present: {'success' in info})")
    assert "success" in info, "info['success'] missing from step() output"
    assert obs2.shape == (MW_OBS_DIM,), f"Step obs shape mismatch: {obs2.shape}"
    env.close()

    print(f"\n  TimeFeatureWrapper check ...")
    sail_factory = make_mw_sail_env(TASK, seed=0)
    sail_env = sail_factory()
    obs_tf = sail_env.reset()
    print(f"  SAIL env obs shape: {obs_tf.shape}  (expect ({MW_OBS_DIM + 1},))")
    assert obs_tf.shape == (MW_OBS_DIM + 1,), f"Expected ({MW_OBS_DIM + 1},), got {obs_tf.shape}"
    sail_env.close()

    print("\nAll checks passed.")
