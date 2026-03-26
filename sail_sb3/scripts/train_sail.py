import argparse
import logging
import torch
import os
import sys
import gym
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sail_sb3.algorithms.sail import SAIL
from sail_sb3.reward_models.adversary import Adversary
from sail_sb3.datasets.teacher_buffer import TeacherBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeFeatureWrapper(gym.Wrapper):
    """
    Appends a time-remaining feature [1 → 0] to observations.
    MUST match the original TF codebase (utils/wrappers.py TimeFeatureWrapper).
    The expert .npz files were collected WITH this wrapper applied, so:
      - HalfCheetah-v2 native obs = 17-dim  → with wrapper = 18-dim
      - Expert demos have 18-dim obs
    Skipping this wrapper causes a distribution shift between expert and policy.
    """
    def __init__(self, env, max_steps: int = 1000):
        super().__init__(env)
        low  = np.concatenate([env.observation_space.low,  [0.0]])
        high = np.concatenate([env.observation_space.high, [1.0]])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)
        self._max_steps = max_steps
        self._current_step = 0

    def reset(self):
        self._current_step = 0
        return self._append_time(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._append_time(obs), reward, done, info

    def _append_time(self, obs):
        time_feature = 1.0 - self._current_step / self._max_steps
        return np.concatenate([obs, [time_feature]])


def make_env_with_time(env_id: str, seed: int = 0):
    """
    Returns a DummyVecEnv-compatible factory.
    Wrapping order (inside-out):
      gym.make → Monitor → TimeFeatureWrapper
    Monitor MUST be applied before TimeFeatureWrapper so SB3 can read
    the 'episode' info dict and populate rollout/ep_rew_mean and
    rollout/ep_len_mean. TimeFeatureWrapper is applied on top so the
    time feature is visible to both the policy AND the discriminator.
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed)
        env = Monitor(env)          # enables rollout/ep_rew_mean logging
        env = TimeFeatureWrapper(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    # ---- Core ----
    parser.add_argument("--env",             type=str,   default="HalfCheetah-v2",
                        help="Must match the environment used to collect expert demos. "
                             "Original TF code ALWAYS used HalfCheetah-v2, never v3.")
    parser.add_argument("--expert_data",     type=str,   required=True)
    parser.add_argument("--seed",            type=int,   default=0)

    # ---- Training length ----
    parser.add_argument("--total_timesteps", type=int,   default=1_000_000)

    # ---- TD3 HPs (matching sail.yml exactly) ----
    parser.add_argument("--batch_size",      type=int,   default=256,
                        help="Original sail.yml: 256")
    parser.add_argument("--learning_rate",   type=float, default=1e-3,
                        help="Original sail.yml: 1e-3")
    parser.add_argument("--learning_starts", type=int,   default=10_000,
                        help="Original sail.yml: 10000")
    parser.add_argument("--train_freq",      type=int,   default=1000,
                        help="Original sail.yml: 1000")
    parser.add_argument("--gradient_steps",  type=int,   default=1000,
                        help="Original sail.yml: 1000")
    parser.add_argument("--buffer_size",     type=int,   default=1_000_000,
                        help="Original sail.yml: 1000000")
    parser.add_argument("--policy_delay",    type=int,   default=2)
    parser.add_argument("--tau",             type=float, default=0.005)
    parser.add_argument("--gamma",           type=float, default=0.99)

    # ---- Discriminator HPs (matching original sail.py defaults) ----
    parser.add_argument("--disc_lr",              type=float, default=3e-4,
                        help="Original: d_learning_rate=3e-4")
    parser.add_argument("--disc_batch_size",      type=int,   default=256,
                        help="Original: d_batch_size = batch_size = 256")
    parser.add_argument("--disc_train_freq",      type=int,   default=500,
                        help="Original: train_discriminator_freq=500 (env steps)")
    parser.add_argument("--disc_gradient_steps",  type=int,   default=10,
                        help="Original: d_gradient_steps=10")
    parser.add_argument("--hidden_size",          type=int,   default=256,
                        help="Original: hidden_size_adversary=256 (hidden_size arg to disc)")
    parser.add_argument("--entcoeff",             type=float, default=0.01,
                        help="Original: adversary_entcoeff=1e-3, but without obs_rms 0.01 prevents saturation")
    parser.add_argument("--gradcoeff",            type=float, default=10.0,
                        help="Original: gradient_penalty_entcoeff=10")

    # ---- Preference Ranking ----
    parser.add_argument("--pref_rank_disc",       action="store_true",
                        help="Enable preference ranking loss in discriminator")
    parser.add_argument("--pref_rank_weight",     type=float, default=0.1,
                        help="Weight for preference ranking loss")
    parser.add_argument("--pref_rank_batch_size", type=int,   default=16,
                        help="Number of pairs to sample per update")
    parser.add_argument("--pref_rm",              type=str,   default=None,
                        help="Path to offline preference reward model")
    parser.add_argument("--pref_expect_obs_dim",  type=int,   default=17,
                        help="Expected observation dimension for offline RM")

    # ---- Debug ----
    parser.add_argument("--debug",           action="store_true",
                        help="Enable per-step discriminator / reward / Q debug output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_sail] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Environment (MUST use v2 with TimeFeatureWrapper to match demos)
    # ------------------------------------------------------------------
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([make_env_with_time(args.env, seed=args.seed)])

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[train_sail] ENV:        {args.env}")
    print(f"[train_sail] obs dim:    {state_dim}  (native + 1 time feature)")
    print(f"[train_sail] action dim: {action_dim}")

    # ------------------------------------------------------------------
    # 2. Teacher buffer
    # ------------------------------------------------------------------
    print("[train_sail] Loading teacher data ...")
    teacher_buffer = TeacherBuffer(
        args.expert_data, 
        device,
        pref_rm_path=args.pref_rm if args.pref_rank_disc else None,
        expect_obs_dim=args.pref_expect_obs_dim
    )
    expert_obs_dim = teacher_buffer.states.shape[1]
    expert_act_dim = teacher_buffer.actions.shape[1]

    print(f"[train_sail] Expert obs dim:    {expert_obs_dim}")
    print(f"[train_sail] Expert action dim: {expert_act_dim}")


    # Compute true expert stats for sanity check
    ep_starts = np.insert(teacher_buffer.dones[:-1].cpu().numpy(), 0, 0) == 1
    ep_slices = np.where(teacher_buffer.dones.cpu().numpy() == 1)[0]
    returns = []
    start = 0
    t_rews = teacher_buffer.rewards.cpu().numpy()
    for end in ep_slices:
        returns.append(t_rews[start:end+1].sum())
        start = end + 1
    if not returns:
        returns.append(t_rews.sum())
    
    print(f"[train_sail] Expert dataset paths: {args.expert_data}")
    print(f"[train_sail] Expert episodes: {len(returns)} | Total transitions: {teacher_buffer.size()}")
    print(f"[train_sail] Expert return stats - Mean: {np.mean(returns):.1f} | Std: {np.std(returns):.1f} | Min: {np.min(returns):.1f} | Max: {np.max(returns):.1f}")
    if np.mean(returns) < 1000 and "5600" in args.expert_data:
        print("[train_sail] WARNING: Expert dataset filename suggests 5600 score but actual returns are < 1000!")

    if expert_obs_dim != state_dim:
        print(
            f"[train_sail] WARNING: expert obs dim ({expert_obs_dim}) ≠ env obs dim ({state_dim}). "
            f"Attempting to reconcile ..."
        )
        diff = state_dim - expert_obs_dim
        if diff > 0:
            # Expert has fewer dims than env (rare): pad with zeros
            padding = torch.zeros(
                teacher_buffer.states.shape[0], diff, device=device)
            teacher_buffer.states = torch.cat(
                [teacher_buffer.states, padding], dim=-1)
            print(f"[train_sail] Zero-padded expert obs by {diff} dims.")
        else:
            # Expert has MORE dims than env: trim TRAILING dims (former time feature)
            trim = -diff  # positive number
            teacher_buffer.states = teacher_buffer.states[:, :state_dim]
            print(f"[train_sail] Trimmed trailing {trim} dim(s) from expert obs.")

    if expert_act_dim != action_dim:
        raise ValueError(
            f"Expert action dim ({expert_act_dim}) ≠ env action dim ({action_dim}). "
            f"Incompatible dataset!"
        )

    # Check for likely env version mismatch
    if "HalfCheetah" in args.env and "v3" in args.env:
        print(
            "[train_sail] WARNING: You are using HalfCheetah-v3 but all expert demos "
            "were collected in HalfCheetah-v2. The reward functions differ: "
            "v2 does NOT penalize control cost, v3 DOES with ctrl_cost_weight=0.1. "
            "This causes a distribution shift in the discriminator that prevents learning."
        )

    # ------------------------------------------------------------------
    # 3. Discriminator
    # ------------------------------------------------------------------
    print("[train_sail] Initializing discriminator ...")
    discriminator = Adversary(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
        entcoeff=args.entcoeff,
        gradcoeff=args.gradcoeff,
    ).to(device)

    # ------------------------------------------------------------------
    # 4. SAIL model (matching sail.yml HPs exactly)
    # ------------------------------------------------------------------
    print("[train_sail] Initializing SAIL (TD3 subclass) ...")
    policy_kwargs = dict(net_arch=[400, 300])   # Original: policy_kwargs: dict(layers=[400, 300])

    model = SAIL(
        policy="MlpPolicy",
        env=env,
        discriminator=discriminator,
        teacher_buffer=teacher_buffer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        buffer_size=args.buffer_size,
        policy_delay=args.policy_delay,
        tau=args.tau,
        gamma=args.gamma,
        disc_lr=args.disc_lr,
        disc_batch_size=args.disc_batch_size,
        disc_train_freq=args.disc_train_freq,
        disc_gradient_steps=args.disc_gradient_steps,
        pref_rank_disc=args.pref_rank_disc,
        pref_rank_weight=args.pref_rank_weight,
        pref_rank_batch_size=args.pref_rank_batch_size,
        debug=args.debug,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./sail_tensorboard/",
    )

    new_logger = configure("./sail_logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"[train_sail] Starting training: {args.total_timesteps} timesteps")
    print(f"[train_sail] TD3 HPs: batch={args.batch_size}  lr={args.learning_rate}"
          f"  train_freq={args.train_freq}  grad_steps={args.gradient_steps}"
          f"  learning_starts={args.learning_starts}")
    print(f"[train_sail] Disc HPs: train_freq={args.disc_train_freq}"
          f"  grad_steps={args.disc_gradient_steps}  batch={args.disc_batch_size}"
          f"  entcoeff={args.entcoeff}  gradcoeff={args.gradcoeff}")

    model.learn(total_timesteps=args.total_timesteps, log_interval=1)
    print("[train_sail] Training finished successfully!")


if __name__ == "__main__":
    main()
