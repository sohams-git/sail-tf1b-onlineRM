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
from sail_sb3.utils.callbacks import SAILAdaptiveCallback
from sail_sb3.datasets.teacher_buffer import TeacherBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import wandb (gracefully handle if not installed)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Logging will be local only.")


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
    parser.add_argument("--device",          type=str,   default=None,
                        help="Torch device (e.g., 'cpu', 'cuda', 'cuda:0'). If None, auto-select.")

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

    # ---- Adaptive SAIL/PAIL ----
    parser.add_argument("--adaptive",        action="store_true",
                        help="Enable adaptive teacher buffer replacement (PAIL). "
                             "Promotes student trajectories to teacher buffer when they exceed expert threshold.")
    parser.add_argument("--lfd_mixing",      action="store_true",
                        help="Enable LfD mixing (TF parity): mix 50%% expert + 50%% policy into every "
                             "critic/actor batch before first promotion. Matches TF gail-lfd-adaptive-dynamic.")
    parser.add_argument("--teacher_buffer_size", type=int, default=None,
                        help="Ring buffer size for teacher buffer (TF parity: 1000 for HalfCheetah). "
                             "If set, buffer is capped at this size with FIFO overwrite semantics. "
                             "None = unbounded append-only (original behavior).")
    parser.add_argument("--adaptive_score_source", type=str, default="gt",
                        choices=["gt", "rm"],
                        help="Score source for adaptive promotion decisions. "
                             "'gt' = ground-truth environment return (default). "
                             "'rm' = preference reward model cumulative score (no GT leakage). "
                             "Requires --pref_rm when set to 'rm'.")

    # ---- Debug ----
    parser.add_argument("--debug",           action="store_true",
                        help="Enable per-step discriminator / reward / Q debug output")
    args = parser.parse_args()
    
    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
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
    if args.adaptive_score_source == "rm" and not args.pref_rm:
        raise ValueError("--adaptive_score_source rm requires --pref_rm")

    print("[train_sail] Loading teacher data ...")
    # Load RM if needed for pref ranking OR RM-based adaptive promotion
    need_pref_rm = args.pref_rank_disc or (args.adaptive_score_source == "rm")
    teacher_buffer = TeacherBuffer(
        args.expert_data,
        device,
        pref_rm_path=args.pref_rm if need_pref_rm else None,
        expect_obs_dim=args.pref_expect_obs_dim,
        max_size=args.teacher_buffer_size,
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

    # Initialize expert_scores list for adaptive SAIL (TF line 1364).
    # TF parity: insertion order (NOT sorted).  TF appends episodes in load order so
    # expert_scores[0] = first-loaded episode = 6932.27 for HalfCheetah.
    # Sorting would give expert_scores[0] = 6741.27 (lowest), a 191-point lower threshold.
    expert_scores = list(returns)  # Insertion order
    if args.adaptive:
        print(f"[train_sail] Adaptive mode: expert score threshold initialized to {expert_scores[0]:.1f} "
              f"(first-loaded episode, TF insertion-order parity)")

    # RM-based adaptive promotion: build rm_expert_scores from RM-scored expert episodes.
    # Insertion order matches expert_scores: rm_expert_scores[0] = RM score of first-loaded episode.
    rm_expert_scores = None
    if args.adaptive and args.adaptive_score_source == "rm":
        if not teacher_buffer.pref_episodes:
            raise ValueError("[train_sail] RM-based promotion requires --pref_rm with valid RM path "
                             "and at least one expert episode scored by the RM.")
        rm_expert_scores = [ep['J'] for ep in teacher_buffer.pref_episodes]  # Insertion order
        print(f"[train_sail] RM adaptive: rm_expert_scores[0] (threshold) = {rm_expert_scores[0]:.1f}")
        print(f"[train_sail] RM adaptive: rm_expert_scores = {[f'{s:.1f}' for s in rm_expert_scores]}")

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
        adaptive=args.adaptive,
        expert_scores=expert_scores if args.adaptive else None,
        lfd_mixing=args.lfd_mixing,
        debug=args.debug,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./sail_tensorboard/",
    )

    new_logger = configure("./sail_logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # ------------------------------------------------------------------
    # 5. Weights & Biases Integration
    # ------------------------------------------------------------------
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            # Get wandb configuration from environment (matching TF implementation)
            wandb_project = os.getenv("WANDB_PROJECT", f"SAIL_SB3_{args.env.split('-')[0]}")
            wandb_name = os.getenv("WANDB_NAME", f"SAIL_{args.env}_s{args.seed}")
            wandb_group = os.getenv("WANDB_GROUP", f"{args.env}_vanilla_{args.total_timesteps}")
            wandb_entity = os.getenv("WANDB_ENTITY", None)

            # Set silent mode (suppress wandb console output)
            os.environ.setdefault("WANDB_SILENT", "true")

            # Initialize wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                group=wandb_group,
                entity=wandb_entity,
                config=vars(args),
                reinit=True
            )

            # Sync TensorBoard logs to wandb (matching TF implementation)
            wandb.tensorboard.patch(root_logdir="./sail_tensorboard/")

            # Log expert dataset statistics to wandb config
            expert_stats = {
                "expert_dataset_path": args.expert_data,
                "expert_episode_count": len(returns),
                "expert_return_mean": float(np.mean(returns)),
                "expert_return_std": float(np.std(returns)),
                "expert_return_min": float(np.min(returns)),
                "expert_return_max": float(np.max(returns)),
                "expert_total_transitions": teacher_buffer.size(),
            }

            # Add preference RM info if used
            if args.pref_rank_disc and args.pref_rm:
                expert_stats["pref_rm_path"] = args.pref_rm
                if hasattr(teacher_buffer, 'pref_episodes') and teacher_buffer.pref_episodes:
                    pref_scores = [ep['J'] for ep in teacher_buffer.pref_episodes]
                    expert_stats["pref_rm_score_mean"] = float(np.mean(pref_scores))
                    expert_stats["pref_rm_score_std"] = float(np.std(pref_scores))
                    expert_stats["pref_rm_score_min"] = float(np.min(pref_scores))
                    expert_stats["pref_rm_score_max"] = float(np.max(pref_scores))

            wandb_run.config.update(expert_stats, allow_val_change=True)

            print(f"[train_sail] Wandb initialized: project={wandb_project}, name={wandb_name}")

        except Exception as e:
            print(f"[WARNING] Failed to initialize wandb: {e}")
            wandb_run = None
    else:
        print("[train_sail] Wandb not available, skipping wandb logging")
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    # Setup callbacks
    callbacks = []
    if args.adaptive:
        adaptive_cb = SAILAdaptiveCallback(
            teacher_buffer=teacher_buffer,
            expert_scores_list=expert_scores,
            gamma=args.gamma,
            debug=args.debug,
            verbose=1,
            score_source=args.adaptive_score_source,
            pref_rm=teacher_buffer.pref_rm if args.adaptive_score_source == "rm" else None,
            rm_expert_scores=rm_expert_scores,
        )
        callbacks.append(adaptive_cb)

    print(f"[train_sail] Starting training: {args.total_timesteps} timesteps")
    print(f"[train_sail] TD3 HPs: batch={args.batch_size}  lr={args.learning_rate}"
          f"  train_freq={args.train_freq}  grad_steps={args.gradient_steps}"
          f"  learning_starts={args.learning_starts}")
    print(f"[train_sail] Disc HPs: train_freq={args.disc_train_freq}"
          f"  grad_steps={args.disc_gradient_steps}  batch={args.disc_batch_size}"
          f"  entcoeff={args.entcoeff}  gradcoeff={args.gradcoeff}")

    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1,
        callback=callbacks
    )
    print("[train_sail] Training finished successfully!")

    # ------------------------------------------------------------------
    # 6. Finish wandb run
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            # Log final summary statistics (matching TF implementation)
            wandb_run.summary["expert_return_mean"] = float(np.mean(returns))
            wandb_run.summary["expert_return_std"] = float(np.std(returns))
            wandb_run.finish()
            print("[train_sail] Wandb run finished")
        except Exception as e:
            print(f"[WARNING] Failed to finish wandb run: {e}")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
