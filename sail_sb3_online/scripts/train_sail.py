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
from stable_baselines3.common.logger import configure, Logger, KVWriter, make_output_format
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


class WandbOutputFormat(KVWriter):
    """
    SB3 KVWriter that sends every logger.dump(step=N) directly to WandB as
    wandb.log({metrics}, step=N).  This uses num_timesteps as the x-axis,
    fixing the broken _step counter produced by wandb.tensorboard.patch()
    (which increments _step once per scalar, not once per dump call).
    """
    def write(self, key_values, key_excluded, step: int = 0) -> None:
        try:
            import wandb as _wandb
            if _wandb.run is None:
                return
            metrics = {}
            for k, v in key_values.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
                elif hasattr(v, 'item'):
                    metrics[k] = float(v.item())
            if metrics:
                _wandb.log(metrics, step=int(step), commit=True)
        except Exception:
            pass

    def close(self) -> None:
        pass

# Per-environment hardcoded expert returns (from actual teacher checkpoints).
# Used for normalized score logging: score = policy_return / expert_return.
# Resolution order: --expert_return CLI arg > this dict > None (disabled).
EXPERT_RETURNS = {
    "HalfCheetah-v2": 9094.0,
    "Walker2d-v2":    4717.0,
    "Hopper-v2":      3606.0,
    "Ant-v2":         5813.0,
    "Swimmer-v2":     359.0,
}


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

    # ---- Preference Reweighted Teacher (PR-SAIL) ----
    parser.add_argument("--pref_reweight_teacher", action="store_true",
                        help="Enable Boltzmann-weighted discriminator expert batch (PR-SAIL). "
                             "Requires --pref_rm. Episodes with higher RM score J_phi(tau) "
                             "receive higher weight in the expert BCE loss. "
                             "TF parity: pref_reweight_teacher flag in sail.py.")
    parser.add_argument("--pref_beta",             type=float, default=1.0,
                        help="Temperature for Boltzmann weighting: softmax(J / beta). "
                             "Larger beta = flatter (more uniform), smaller = sharper "
                             "(concentrate on top episodes). Only used when "
                             "--pref_reweight_teacher is set. TF default: 1.0.")
    parser.add_argument("--pref_max_teacher_trajs", type=int,  default=500,
                        help="Max episodes in pref pool before quantile-based pruning. "
                             "TF default: 500. Only used when --pref_reweight_teacher is set.")
    parser.add_argument("--pref_promote_quantile",  type=float, default=0.75,
                        help="Quantile threshold for pruning low-J episodes from the pref pool. "
                             "Episodes with J < quantile(scores, q) are dropped. "
                             "TF default: 0.75. Only used when --pref_reweight_teacher is set.")

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

    # ---- QPREF: Q-preference ranking loss on TD3 critic ----
    parser.add_argument("--qpref",               action="store_true",
                        help="Enable Q-preference ranking loss on the TD3 critic.")
    parser.add_argument("--qpref_weight",         type=float, default=0.0,
                        help="Lambda for QPREF loss. Auto-defaults to 0.1 if --qpref is set "
                             "without an explicit weight.")
    parser.add_argument("--qpref_temp",           type=float, default=1.0,
                        help="Temperature T: loss = mean(softplus(-(meanQ_pos - meanQ_neg)/T)).")
    parser.add_argument("--qpref_batch_size",     type=int,   default=4,
                        help="Trajectory pairs per critic gradient step. "
                             "Default 4 (trajectory-mean-Q is ~T_max x more expensive than single-step).")
    parser.add_argument("--qpref_start_step",     type=int,   default=0,
                        help="Apply QPREF only after this many env steps (0 = from first update).")
    parser.add_argument("--qpref_source",         type=str,   default="teacher",
                        choices=["teacher", "student"],
                        help="Episode pool for QPREF pairs: "
                             "'teacher' (pref_episodes, ready from step 0) or "
                             "'student' (built from rollouts).")
    parser.add_argument("--pref_max_student_trajs",  type=int, default=500,
                        help="Max student episodes in QPREF student pool (pruned by recency).")
    parser.add_argument("--qpref_grad_interval",     type=int, default=10,
                        help="Apply QPREF loss every Nth critic gradient step. "
                             "Default 10: with gradient_steps=1000, QPREF fires 100×/train() call. "
                             "Reduces CPU cost ~10× vs firing every step.")
    parser.add_argument("--qpref_guard_threshold",  type=float, default=-0.3,
                        help="Rolling-mean delta threshold for the QPREF safety guard. "
                             "When guard_mean (mean of last --qpref_guard_window train() deltas) drops "
                             "below this value for --qpref_guard_confirm consecutive train() calls, "
                             "QPREF is permanently disabled.")
    parser.add_argument("--qpref_guard_confirm",    type=int, default=3,
                        help="Number of consecutive train() calls with guard_mean below "
                             "--qpref_guard_threshold required to trigger the QPREF guard. "
                             "Resets to 0 on any train() call where guard_mean >= threshold.")
    parser.add_argument("--qpref_guard_window",     type=int, default=10,
                        help="Number of most-recent deque entries used for guard decision. "
                             "Smaller = more reactive; larger = more stable. Default 10.")
    parser.add_argument("--qpref_guard_positive_threshold", type=float, default=1.0,
                        help="Minimum peak guard_mean required before the guard can trigger. "
                             "Prevents disabling QPREF during early critic instability that was "
                             "never preceded by a genuine positive phase. Guard requires "
                             "peak_guard_mean >= this value AND current guard_mean < guard_threshold.")

    # ---- Soft-TAC: tanh discriminator alignment with RM-derived preference labels ----
    parser.add_argument("--soft_tac",           action="store_true",
                        help="Enable Soft-TAC loss on discriminator. "
                             "Requires --pref_rm. Compatible with pref_reweight_teacher, qpref, adaptive. "
                             "TF reference: --pref-soft-rank-disc + --pref-soft-rank-weight.")
    parser.add_argument("--soft_tac_weight",    type=float, default=0.0,
                        help="Weight for Soft-TAC loss. Auto-defaults to 0.5 if --soft_tac set "
                             "without an explicit weight. TF reference: --pref-soft-rank-weight.")
    parser.add_argument("--soft_tac_temp",      type=float, default=1.0,
                        help="Temperature T for tanh(delta_J/T) in Soft-TAC. "
                             "Lower T = steeper alignment (harder to satisfy). "
                             "TF reference: --pref-soft-rank-temp.")
    parser.add_argument("--tac_tie_eps",        type=float, default=0.0,
                        help="Tie margin epsilon: pairs with |J_rm_pos - J_rm_neg| <= eps "
                             "produce y=0 (zero gradient). TF reference: --pref-tac-tie-eps.")
    parser.add_argument("--soft_tac_max_student_trajs", type=int, default=200,
                        help="Max student episodes in the Soft-TAC dedicated pool (recency-pruned). "
                             "Expert episodes are permanent. Default 200.")

    # ---- Online Reward Model ----
    parser.add_argument("--online_rm",           action="store_true",
                        help="Enable online preference reward model training (pure online pipeline, "
                             "no offline RM required). Trains a reward model in parallel with SAIL "
                             "using true environment returns as preference labels over segments.")
    parser.add_argument("--rm_segment_len",      type=int,   default=50,
                        help="Fixed segment length (timesteps) for online RM preference pairs.")
    parser.add_argument("--rm_train_freq",       type=int,   default=1000,
                        help="Train online RM every N env steps.")
    parser.add_argument("--rm_gradient_steps",   type=int,   default=10,
                        help="RM gradient steps per training trigger.")
    parser.add_argument("--rm_batch_size",       type=int,   default=256,
                        help="Preference pair batch size for RM training.")
    parser.add_argument("--rm_lr",               type=float, default=3e-4,
                        help="AdamW learning rate for online RM.")
    parser.add_argument("--rm_max_segments",     type=int,   default=10000,
                        help="Ring buffer capacity of the segment store.")
    parser.add_argument("--rm_min_segments",     type=int,   default=500,
                        help="Gate 1: minimum segments before RM training starts.")
    parser.add_argument("--rm_min_updates",      type=int,   default=50,
                        help="Gate 2: minimum RM updates before activation.")
    parser.add_argument("--rm_min_acc",          type=float, default=0.60,
                        help="Gate 3: minimum held-out pair accuracy before RM is considered active.")
    parser.add_argument("--rm_tie_margin",       type=float, default=0.5,
                        help="Minimum |ret_1 - ret_2| to form a non-tie preference pair.")
    parser.add_argument("--rm_rescore_freq",     type=int,   default=20000,
                        help="Rescore pref_episodes with online RM every N env steps (once active).")

    # ---- Debug ----
    parser.add_argument("--debug",           action="store_true",
                        help="Enable per-step discriminator / reward / Q debug output")

    # ---- Normalized score ----
    parser.add_argument("--expert_return", type=float, default=None,
                        help="Expert policy return used to compute normalized score "
                             "(normalized_score = policy_return / expert_return). "
                             "Overrides per-env hardcoded value in EXPERT_RETURNS dict. "
                             "If not provided and env has no entry in EXPERT_RETURNS, "
                             "normalized score logging is silently disabled.")
    args = parser.parse_args()

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_sail] Using device: {device}")

    # Resolve expert_return: CLI arg takes priority, then per-env dict, then None.
    expert_return = args.expert_return if args.expert_return is not None \
        else EXPERT_RETURNS.get(args.env)
    expert_return_source = "cli" if args.expert_return is not None \
        else ("dict" if expert_return is not None else None)
    if expert_return is not None:
        print(f"[train_sail] Normalized score enabled: expert_return={expert_return:.1f}"
              f"  (source={expert_return_source})")
    else:
        print("[train_sail] Normalized score disabled (no expert_return for this env)")

    # ------------------------------------------------------------------
    # W&B: init early so wandb_run is available when building the logger.
    # WandbOutputFormat (added to the SB3 logger below) calls wandb.log()
    # with step=num_timesteps at each dump, giving the correct x-axis.
    # ------------------------------------------------------------------
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_project = os.getenv("WANDB_PROJECT", f"SAIL_SB3_{args.env.split('-')[0]}")
            wandb_name    = os.getenv("WANDB_NAME",    f"SAIL_{args.env}_s{args.seed}")
            wandb_group   = os.getenv("WANDB_GROUP",   f"{args.env}_vanilla_{args.total_timesteps}")
            wandb_entity  = os.getenv("WANDB_ENTITY",  None)
            os.environ.setdefault("WANDB_SILENT", "true")

            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                group=wandb_group,
                entity=wandb_entity,
                config=vars(args),
                reinit=True,
            )
            print(f"[train_sail] Wandb initialized: project={wandb_project}, name={wandb_name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize wandb (early init): {e}")
            wandb_run = None
    else:
        print("[train_sail] Wandb not available, skipping wandb logging")

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
    # Load RM if needed for pref ranking, RM-based adaptive promotion, or reweighting
    # Auto-default qpref_weight to 0.1 when --qpref is set without an explicit weight
    if args.qpref and args.qpref_weight <= 0.0:
        args.qpref_weight = 0.1
        print(f"[train_sail] QPREF: qpref_weight not set, defaulting to 0.1")

    # Auto-default soft_tac_weight to 0.5 when --soft_tac is set without an explicit weight
    if args.soft_tac and args.soft_tac_weight <= 0.0:
        args.soft_tac_weight = 0.5
        print(f"[train_sail] Soft-TAC: soft_tac_weight not set, defaulting to 0.5")

    # When --online_rm is active, it replaces the offline pref_rm for:
    #   pref_reweight_teacher, pref_rank_disc, soft_tac, qpref (student source)
    # so we do NOT require --pref_rm for those paths.
    _is_online_rm = getattr(args, 'online_rm', False)
    _reweight_needs_pref_rm  = args.pref_reweight_teacher and not _is_online_rm
    _pref_rank_needs_pref_rm = args.pref_rank_disc        and not _is_online_rm
    _soft_tac_needs_pref_rm  = args.soft_tac              and not _is_online_rm
    _qpref_needs_pref_rm     = args.qpref                 and not _is_online_rm
    need_pref_rm = (
        _pref_rank_needs_pref_rm
        or (args.adaptive_score_source == "rm")
        or _reweight_needs_pref_rm
        or _qpref_needs_pref_rm
        or _soft_tac_needs_pref_rm
    )
    # Build the pref pool from GT returns INSIDE __init__ BEFORE ring truncation whenever
    # online_rm is paired with any feature that reads pref_episodes (reweight, pref_rank_disc,
    # soft_tac).  This ensures pool_size=4 at startup regardless of ring-buffer truncation.
    _pref_build_gt = _is_online_rm and (
        args.pref_reweight_teacher or args.pref_rank_disc or args.soft_tac
    )
    # Pool cap: apply pref_max_teacher_trajs whenever the pref pool is actually used.
    _any_pref_pool = args.pref_reweight_teacher or (
        _is_online_rm and (args.pref_rank_disc or args.soft_tac)
    )
    teacher_buffer = TeacherBuffer(
        args.expert_data,
        device,
        pref_rm_path=args.pref_rm if need_pref_rm else None,
        expect_obs_dim=args.pref_expect_obs_dim,
        max_size=args.teacher_buffer_size,
        pref_max_teacher_trajs=args.pref_max_teacher_trajs if _any_pref_pool else None,
        pref_promote_quantile=args.pref_promote_quantile,
        pref_max_student_trajs=args.pref_max_student_trajs,
        soft_tac_max_student_trajs=args.soft_tac_max_student_trajs,
        pref_build_gt=_pref_build_gt,
    )
    # Set Boltzmann temperature for pref reweighting (used by _recompute_pref_weights).
    if args.pref_reweight_teacher:
        teacher_buffer._pref_reweight_beta = args.pref_beta
        # Recompute weights with the user-supplied beta (initial build used default beta=1.0).
        if teacher_buffer.pref_episodes:
            teacher_buffer._recompute_pref_weights(beta=args.pref_beta)
            w = teacher_buffer.pref_teacher_weights
            print(f"[train_sail] PrefReweight: beta={args.pref_beta}  "
                  f"pool={len(teacher_buffer.pref_episodes)} eps  "
                  f"weights min={w.min():.4f} max={w.max():.4f} sum={w.sum():.4f}")
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

    if args.qpref:
        n_teacher_pool = len(teacher_buffer.pref_episodes)
        print(f"[train_sail] QPREF: source={args.qpref_source} weight={args.qpref_weight} "
              f"temp={args.qpref_temp} batch={args.qpref_batch_size} "
              f"start_step={args.qpref_start_step}  teacher_pool={n_teacher_pool} eps")
        print(f"[train_sail] QPREF guard: threshold={args.qpref_guard_threshold} "
              f"confirm={args.qpref_guard_confirm} window={args.qpref_guard_window} "
              f"positive_threshold={args.qpref_guard_positive_threshold}")
        if n_teacher_pool < 2 and args.qpref_source == "teacher":
            print("[train_sail] QPREF WARNING: teacher pool has < 2 episodes — "
                  "QPREF will be silent until pool grows (requires --pref_rm).")

    if args.soft_tac:
        n_tac_pool = len(teacher_buffer.soft_tac_pool)
        print(f"[train_sail] Soft-TAC: weight={args.soft_tac_weight} temp={args.soft_tac_temp} "
              f"tie_eps={args.tac_tie_eps}  soft_tac_pool={n_tac_pool} eps (expert-only at startup; "
              f"grows with every student episode)")
        if n_tac_pool < 2:
            print("[train_sail] Soft-TAC WARNING: soft_tac_pool has < 2 episodes — "
                  "Soft-TAC will be silent until pool grows. Ensure --pref_rm is set.")

    # ------------------------------------------------------------------
    # Online RM manager construction (Phase 3b)
    # ------------------------------------------------------------------
    online_rm_manager = None
    if getattr(args, 'online_rm', False):
        from sail_sb3.reward_models.online_rm_manager import OnlineRMManager
        # Use full pre-truncation arrays cached by TeacherBuffer.__init__ when available
        # (pref_build_gt=True path). Falls back to post-truncation data for non-reweight runs.
        if (hasattr(teacher_buffer, '_full_expert_obs_np')
                and teacher_buffer._full_expert_obs_np is not None):
            _expert_obs_np   = teacher_buffer._full_expert_obs_np
            _expert_act_np   = teacher_buffer._full_expert_act_np
            _expert_rew_np   = teacher_buffer._full_expert_rew_np
            _expert_dones_np = teacher_buffer._full_expert_dones_np
        else:
            _expert_obs_np   = teacher_buffer.states.cpu().numpy()
            _expert_act_np   = teacher_buffer.actions.cpu().numpy()
            _expert_rew_np   = teacher_buffer.rewards.cpu().numpy().ravel()
            _expert_dones_np = teacher_buffer.dones.cpu().numpy().ravel()
        online_rm_manager = OnlineRMManager(
            obs_dim=state_dim,
            act_dim=action_dim,
            max_segments=args.rm_max_segments,
            segment_len=args.rm_segment_len,
            rm_lr=args.rm_lr,
            rm_train_freq=args.rm_train_freq,
            rm_gradient_steps=args.rm_gradient_steps,
            rm_batch_size=args.rm_batch_size,
            min_segments=args.rm_min_segments,
            min_rm_updates=args.rm_min_updates,
            min_rm_accuracy=args.rm_min_acc,
            tie_margin=args.rm_tie_margin,
            rescore_freq=args.rm_rescore_freq,
            device=device,
            expert_obs=_expert_obs_np,
            expert_act=_expert_act_np,
            expert_rew=_expert_rew_np,
            expert_dones=_expert_dones_np,
        )
        print(f"[train_sail] OnlineRMManager constructed: "
              f"segment_len={args.rm_segment_len}  train_freq={args.rm_train_freq}  "
              f"min_segments={args.rm_min_segments}  min_acc={args.rm_min_acc}")
        # Pref pool is now built inside TeacherBuffer.__init__ via pref_build_gt=True
        # (all expert episodes with GT J, before ring truncation). Log current state.
        if teacher_buffer.pref_episodes:
            n_eps = len(teacher_buffer.pref_episodes)
            j_vals = [ep['J'] for ep in teacher_buffer.pref_episodes]
            print(f"[train_sail] OnlineRM pref pool: {n_eps} eps (GT J, pre-truncation)  "
                  f"J mean={np.mean(j_vals):.1f} spread={max(j_vals)-min(j_vals):.1f}")
            if args.pref_rank_disc:
                print(f"[train_sail] pref_rank_disc: will use online RM J for pair ordering "
                      f"(GT before activation, RM after rescore)")
            if args.soft_tac:
                print(f"[train_sail] soft_tac: will use online RM J for y=sign(J_pos-J_neg) "
                      f"(GT before activation, RM after rescore)")
            if args.pref_reweight_teacher:
                teacher_buffer._recompute_pref_weights(beta=args.pref_beta)
                w = teacher_buffer.pref_teacher_weights
                print(f"[train_sail] pref_reweight: beta={args.pref_beta}  "
                      f"weights min={w.min():.4f} max={w.max():.4f}")

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
        use_expert_weights=args.pref_reweight_teacher,
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
        pref_reweight_teacher=args.pref_reweight_teacher,
        adaptive=args.adaptive,
        expert_scores=expert_scores if args.adaptive else None,
        lfd_mixing=args.lfd_mixing,
        debug=args.debug,
        # QPREF
        qpref=args.qpref,
        qpref_weight=args.qpref_weight,
        qpref_temp=args.qpref_temp,
        qpref_batch_size=args.qpref_batch_size,
        qpref_start_step=args.qpref_start_step,
        qpref_source=args.qpref_source,
        qpref_mean_trajectory_q=True,   # active implementation path
        qpref_grad_interval=args.qpref_grad_interval,
        qpref_guard_threshold=args.qpref_guard_threshold,
        qpref_guard_confirm=args.qpref_guard_confirm,
        qpref_guard_window=args.qpref_guard_window,
        qpref_guard_positive_threshold=args.qpref_guard_positive_threshold,
        # Soft-TAC
        soft_tac=args.soft_tac,
        soft_tac_weight=args.soft_tac_weight,
        soft_tac_temp=args.soft_tac_temp,
        tac_tie_eps=args.tac_tie_eps,
        online_rm_manager=online_rm_manager,
        # Normalized score
        expert_return=expert_return,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./sail_tensorboard/",
    )

    os.makedirs("./sail_logs/", exist_ok=True)
    _output_formats = [make_output_format(f, "./sail_logs/", "") for f in ["stdout", "csv", "tensorboard"]]
    if WANDB_AVAILABLE and wandb_run is not None:
        _output_formats.append(WandbOutputFormat())
    new_logger = Logger(folder="./sail_logs/", output_formats=_output_formats)
    model.set_logger(new_logger)

    # ------------------------------------------------------------------
    # 5. Weights & Biases — log expert dataset stats to config
    #    (wandb was already init'd early, before configure() / SAIL constructor)
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            expert_stats = {
                "expert_dataset_path": args.expert_data,
                "expert_episode_count": len(returns),
                "expert_return_mean": float(np.mean(returns)),
                "expert_return_std": float(np.std(returns)),
                "expert_return_min": float(np.min(returns)),
                "expert_return_max": float(np.max(returns)),
                "expert_total_transitions": teacher_buffer.size(),
            }
            if args.pref_rank_disc and args.pref_rm:
                expert_stats["pref_rm_path"] = args.pref_rm
                if hasattr(teacher_buffer, 'pref_episodes') and teacher_buffer.pref_episodes:
                    pref_scores = [ep['J'] for ep in teacher_buffer.pref_episodes]
                    expert_stats["pref_rm_score_mean"] = float(np.mean(pref_scores))
                    expert_stats["pref_rm_score_std"] = float(np.std(pref_scores))
                    expert_stats["pref_rm_score_min"] = float(np.min(pref_scores))
                    expert_stats["pref_rm_score_max"] = float(np.max(pref_scores))
            if expert_return is not None:
                expert_stats["expert_return_for_norm"] = float(expert_return)
                expert_stats["expert_return_source"]   = expert_return_source
            wandb_run.config.update(expert_stats, allow_val_change=True)
        except Exception as e:
            print(f"[WARNING] Failed to update wandb config with expert stats: {e}")
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
            qpref_source=args.qpref_source if args.qpref else "teacher",
            soft_tac=args.soft_tac,
            online_rm_manager=online_rm_manager,
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
    # Final normalized score (using last ep_info_buffer window as final return)
    # ------------------------------------------------------------------
    final_policy_return = None
    final_normalized_score = None
    final_normalized_score_pct = None
    if len(model.ep_info_buffer) > 0:
        from stable_baselines3.common.utils import safe_mean as _safe_mean
        final_policy_return = float(_safe_mean(
            [ep_info["r"] for ep_info in model.ep_info_buffer]))
        print(f"[train_sail] Final policy return (ep_info_buffer mean): "
              f"{final_policy_return:.2f}")
        if expert_return is not None and expert_return != 0.0 \
                and np.isfinite(final_policy_return):
            final_normalized_score     = final_policy_return / expert_return
            final_normalized_score_pct = final_normalized_score * 100.0
            print(f"[train_sail] Final normalized score: "
                  f"{final_normalized_score:.4f}  "
                  f"({final_normalized_score_pct:.1f}% of expert)")

    # ------------------------------------------------------------------
    # 6. Finish wandb run
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            # Log final summary statistics (matching TF implementation)
            wandb_run.summary["expert_return_mean"] = float(np.mean(returns))
            wandb_run.summary["expert_return_std"] = float(np.std(returns))
            if final_policy_return is not None:
                wandb_run.summary["final/final_policy_return"]       = final_policy_return
            if final_normalized_score is not None:
                wandb_run.summary["final/final_normalized_score"]     = final_normalized_score
                wandb_run.summary["final/final_normalized_score_pct"] = final_normalized_score_pct
            wandb_run.finish()
            print("[train_sail] Wandb run finished")
        except Exception as e:
            print(f"[WARNING] Failed to finish wandb run: {e}")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
