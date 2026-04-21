"""
OnlineRMManager — owns the online preference RM, the segment store,
the training schedule, the three-gate activation policy, and the
rescoring logic for pref_episodes.

This is the single interface that sail.py and callbacks.py use.
Neither file needs to know about SegmentStore or OnlinePrefRewardModel directly.

Activation policy (three gates — ALL must be open simultaneously):
    Gate 1 (data):     store.size >= min_segments
    Gate 2 (training): rm_update_count >= min_rm_updates
    Gate 3 (quality):  held_out_pair_accuracy >= min_rm_accuracy

When is_active=False:  teacher weights remain uniform; no rescoring.
When is_active=True:   rescore_pref_episodes() is called on rescore_freq schedule.

Safe deactivation: if held_out accuracy drops below (min_rm_accuracy - hysteresis),
is_active reverts to False until the quality gate re-opens.
"""

import numpy as np
import torch
from typing import List, Dict, Optional

from sail_sb3.datasets.segment_pref_buffer import SegmentStore, SOURCE_TEACHER, SOURCE_STUDENT
from sail_sb3.reward_models.online_pref_rm import OnlinePrefRewardModel


class OnlineRMManager:
    """
    Manages the online preference reward model lifecycle during SAIL training.

    Lifecycle:
    1. __init__:
       - Builds SegmentStore and OnlinePrefRewardModel.
       - Segments all teacher episodes once and caches them.
       - Builds a fixed held-out pair set for quality evaluation.

    2. add_student_episode(obs_ep, act_ep, rew_ep):
       - Called by SAILAdaptiveCallback at each episode end.
       - Adds student segments to SegmentStore.

    3. maybe_train(global_step):
       - Called by SAIL._store_transition() every env step.
       - Checks rm_train_freq; if triggered, runs rm_gradient_steps RM updates.
       - Checks eval_freq; if triggered, evaluates held-out accuracy.
       - Updates is_active based on gate states.
       - Checks rescore_freq; if triggered AND is_active, calls rescore_pref_episodes.

    4. rescore_pref_episodes(pref_episodes):
       - Re-runs reward() over all episodes in teacher_buffer.pref_episodes.
       - Updates ep['J'] in-place for each episode.
       - Caller (sail.py) then calls teacher_buffer._recompute_pref_weights().

    5. reward(obs_np, act_np):
       - Drop-in interface for offline RM queries.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        # Segment buffer config
        max_segments: int = 10000,
        segment_len: int = 50,
        # RM architecture
        hidden_size: int = 256,
        rm_lr: float = 3e-4,
        rm_weight_decay: float = 1e-4,
        # Training schedule
        rm_train_freq: int = 1000,
        rm_gradient_steps: int = 10,
        rm_batch_size: int = 256,
        # Activation gates
        min_segments: int = 500,
        min_rm_updates: int = 50,
        min_rm_accuracy: float = 0.60,
        activation_hysteresis: float = 0.05,
        # Sampling and pairing
        tie_margin: float = 0.5,
        pair_strategy: str = 'any',
        # Evaluation and rescoring
        eval_freq: int = 10,          # evaluate held-out every N RM updates
        rescore_freq: int = 20000,    # rescore pref_episodes every N env steps
        n_held_out_pairs: int = 100,
        device: torch.device = None,
        # Expert data (for teacher segmentation at startup)
        expert_obs: np.ndarray = None,
        expert_act: np.ndarray = None,
        expert_rew: np.ndarray = None,
        expert_dones: np.ndarray = None,
    ):
        self.obs_dim          = obs_dim
        self.act_dim          = act_dim
        self.segment_len      = segment_len
        self.rm_train_freq    = rm_train_freq
        self.rm_gradient_steps = rm_gradient_steps
        self.rm_batch_size    = rm_batch_size
        self.min_segments     = min_segments
        self.min_rm_updates   = min_rm_updates
        self.min_rm_accuracy  = min_rm_accuracy
        self.activation_hysteresis = activation_hysteresis
        self.tie_margin       = tie_margin
        self.pair_strategy    = pair_strategy
        self.eval_freq        = eval_freq
        self.rescore_freq     = rescore_freq
        self.n_held_out_pairs = n_held_out_pairs
        self.device           = device if device is not None else torch.device("cpu")

        # Core components
        self.segment_store = SegmentStore(
            max_segments=max_segments,
            segment_len=segment_len,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=self.device,
        )
        self.rm = OnlinePrefRewardModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            lr=rm_lr,
            weight_decay=rm_weight_decay,
            device=self.device,
        )

        # Activation state
        self._is_active   = False
        self._rm_update_count = 0
        self._held_out_acc: float = 0.0
        self._last_rm_loss: float = float('nan')

        # Rescoring tracking
        self._last_rescore_step: int = -rescore_freq   # ensure first rescore happens on schedule
        self._rescore_count: int = 0

        # Held-out pair set (built after teacher segmentation)
        self._held_out_batch: Optional[dict] = None

        # Step tracking to avoid re-triggering on same step
        self._last_train_step: int = -rm_train_freq

        # Teacher segmentation — runs once at startup
        self._n_teacher_segments = 0
        if expert_obs is not None:
            self._segment_teacher_episodes(
                expert_obs, expert_act, expert_rew, expert_dones)
            self._build_held_out_pairs()
            print(f"[OnlineRMManager] Teacher segmentation: "
                  f"{self._n_teacher_segments} segments from expert data")
            print(f"[OnlineRMManager] Held-out pairs: {n_held_out_pairs}")
            if self.segment_store.return_stats():
                s = self.segment_store.return_stats()
                print(f"[OnlineRMManager] Teacher segment return stats: "
                      f"min={s['ret_min']:.1f} mean={s['ret_mean']:.1f} max={s['ret_max']:.1f}")

    # ------------------------------------------------------------------
    # Teacher segmentation — called once at __init__
    # ------------------------------------------------------------------

    def _segment_teacher_episodes(
        self,
        expert_obs: np.ndarray,
        expert_act: np.ndarray,
        expert_rew: np.ndarray,
        expert_dones: np.ndarray,
    ) -> None:
        """
        Extract fixed-length segments from all expert episodes and add to store.

        expert_obs:   [N, obs_dim]
        expert_act:   [N, act_dim]
        expert_rew:   [N]
        expert_dones: [N]  (1.0 at episode end)
        """
        if expert_obs is None or len(expert_obs) == 0:
            return

        dones_np = np.asarray(expert_dones, dtype=np.float32).ravel()
        done_idx = np.where(dones_np == 1.0)[0]

        if len(done_idx) == 0:
            # No episode boundaries found — treat entire data as one episode
            n = self.segment_store.add_from_episode(
                expert_obs, expert_act, expert_rew, SOURCE_TEACHER)
            self._n_teacher_segments += n
            return

        start = 0
        for last in done_idx:
            obs_ep = expert_obs[start:last + 1]
            act_ep = expert_act[start:last + 1]
            rew_ep = expert_rew[start:last + 1]
            n = self.segment_store.add_from_episode(
                obs_ep, act_ep, rew_ep, SOURCE_TEACHER)
            self._n_teacher_segments += n
            start = last + 1

    # ------------------------------------------------------------------
    # Held-out pair set — built from teacher segments only
    # ------------------------------------------------------------------

    def _build_held_out_pairs(self) -> None:
        """
        Sample a fixed held-out pair set from teacher segments.
        These pairs are NEVER used for RM training; only for evaluation.
        """
        if self.segment_store.teacher_count < 2:
            self._held_out_batch = None
            return
        try:
            batch = self.segment_store.sample_pair_batch(
                batch_size=self.n_held_out_pairs,
                tie_margin=self.tie_margin,
                strategy='any',
            )
            # Move to device (already on device from SegmentStore)
            self._held_out_batch = batch
        except Exception as e:
            print(f"[OnlineRMManager] WARNING: Could not build held-out pairs: {e}")
            self._held_out_batch = None

    # ------------------------------------------------------------------
    # Public: add student data
    # ------------------------------------------------------------------

    def add_student_episode(
        self,
        obs_ep: np.ndarray,
        act_ep: np.ndarray,
        rew_ep: np.ndarray,
    ) -> int:
        """
        Add a completed student episode to the segment store.
        Returns number of segments added.
        """
        n = self.segment_store.add_from_episode(
            obs_ep, act_ep, rew_ep, SOURCE_STUDENT)
        return n

    # ------------------------------------------------------------------
    # Public: training trigger (called every env step)
    # ------------------------------------------------------------------

    def maybe_train(self, global_step: int) -> None:
        """
        Called from SAIL._store_transition() on every env step.
        Fires RM training when conditions are met.
        """
        # Gate 1: enough data
        if not self.segment_store.is_ready(self.min_segments):
            return

        # Check training frequency
        if global_step - self._last_train_step < self.rm_train_freq:
            return
        self._last_train_step = global_step

        # Run gradient steps
        losses = []
        for _ in range(self.rm_gradient_steps):
            try:
                batch = self.segment_store.sample_pair_batch(
                    batch_size=self.rm_batch_size,
                    tie_margin=self.tie_margin,
                    strategy=self.pair_strategy,
                )
                loss = self.rm.update(batch)
                losses.append(loss)
                self._rm_update_count += 1
            except Exception as e:
                print(f"[OnlineRMManager] WARNING: RM update failed at step {global_step}: {e}")
                break

        if losses:
            self._last_rm_loss = float(np.mean(losses))

        # Evaluate held-out accuracy periodically
        if self._rm_update_count > 0 and self._rm_update_count % self.eval_freq == 0:
            self._evaluate_held_out()
            self._update_activation_state()

        # Rescoring trigger — only when is_active
        if (self._is_active
                and global_step - self._last_rescore_step >= self.rescore_freq):
            # Gated by held-out accuracy to prevent noisy rescore
            if self._held_out_acc >= self.min_rm_accuracy:
                self._pending_rescore = True
            self._last_rescore_step = global_step

    # ------------------------------------------------------------------
    # Public: rescore pref_episodes (called from sail.py on rescore trigger)
    # ------------------------------------------------------------------

    def rescore_pref_episodes(self, pref_episodes: List[Dict]) -> int:
        """
        Re-score all episodes in pref_episodes with the current RM.
        Updates ep['J'] in-place.

        Called by SAIL._update_discriminator() when pending_rescore is set.
        Caller is responsible for calling teacher_buffer._recompute_pref_weights()
        afterward.

        Returns: number of episodes rescored.
        """
        if not pref_episodes:
            return 0
        if not self._is_active:
            return 0

        n_rescored = 0
        j_before = [ep['J'] for ep in pref_episodes]

        for ep in pref_episodes:
            try:
                obs_np = ep['obs'].cpu().numpy() if hasattr(ep['obs'], 'cpu') \
                         else np.asarray(ep['obs'], dtype=np.float32)
                act_np = ep['acs'].cpu().numpy() if hasattr(ep['acs'], 'cpu') \
                         else np.asarray(ep['acs'], dtype=np.float32)
                r = self.rm.reward(obs_np, act_np)  # [T]
                ep['J'] = float(np.sum(r))
                n_rescored += 1
            except Exception as e:
                print(f"[OnlineRMManager] WARNING: rescore failed for one episode: {e}")

        j_after = [ep['J'] for ep in pref_episodes]
        j_delta = np.mean(np.abs(np.array(j_after) - np.array(j_before)))

        self._rescore_count += 1
        self._pending_rescore = False

        print(f"[OnlineRMManager] Rescore #{self._rescore_count}: "
              f"{n_rescored} episodes updated, "
              f"mean |ΔJ|={j_delta:.2f}, "
              f"J std={np.std(j_after):.2f}")
        return n_rescored

    # ------------------------------------------------------------------
    # Public: reward interface — matches offline PrefRewardModel.reward()
    # ------------------------------------------------------------------

    def reward(self, obs_np: np.ndarray, act_np: np.ndarray) -> np.ndarray:
        """
        Per-step rewards for a trajectory.  Drop-in for PrefRewardModel.reward().

        Args:
            obs_np: [T, obs_dim]
            act_np: [T, act_dim]
        Returns:
            [T] float32 numpy array
        """
        return self.rm.reward(obs_np, act_np)

    # ------------------------------------------------------------------
    # Activation state management
    # ------------------------------------------------------------------

    def _evaluate_held_out(self) -> None:
        """Compute held-out pair accuracy with no gradient."""
        if self._held_out_batch is None:
            return
        try:
            self._held_out_acc = self.rm.evaluate_pairs(self._held_out_batch)
        except Exception as e:
            print(f"[OnlineRMManager] WARNING: held-out eval failed: {e}")

    def _update_activation_state(self) -> None:
        """Apply three-gate logic + hysteresis to update _is_active."""
        gate1 = self.segment_store.is_ready(self.min_segments)
        gate2 = self._rm_update_count >= self.min_rm_updates
        gate3 = self._held_out_acc >= self.min_rm_accuracy

        if not self._is_active:
            # Activate when ALL three gates open
            if gate1 and gate2 and gate3:
                self._is_active = True
                print(f"[OnlineRMManager] RM ACTIVATED: "
                      f"segments={len(self.segment_store)} "
                      f"updates={self._rm_update_count} "
                      f"held_out_acc={self._held_out_acc:.3f}")
        else:
            # Deactivate only if quality drops below threshold - hysteresis
            deact_threshold = self.min_rm_accuracy - self.activation_hysteresis
            if self._held_out_acc < deact_threshold:
                self._is_active = False
                print(f"[OnlineRMManager] RM DEACTIVATED (quality drop): "
                      f"held_out_acc={self._held_out_acc:.3f} < {deact_threshold:.3f}")

    # ------------------------------------------------------------------
    # Properties and diagnostics
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when all three activation gates are open."""
        return self._is_active

    @property
    def pending_rescore(self) -> bool:
        """True when a rescore of pref_episodes is due."""
        return getattr(self, '_pending_rescore', False)

    @property
    def rm_update_count(self) -> int:
        return self._rm_update_count

    def get_metrics(self) -> dict:
        """Return metrics dict for logging in SAIL.train()."""
        return {
            'online_rm/loss':               self._last_rm_loss,
            'online_rm/updates':            self._rm_update_count,
            'online_rm/held_out_acc':       self._held_out_acc,
            'online_rm/is_active':          int(self._is_active),
            'online_rm/segment_store_size': len(self.segment_store),
            'online_rm/teacher_count':      self.segment_store.teacher_count,
            'online_rm/student_count':      self.segment_store.student_count,
            'online_rm/rescore_count':      self._rescore_count,
        }

    def gate_status(self) -> str:
        """Human-readable gate status string for debugging."""
        g1 = self.segment_store.is_ready(self.min_segments)
        g2 = self._rm_update_count >= self.min_rm_updates
        g3 = self._held_out_acc >= self.min_rm_accuracy
        return (f"G1(data:{len(self.segment_store)}>={self.min_segments})={'OPEN' if g1 else 'CLOSED'} "
                f"G2(updates:{self._rm_update_count}>={self.min_rm_updates})={'OPEN' if g2 else 'CLOSED'} "
                f"G3(acc:{self._held_out_acc:.3f}>={self.min_rm_accuracy:.3f})={'OPEN' if g3 else 'CLOSED'}")
