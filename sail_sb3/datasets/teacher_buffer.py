import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sail_sb3.datasets.expert_loader import load_expert_npz

class TeacherBuffer(Dataset):
    """
    A PyTorch Dataset that loads expert/teacher demonstrations into memory.
    Provides an interface to sample minibatches of (state, action) for the discriminator.

    Supports dynamic growth (adaptive SAIL/PAIL):
    - Initialized with expert data from NPZ
    - Can dynamically add student episodes via add_episode()
    - Grows teacher buffer when student exceeds expert threshold
    """
    def __init__(self, data_path: str, device: torch.device, pref_rm_path: str = None,
                 expect_obs_dim: int = 17, max_size: int = None,
                 pref_max_teacher_trajs: int = None, pref_promote_quantile: float = 0.75,
                 pref_max_student_trajs: int = 500, soft_tac_max_student_trajs: int = 200):
        self.data_path = data_path
        self.device = device
        self.pref_rm_path = pref_rm_path
        self.max_size = max_size
        self.pref_max_teacher_trajs = pref_max_teacher_trajs
        self.pref_promote_quantile = pref_promote_quantile
        self.pref_max_student_trajs = pref_max_student_trajs
        self.soft_tac_max_student_trajs = soft_tac_max_student_trajs

        # Load structured dictionary from the NPZ utility
        parsed_data = load_expert_npz(data_path)

        # Convert numpy arrays to PyTorch tensors and move to device
        self.states = torch.tensor(parsed_data['observations'], dtype=torch.float32).to(self.device)
        self.actions = torch.tensor(parsed_data['actions'], dtype=torch.float32).to(self.device)
        self.num_transitions = len(self.states)

        if 'dones' in parsed_data:
            self.dones = torch.tensor(parsed_data['dones'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            self.dones = torch.zeros(self.num_transitions, 1, dtype=torch.float32).to(self.device)

        if 'rewards' in parsed_data:
            self.rewards = torch.tensor(parsed_data['rewards'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            self.rewards = torch.zeros(self.num_transitions, 1, dtype=torch.float32).to(self.device)

        # next_states: required for LfD mixing (Bellman target needs s').
        # Built by expert_loader when episode_starts is present.
        # Falls back to a self-loop (next = current obs) if unavailable —
        # safe because terminal transitions are masked by (1-done) in the target.
        if 'next_observations' in parsed_data:
            self.next_states = torch.tensor(
                parsed_data['next_observations'], dtype=torch.float32).to(self.device)
        else:
            # Fallback: shift obs by 1, wrap last with self-loop
            obs_np = parsed_data['observations']
            next_obs_np = np.empty_like(obs_np)
            next_obs_np[:-1] = obs_np[1:]
            next_obs_np[-1] = obs_np[-1]
            self.next_states = torch.tensor(next_obs_np, dtype=torch.float32).to(self.device)

        # Preference RM + separate pref pool — built from FULL expert data BEFORE ring truncation.
        # This ensures all original expert episodes are in the pref pool regardless of ring size.
        # The pref pool is episode-level and grows independently of the transition-level ring buffer:
        #   - Initialized here with all N_expert episodes from the full NPZ
        #   - add_episode() appends promoted student episodes to this same pool
        #   - sample_pref_pairs() samples from this pool for the PrefRank discriminator loss
        # The transition ring buffer (states/actions/etc.) is truncated below and used only for
        # GAIL/LfD disc training and critic LfD mixing — it is NOT used for pref pair sampling.
        self.pref_rm = None
        self.pref_episodes = []
        # Boltzmann weights over pref_episodes; recomputed after every pool change.
        # None until _recompute_pref_weights() is called (requires >= 1 episode).
        self.pref_teacher_weights = None   # np.array[N_eps], softmax(J/beta), sums to ~1
        self._pref_reweight_beta = 1.0     # temperature; overwritten by train_sail.py
        # Soft-TAC dedicated pool — must be initialized BEFORE _build_pref_episodes() so the
        # assignment inside that method is not overwritten by later __init__ code.
        self._soft_tac_expert_episodes = []   # populated in _build_pref_episodes()
        self.soft_tac_student_episodes = []   # populated by add_soft_tac_student_episode()
        if pref_rm_path:
            import os
            if os.path.exists(pref_rm_path):
                from sail_sb3.reward_models.pref_rm_eval import PrefRewardModel
                self.pref_rm = PrefRewardModel(pref_rm_path, device=str(self.device), expect_obs_dim=expect_obs_dim)
                self._build_pref_episodes()  # Uses full data — all N_expert episodes scored
            else:
                print(f"[TeacherBuffer] WARNING: RM path {pref_rm_path} does not exist.")

        # Ring buffer: if max_size set and expert data exceeds it, keep only the
        # last max_size transitions for GAIL/LfD use.  Matches TF ReplayBufferExtend
        # semantics: loading 4000 transitions into a 1000-slot ring → only episode 3
        # survives in the transition buffer.  pref_episodes is unaffected.
        if max_size is not None and self.num_transitions > max_size:
            self.states      = self.states[-max_size:].clone()
            self.actions     = self.actions[-max_size:].clone()
            self.next_states = self.next_states[-max_size:].clone()
            self.rewards     = self.rewards[-max_size:].clone()
            self.dones       = self.dones[-max_size:].clone()
            self.num_transitions = max_size
            print(f"[TeacherBuffer] Ring buffer: truncated transition buffer to last {max_size} transitions "
                  f"(pref pool retains all {len(self.pref_episodes)} expert episodes)")

        # Track initial expert size for logging
        self.initial_size = self.num_transitions

        # _has_promotions: True after first add_episode() call.
        # Used by sail.py to gate LfD mixing (replaces num_transitions == initial_size
        # check which breaks with ring buffer once max_size is filled).
        self._has_promotions = False

        # Student episode pool for QPREF with qpref_source='student'.
        # Built incrementally during training: add_student_episode() is called from the
        # adaptive callback at episode end.  Pruned by recency (TF parity).
        self.pref_student_episodes = []

    @property
    def soft_tac_pool(self):
        return self._soft_tac_expert_episodes + self.soft_tac_student_episodes

    def __len__(self):
        return self.num_transitions

    def size(self):
        return self.num_transitions

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'dones': self.dones[idx]
        }
        
    def sample_batch(self, batch_size: int):
        """
        Helper method to sample a batch without a full DataLoader if preferred in RL loops.
        Returns next_states for LfD mixing (Bellman target requires s').
        """
        indices = torch.randint(0, self.num_transitions, (batch_size,), device=self.device)
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }

    def add_episode(self, episode_obs: np.ndarray, episode_actions: np.ndarray,
                    episode_rewards: np.ndarray = None, episode_dones: np.ndarray = None):
        """
        Add a complete student episode to the teacher buffer (adaptive SAIL/PAIL).

        Matches TF behavior:
        - Appends transitions to demo_replay_buffer (lines 1554 of TF sail.py)
        - Grows teacher buffer dynamically during training
        - Used when student trajectory exceeds expert score threshold

        Args:
            episode_obs: Observations (T, obs_dim)
            episode_actions: Actions (T, act_dim)
            episode_rewards: Rewards (T,) - optional
            episode_dones: Done flags (T,) - optional, will auto-mark last step as done

        Returns:
            None (modifies buffer in-place)
        """
        episode_len = len(episode_obs)
        if episode_len == 0:
            print("[TeacherBuffer] WARNING: Attempted to add empty episode")
            return

        # Ensure numpy arrays
        episode_obs = np.asarray(episode_obs, dtype=np.float32)
        episode_actions = np.asarray(episode_actions, dtype=np.float32)

        if episode_rewards is None:
            episode_rewards = np.zeros(episode_len, dtype=np.float32)
        else:
            episode_rewards = np.asarray(episode_rewards, dtype=np.float32)

        if episode_dones is None:
            # Mark only last step as done (standard episode structure)
            episode_dones = np.zeros(episode_len, dtype=np.float32)
            episode_dones[-1] = 1.0
        else:
            episode_dones = np.asarray(episode_dones, dtype=np.float32)

        # Construct next_obs for this episode:
        #   next_obs[t] = obs[t+1]  for t < T-1  (within-episode transitions)
        #   next_obs[T-1] = obs[T-1]              (terminal self-loop, masked by done=1)
        next_obs_ep = np.empty_like(episode_obs)
        next_obs_ep[:-1] = episode_obs[1:]
        next_obs_ep[-1] = episode_obs[-1]

        # Convert to tensors and concatenate to existing buffers
        # Shape: (episode_len, 1) to match expert data initialization (N, 1)
        new_obs = torch.tensor(episode_obs, dtype=torch.float32, device=self.device)
        new_next_obs = torch.tensor(next_obs_ep, dtype=torch.float32, device=self.device)
        new_actions = torch.tensor(episode_actions, dtype=torch.float32, device=self.device)
        new_rewards = torch.tensor(episode_rewards, dtype=torch.float32, device=self.device).reshape(-1, 1)
        new_dones = torch.tensor(episode_dones, dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.states      = torch.cat([self.states,      new_obs],      dim=0)
        self.next_states = torch.cat([self.next_states, new_next_obs], dim=0)
        self.actions     = torch.cat([self.actions,     new_actions],  dim=0)
        self.rewards     = torch.cat([self.rewards,     new_rewards],  dim=0)
        self.dones       = torch.cat([self.dones,       new_dones],    dim=0)

        # Ring buffer: keep only last max_size transitions (FIFO overwrite).
        # Matches TF ReplayBufferExtend._next_idx ring semantics.
        if self.max_size is not None and len(self.states) > self.max_size:
            self.states      = self.states[-self.max_size:]
            self.next_states = self.next_states[-self.max_size:]
            self.actions     = self.actions[-self.max_size:]
            self.rewards     = self.rewards[-self.max_size:]
            self.dones       = self.dones[-self.max_size:]

        self.num_transitions = len(self.states)
        self._has_promotions = True

        # If preference RM is enabled, also add to pref_episodes
        if self.pref_rm is not None:
            try:
                r_pref = self.pref_rm.reward(episode_obs, episode_actions)
                J = float(np.sum(r_pref))

                self.pref_episodes.append({
                    'obs': new_obs,
                    'acs': new_actions,
                    'J': J
                })

                # Quantile-based pruning: keep J >= quantile(scores, q) when pool exceeds limit.
                # TF parity: pref_max_teacher_trajs=500, pref_promote_quantile=0.75.
                # Always keep at least the single best episode.
                if (self.pref_max_teacher_trajs is not None
                        and len(self.pref_episodes) > self.pref_max_teacher_trajs):
                    scores_arr = np.array([ep['J'] for ep in self.pref_episodes], dtype=np.float64)
                    thresh = np.quantile(scores_arr, self.pref_promote_quantile)
                    keep = [i for i, ep in enumerate(self.pref_episodes) if ep['J'] >= thresh]
                    if len(keep) == 0:
                        keep = [int(np.argmax(scores_arr))]
                    self.pref_episodes = [self.pref_episodes[i] for i in keep]

                # Recompute Boltzmann weights over the full updated pool.
                self._recompute_pref_weights()

            except Exception as e:
                print(f"[TeacherBuffer] WARNING: Failed to compute pref RM score for added episode: {e}")

    def get_growth_stats(self):
        """Return statistics about buffer growth for logging."""
        return {
            'initial_size': self.initial_size,
            'current_size': self.num_transitions,
            'added_transitions': self.num_transitions - self.initial_size,
            'growth_ratio': self.num_transitions / self.initial_size if self.initial_size > 0 else 1.0
        }

    def _build_pref_episodes(self):
        """
        Build the preference episode pool from the current self.states/actions/dones.
        Called BEFORE ring truncation so all N_expert episodes are included.
        Each entry: {'obs': tensor(T,obs_dim), 'acs': tensor(T,act_dim), 'J': float}
        """
        dones_np = self.dones.cpu().numpy().ravel()
        done_idx = np.where(dones_np == 1)[0]
        if len(done_idx) == 0:
            print("[TeacherBuffer] No episode boundaries found for preference ranking.")
            return

        start = 0
        for ep_i, last in enumerate(done_idx):
            obs_ep = self.states[start:last+1]
            acs_ep = self.actions[start:last+1]

            r_pref = self.pref_rm.reward(obs_ep.cpu().numpy(), acs_ep.cpu().numpy())
            J = float(np.sum(r_pref))

            self.pref_episodes.append({
                'obs': obs_ep,
                'acs': acs_ep,
                'J': J
            })
            start = last + 1

        scores = [ep['J'] for ep in self.pref_episodes]
        print(f"[TeacherBuffer] Pref pool: {len(self.pref_episodes)} expert episodes "
              f"(built from full dataset before ring truncation)")
        if scores:
            print(f"[TeacherBuffer] Pref pool RM scores: "
                  f"mean={np.mean(scores):.1f} min={np.min(scores):.1f} max={np.max(scores):.1f} "
                  f"spread={np.max(scores)-np.min(scores):.1f}")

        # Mirror expert episodes into the Soft-TAC dedicated pool (permanent).
        self._soft_tac_expert_episodes = list(self.pref_episodes)

        # Compute initial Boltzmann weights over the expert episodes.
        self._recompute_pref_weights()

    def _recompute_pref_weights(self, beta: float = None):
        """
        Compute Boltzmann softmax weights over pref_episodes using stored J scores.

        TF parity (sail.py:_recompute_pref_teacher_weights):
            s = scores / beta
            s = s - s.max()          # numerically stable max-subtraction
            w = exp(s)
            w = w / (w.sum() + 1e-8) # normalise to sum≈1

        Uses self._pref_reweight_beta if beta is not passed explicitly.
        Stores result in self.pref_teacher_weights (np.array[N_eps]).
        """
        if not self.pref_episodes:
            self.pref_teacher_weights = None
            return
        if beta is None:
            beta = self._pref_reweight_beta
        scores = np.array([ep['J'] for ep in self.pref_episodes], dtype=np.float64)
        s = scores / max(float(beta), 1e-8)
        s = s - s.max()
        w = np.exp(s)
        w = w / (w.sum() + 1e-8)
        self.pref_teacher_weights = w

    def sample_batch_weighted(self, batch_size: int):
        """
        Sample a batch for the discriminator expert path using Boltzmann-weighted episodes.

        TF parity (_sample_pref_weighted_expert, sail.py:780):
        - Episodes are sampled UNIFORMLY by index (NOT proportionally to weight).
        - The softmax weight of the sampled episode is attached as a per-sample
          loss multiplier (expert_w), returned alongside (states, actions).
        - Fallback to uniform sample_batch() + ones weights when pref pool unavailable.

        Returns:
            states:   Tensor (B, obs_dim)
            actions:  Tensor (B, act_dim)
            expert_w: Tensor (B, 1) — softmax weight for each sample, for weighted BCE loss
        """
        if (self.pref_teacher_weights is None or len(self.pref_episodes) == 0):
            # Fallback: uniform sampling + ones weights (no reweighting effect)
            batch = self.sample_batch(batch_size)
            expert_w = torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)
            return batch['states'], batch['actions'], expert_w

        N = len(self.pref_episodes)
        obs_list, acs_list, w_list = [], [], []
        for _ in range(batch_size):
            ep_idx = np.random.randint(0, N)       # uniform episode — TF parity line 804
            ep = self.pref_episodes[ep_idx]
            T = ep['obs'].shape[0]
            t = np.random.randint(0, T)            # uniform timestep within episode
            obs_list.append(ep['obs'][t].float())
            acs_list.append(ep['acs'][t].float())
            w_list.append(float(self.pref_teacher_weights[ep_idx]))

        states   = torch.stack(obs_list).to(self.device)
        actions  = torch.stack(acs_list).to(self.device)
        expert_w = torch.tensor(w_list, dtype=torch.float32, device=self.device).reshape(-1, 1)
        return states, actions, expert_w

    def sample_pref_pairs(self, batch_size: int, return_J: bool = False):
        """
        Sample `batch_size` preference pairs from pref_episodes.

        Args:
            batch_size: Number of pairs B.
            return_J:   When False (default), returns the standard 6-tuple:
                            (pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask)
                        When True, returns an 8-tuple appending RM J scores:
                            (..., pos_J, neg_J)  — Tensor [B] on self.device.
                        Used by Soft-TAC to build discrete preference labels y.
        """
        import random
        N = len(self.pref_episodes)
        if N < 2:
            raise ValueError("Not enough expert episodes for preference ranking")

        batch_pos_obs, batch_pos_acs = [], []
        batch_neg_obs, batch_neg_acs = [], []
        pos_J_list, neg_J_list = [], []

        for _ in range(batch_size):
            a, b = random.sample(range(N), 2)
            # Try to find a pair the RM strictly distinguishes (avoids exact-tie pairs).
            # Soft-TAC handles ties via y=0 (zero gradient), so this is an optimisation
            # only — remaining ties after max_tries are fine.
            tries = 0
            while self.pref_episodes[a]['J'] == self.pref_episodes[b]['J'] and tries < 50:
                a, b = random.sample(range(N), 2)
                tries += 1

            if self.pref_episodes[a]['J'] > self.pref_episodes[b]['J']:
                pos_ep, neg_ep = self.pref_episodes[a], self.pref_episodes[b]
            else:
                pos_ep, neg_ep = self.pref_episodes[b], self.pref_episodes[a]

            batch_pos_obs.append(pos_ep['obs'])
            batch_pos_acs.append(pos_ep['acs'])
            batch_neg_obs.append(neg_ep['obs'])
            batch_neg_acs.append(neg_ep['acs'])
            pos_J_list.append(float(pos_ep['J']))
            neg_J_list.append(float(neg_ep['J']))

        # Pad sequence to max length in this batch
        max_len = max([len(o) for o in batch_pos_obs + batch_neg_obs])

        def pad_and_mask(tensors):
            padded = torch.zeros((batch_size, max_len, tensors[0].shape[1]), device=self.device)
            mask = torch.zeros((batch_size, max_len), device=self.device)
            for i, t in enumerate(tensors):
                length = t.shape[0]
                padded[i, :length] = t
                mask[i, :length] = 1.0
            return padded, mask

        pos_obs, pos_mask = pad_and_mask(batch_pos_obs)
        pos_acs, _ = pad_and_mask(batch_pos_acs)
        neg_obs, neg_mask = pad_and_mask(batch_neg_obs)
        neg_acs, _ = pad_and_mask(batch_neg_acs)

        if return_J:
            pos_J = torch.tensor(pos_J_list, dtype=torch.float32, device=self.device)
            neg_J = torch.tensor(neg_J_list, dtype=torch.float32, device=self.device)
            return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J

        return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask

    # ------------------------------------------------------------------
    # QPREF support: student pool + trajectory-mean-Q pair sampling
    # ------------------------------------------------------------------

    def add_student_episode(self, obs_ep: np.ndarray, acs_ep: np.ndarray) -> None:
        """
        Score a completed student episode with the pref RM and append it to the
        student preference pool (pref_student_episodes).

        Called by SAILAdaptiveCallback at episode end when qpref_source='student'.
        Pruning: recency-based — keep last pref_max_student_trajs entries (TF parity:
        _add_student_episode_to_pref_buffer line 776).

        No-op if pref_rm is not loaded.
        """
        if self.pref_rm is None:
            return
        try:
            r_pref = self.pref_rm.reward(obs_ep, acs_ep)
            J = float(np.sum(np.asarray(r_pref).reshape(-1)))
        except Exception as e:
            print(f"[TeacherBuffer] WARNING: pref RM scoring failed for student episode: {e}")
            return

        self.pref_student_episodes.append({
            'obs': torch.tensor(obs_ep, dtype=torch.float32, device=self.device),
            'acs': torch.tensor(acs_ep, dtype=torch.float32, device=self.device),
            'J':   J,
        })
        # Recency pruning (TF parity: keep last N)
        if len(self.pref_student_episodes) > self.pref_max_student_trajs:
            self.pref_student_episodes = \
                self.pref_student_episodes[-self.pref_max_student_trajs:]

    def sample_qpref_pairs_aggregate(self, batch_size: int, source: str = 'teacher'):
        """
        Sample `batch_size` preference pairs for trajectory-mean-Q QPREF.

        Each pair contains FULL padded trajectories so the caller can evaluate Q at
        every valid timestep and average.  Pair selection and preference labels follow
        the same J-based logic used by pref_rank_disc and pref_reweight_teacher.

        Args:
            batch_size: Number of pairs B.
            source:     'teacher' -> pref_episodes (expert + promoted);
                        'student' -> pref_student_episodes.

        Returns tuple of 6 tensors on self.device:
            pos_obs:  [B, T_max, obs_dim]
            pos_acs:  [B, T_max, act_dim]
            pos_mask: [B, T_max]   — 1.0 for valid steps, 0.0 for padding
            neg_obs:  [B, T_max, obs_dim]
            neg_acs:  [B, T_max, act_dim]
            neg_mask: [B, T_max]
        Returns None when the chosen pool has fewer than 2 episodes.
        """
        eps = self.pref_episodes if source == 'teacher' else self.pref_student_episodes
        if (not eps) or (len(eps) < 2):
            return None

        N = len(eps)
        idx_a = np.random.randint(0, N, size=batch_size)
        idx_b = np.random.randint(0, N, size=batch_size)
        for k in range(batch_size):
            if idx_b[k] == idx_a[k]:
                idx_b[k] = (idx_b[k] + 1) % N  # deterministic tie-break (TF parity)

        pos_eps_sel, neg_eps_sel = [], []
        for a, b in zip(idx_a, idx_b):
            Ja = float(eps[a]['J'])
            Jb = float(eps[b]['J'])
            epP, epN = (eps[a], eps[b]) if Ja >= Jb else (eps[b], eps[a])
            pos_eps_sel.append(epP)
            neg_eps_sel.append(epN)

        # Pad to T_max of this batch
        def ep_len(ep):
            return int(ep['acs'].shape[0])

        T_max   = max(max(ep_len(e) for e in pos_eps_sel),
                      max(ep_len(e) for e in neg_eps_sel))
        obs_dim = pos_eps_sel[0]['obs'].shape[-1]
        act_dim = pos_eps_sel[0]['acs'].shape[-1]
        B       = batch_size

        pos_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32, device=self.device)
        pos_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32, device=self.device)
        pos_mask = torch.zeros(B, T_max,           dtype=torch.float32, device=self.device)
        neg_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32, device=self.device)
        neg_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32, device=self.device)
        neg_mask = torch.zeros(B, T_max,           dtype=torch.float32, device=self.device)

        for i in range(B):
            epP, epN = pos_eps_sel[i], neg_eps_sel[i]
            LP, LN   = ep_len(epP), ep_len(epN)
            pos_obs[i, :LP]  = epP['obs'][:LP].float()
            pos_acs[i, :LP]  = epP['acs'][:LP].float()
            pos_mask[i, :LP] = 1.0
            neg_obs[i, :LN]  = epN['obs'][:LN].float()
            neg_acs[i, :LN]  = epN['acs'][:LN].float()
            neg_mask[i, :LN] = 1.0

        return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask

    def sample_qpref_pairs_cross_pool(self, batch_size: int,
                                      min_student: int = 0):
        """
        Sample `batch_size` preference pairs for cross-pool QPREF.

        Each pair is one episode from pref_episodes (teacher/expert) and one
        episode from pref_student_episodes (student rollouts).  Pos/neg
        assignment is by J comparison — either side can be positive; early in
        training the teacher is almost always positive, but if a student
        episode surpasses expert quality the assignment flips automatically.

        Args:
            batch_size:  Number of pairs B.
            min_student: If > 0 and len(pref_student_episodes) >= min_student,
                         fall back to student-student sampling instead.

        Returns same 6-tuple as sample_qpref_pairs_aggregate, or None when
        either pool is empty.

        Side-effect: sets self._last_qpref_j_spread_mean (float) for logging.
        """
        # Optional warmup gate: fall back to student-student once pool is large
        if min_student > 0 and len(self.pref_student_episodes) >= min_student:
            self._last_qpref_j_spread_mean = None  # not cross-pool this step
            return self.sample_qpref_pairs_aggregate(batch_size, source='student')

        if len(self.pref_episodes) == 0 or len(self.pref_student_episodes) == 0:
            self._last_qpref_j_spread_mean = None
            return None

        N_t = len(self.pref_episodes)
        N_s = len(self.pref_student_episodes)
        t_idx = np.random.randint(0, N_t, size=batch_size)
        s_idx = np.random.randint(0, N_s, size=batch_size)

        pos_eps_sel, neg_eps_sel = [], []
        j_spreads = []
        for ti, si in zip(t_idx, s_idx):
            ep_t = self.pref_episodes[ti]
            ep_s = self.pref_student_episodes[si]
            Jt, Js = float(ep_t['J']), float(ep_s['J'])
            j_spreads.append(abs(Jt - Js))
            if Jt >= Js:
                pos_eps_sel.append(ep_t)
                neg_eps_sel.append(ep_s)
            else:
                pos_eps_sel.append(ep_s)
                neg_eps_sel.append(ep_t)

        self._last_qpref_j_spread_mean = float(np.mean(j_spreads))

        def ep_len(ep):
            return int(ep['acs'].shape[0])

        T_max   = max(max(ep_len(e) for e in pos_eps_sel),
                      max(ep_len(e) for e in neg_eps_sel))
        obs_dim = pos_eps_sel[0]['obs'].shape[-1]
        act_dim = pos_eps_sel[0]['acs'].shape[-1]
        B       = batch_size

        pos_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32, device=self.device)
        pos_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32, device=self.device)
        pos_mask = torch.zeros(B, T_max,           dtype=torch.float32, device=self.device)
        neg_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32, device=self.device)
        neg_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32, device=self.device)
        neg_mask = torch.zeros(B, T_max,           dtype=torch.float32, device=self.device)

        for i in range(B):
            epP, epN = pos_eps_sel[i], neg_eps_sel[i]
            LP, LN   = ep_len(epP), ep_len(epN)
            pos_obs[i, :LP]  = epP['obs'][:LP].float()
            pos_acs[i, :LP]  = epP['acs'][:LP].float()
            pos_mask[i, :LP] = 1.0
            neg_obs[i, :LN]  = epN['obs'][:LN].float()
            neg_acs[i, :LN]  = epN['acs'][:LN].float()
            neg_mask[i, :LN] = 1.0

        return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask

    # ------------------------------------------------------------------
    # Soft-TAC support: dedicated pool with all student episodes
    # ------------------------------------------------------------------

    def add_soft_tac_student_episode(self, obs_ep: np.ndarray, acs_ep: np.ndarray,
                                     J: float = None) -> None:
        """
        Score a completed student episode and add it to the Soft-TAC student pool.
        Called every episode (unlike pref_episodes which only grows on promotion).
        Recency-pruned to soft_tac_max_student_trajs.

        J: optional pre-computed quality score. When provided, skips pref_rm scoring
           (used by online path where GT return is the available signal).
           When None, scores with pref_rm. No-op if both J is None and pref_rm is absent.
        """
        if J is None:
            if self.pref_rm is None:
                return
            try:
                r_pref = self.pref_rm.reward(obs_ep, acs_ep)
                J = float(np.sum(np.asarray(r_pref).reshape(-1)))
            except Exception as e:
                print(f"[TeacherBuffer] WARNING: pref RM scoring failed for soft-tac student episode: {e}")
                return

        self.soft_tac_student_episodes.append({
            'obs': torch.tensor(obs_ep, dtype=torch.float32, device=self.device),
            'acs': torch.tensor(acs_ep, dtype=torch.float32, device=self.device),
            'J':   J,
        })
        if len(self.soft_tac_student_episodes) > self.soft_tac_max_student_trajs:
            self.soft_tac_student_episodes = \
                self.soft_tac_student_episodes[-self.soft_tac_max_student_trajs:]

    def sample_soft_tac_pairs(self, batch_size: int):
        """
        Sample batch_size pairs from soft_tac_pool (expert + all students).
        Always returns an 8-tuple including J scores for label construction.

        Returns:
            (pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J)
        Raises ValueError when pool has fewer than 2 episodes.
        """
        import random
        pool = self.soft_tac_pool
        N = len(pool)
        if N < 2:
            raise ValueError("Not enough episodes in soft_tac_pool for Soft-TAC")

        batch_pos_obs, batch_pos_acs = [], []
        batch_neg_obs, batch_neg_acs = [], []
        pos_J_list, neg_J_list = [], []

        for _ in range(batch_size):
            a, b = random.sample(range(N), 2)
            tries = 0
            while pool[a]['J'] == pool[b]['J'] and tries < 50:
                a, b = random.sample(range(N), 2)
                tries += 1

            if pool[a]['J'] > pool[b]['J']:
                pos_ep, neg_ep = pool[a], pool[b]
            else:
                pos_ep, neg_ep = pool[b], pool[a]

            batch_pos_obs.append(pos_ep['obs'])
            batch_pos_acs.append(pos_ep['acs'])
            batch_neg_obs.append(neg_ep['obs'])
            batch_neg_acs.append(neg_ep['acs'])
            pos_J_list.append(float(pos_ep['J']))
            neg_J_list.append(float(neg_ep['J']))

        max_len = max(t.shape[0] for t in batch_pos_obs + batch_neg_obs)

        def pad_and_mask(tensors):
            padded = torch.zeros((batch_size, max_len, tensors[0].shape[1]), device=self.device)
            mask = torch.zeros((batch_size, max_len), device=self.device)
            for i, t in enumerate(tensors):
                length = t.shape[0]
                padded[i, :length] = t
                mask[i, :length] = 1.0
            return padded, mask

        pos_obs, pos_mask = pad_and_mask(batch_pos_obs)
        pos_acs, _        = pad_and_mask(batch_pos_acs)
        neg_obs, neg_mask = pad_and_mask(batch_neg_obs)
        neg_acs, _        = pad_and_mask(batch_neg_acs)

        pos_J = torch.tensor(pos_J_list, dtype=torch.float32, device=self.device)
        neg_J = torch.tensor(neg_J_list, dtype=torch.float32, device=self.device)
        return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J


def create_teacher_dataloader(data_path: str, device: torch.device, batch_size: int = 256, shuffle: bool = True):
    """Optional helper to create a standard PyTorch DataLoader."""
    dataset = TeacherBuffer(data_path, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    import glob
    import os
    import sys
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)
    
    test_files = glob.glob(os.path.join(project_root, "**/*.npz"), recursive=True)
    
    if test_files:
        test_file = test_files[0]
        device = torch.device("cpu")
        print(f"--- Testing TeacherBuffer on {test_file} ---")
        buffer = TeacherBuffer(test_file, device)
        print(f"Loaded {len(buffer)} transitions onto {device}")
        
        # Test __getitem__
        sample = buffer[0]
        print(f"Single item shapes -> State: {sample['states'].shape}, Action: {sample['actions'].shape}")
        
        # Test batching
        batch = buffer.sample_batch(64)
        print(f"Sampled batch shapes -> State: {batch['states'].shape}, Action: {batch['actions'].shape}")
    else:
        print("No .npz files found in the project to test.")
