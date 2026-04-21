import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SAILAdaptiveCallback(BaseCallback):
    """
    SB3 callback for Adaptive SAIL.
    Tracks full student episodes and promotes them to the teacher buffer if they
    exceed the current expert score threshold.

    Supports two score sources for the promotion decision:
    - 'gt': ground-truth environment return (default, requires Monitor wrapper)
    - 'rm': preference reward model cumulative score (no GT leakage; requires pref_rm)

    Also supports QPREF student-source: when qpref_source='student', every completed
    episode is scored with the pref RM and added to teacher_buffer.pref_student_episodes.

    Ensures Vanilla/Adaptive rollout parity by moving tracking out of the main logic path.
    """
    def __init__(self, teacher_buffer, expert_scores_list, gamma=0.99, debug=False, verbose=0,
                 score_source="gt", pref_rm=None, rm_expert_scores=None,
                 qpref_source: str = "teacher", soft_tac: bool = False,
                 online_rm_manager=None):
        super().__init__(verbose)
        self.teacher_buffer = teacher_buffer
        self.expert_scores = expert_scores_list
        self.gamma = gamma
        self.debug = debug
        self.score_source = score_source
        self.pref_rm = pref_rm

        # RM-based promotion: separate threshold list (insertion order, no GT leakage).
        # rm_expert_scores[0] = RM score of first-loaded expert episode = threshold.
        if score_source == "rm":
            if pref_rm is None:
                raise ValueError("SAILAdaptiveCallback: score_source='rm' requires pref_rm")
            if rm_expert_scores is None:
                raise ValueError("SAILAdaptiveCallback: score_source='rm' requires rm_expert_scores")
            self.rm_scores = list(rm_expert_scores)  # Mutable copy; FIFO sliding window
        else:
            self.rm_scores = None

        # QPREF student-source: collect every student episode into pref_student_episodes.
        # active when qpref_source='student' AND teacher_buffer.pref_rm is loaded.
        self.qpref_source = qpref_source
        # Soft-TAC student pool: feed every episode to the dedicated soft_tac_pool.
        self.soft_tac = soft_tac

        # Online RM manager: receives student episodes for segment buffer population.
        self.online_rm_manager = online_rm_manager

        # Lazy import to avoid circular dependencies
        from sail_sb3.datasets.episode_buffer import EpisodeBuffer
        self.episode_buffer = EpisodeBuffer(max_size=int(1e5), gamma=self.gamma)

        # Track observation state manually to avoid drift/off-by-one errors
        self._current_obs = None

        # Event counters
        self._promotions_total = 0
        self._first_promotion_step = -1
        self._episodes_completed = 0

    def _on_rollout_start(self) -> None:
        """Called at the beginning of every rollout collection block."""
        # Initial model._last_obs is captured once
        if self.model is not None:
            self._current_obs = self.model._last_obs.copy()

    def _on_step(self) -> bool:
        """
        Called by SB3's collect_rollouts after each environment step.
        Obtains transition data safely from self.locals.
        """
        # Retrieve transition data from locals()
        # In SB3 OffPolicyAlgorithm, these are explicitly named in the local scope:
        # actions  = Sampled from policy
        # new_obs  = Observation resulting from step
        # rewards  = Reward from step
        # dones    = Termination flag
        # infos    = Environment info dicts
        actions = self.locals.get('actions')
        new_obs = self.locals.get('new_obs')
        rewards = self.locals.get('rewards')
        dones   = self.locals.get('dones')
        infos   = self.locals.get('infos')

        if any(x is None for x in [actions, new_obs, rewards, dones, infos]):
            return True # Defensive check for early stages or unusual wrappers

        # Extract data for episode tracking (handling vectorized envs)
        n_envs = len(dones)
        for idx in range(n_envs):
            # Safe indexing
            obs_t   = self._current_obs[idx] if len(self._current_obs.shape) > 1 else self._current_obs
            act_t   = actions[idx] if len(actions.shape) > 1 else actions
            rew_t   = rewards[idx] if isinstance(rewards, np.ndarray) else rewards
            next_t  = new_obs[idx] if len(new_obs.shape) > 1 else new_obs
            done_t  = dones[idx] if isinstance(dones, np.ndarray) else dones
            info_t  = infos[idx] if isinstance(infos, list) else infos

            # Add to internal tracking buffer
            self.episode_buffer.add(
                obs=obs_t,
                action=act_t,
                reward=rew_t,
                next_obs=next_t,
                done=done_t,
                true_reward=rew_t
            )

            # Check if an episode just ended in this env
            if done_t and 'episode' in info_t:
                self._episodes_completed += 1
                gt_score = float(info_t['episode']['r'])
                episode_length = int(info_t['episode']['l'])

                # Collect trajectory for potential promotion
                traj_list = list(self.episode_buffer.get_episode_return())
                obs_ep = np.array([t[0] for t in traj_list], dtype=np.float32) if traj_list else None
                acs_ep = np.array([t[1] for t in traj_list], dtype=np.float32) if traj_list else None

                # Compute promotion score based on score_source
                if self.score_source == "rm" and self.pref_rm is not None and obs_ep is not None:
                    try:
                        r_rm = self.pref_rm.reward(obs_ep, acs_ep)
                        student_score = float(np.sum(r_rm))
                    except Exception as e:
                        if self.debug:
                            print(f"[SAIL-Adaptive] RM scoring failed: {e}. Falling back to gt.")
                        student_score = gt_score
                    threshold_list = self.rm_scores
                    score_tag = "rm"
                else:
                    student_score = gt_score
                    threshold_list = self.expert_scores
                    score_tag = "gt"

                # Adaptive threshold check (TF line 1542)
                if len(threshold_list) > 0 and student_score > threshold_list[0]:
                    if self.verbose > 0 or self.debug:
                        print(f"[SAIL-Adaptive] Student {score_tag}_score={student_score:.1f} > "
                              f"threshold={threshold_list[0]:.1f}. Promoting.")

                    if obs_ep is not None and len(obs_ep) > 0:
                        self.teacher_buffer.add_episode(
                            episode_obs=obs_ep,
                            episode_actions=acs_ep
                        )

                        self._promotions_total += 1
                        if self._first_promotion_step < 0:
                            self._first_promotion_step = self.num_timesteps

                        # Online RM path: also add to pref pool with the same J used for scoring.
                        # Pre-activation: J = GT return (RM not reliable yet).
                        # Post-activation: J = RM cumulative reward (creates spread between episodes).
                        if (self.online_rm_manager is not None
                                and hasattr(self.teacher_buffer, 'add_pref_episode')):
                            try:
                                if self.online_rm_manager.is_active:
                                    r_rm = self.online_rm_manager.reward(
                                        np.asarray(obs_ep, dtype=np.float32),
                                        np.asarray(acs_ep, dtype=np.float32),
                                    )
                                    J_pref = float(np.sum(r_rm))
                                else:
                                    J_pref = gt_score
                                self.teacher_buffer.add_pref_episode(obs_ep, acs_ep, J_pref)
                                if self.debug:
                                    src = "rm" if self.online_rm_manager.is_active else "gt"
                                    print(f"[SAIL-Adaptive] add_pref_episode: J={J_pref:.1f} "
                                          f"(src={src}) pool={len(self.teacher_buffer.pref_episodes)}")
                            except Exception as e:
                                if self.debug:
                                    print(f"[SAIL-Adaptive] add_pref_episode failed: {e}")

                        # Update threshold list (FIFO sliding window, insertion order — no sort).
                        # threshold_list[0] = oldest entry = current threshold.
                        if len(threshold_list) >= 10:
                            threshold_list.pop(0)
                        threshold_list.append(student_score)

                        # Logging via model logger (standard SB3 practice)
                        stats = self.teacher_buffer.get_growth_stats()
                        self.logger.record("adaptive/teacher_buffer_size", stats['current_size'])
                        self.logger.record("adaptive/expert_threshold", threshold_list[0])
                        self.logger.record("adaptive/promoted_episodes",
                                           stats['added_transitions'] // max(episode_length, 1))
                        # pref_pool_size: how many episodes are available for PrefRank sampling
                        self.logger.record("adaptive/pref_pool_size",
                                           len(self.teacher_buffer.pref_episodes))
                        if score_tag == "rm":
                            self.logger.record("adaptive/student_rm_score", student_score)
                            self.logger.record("adaptive/rm_threshold", threshold_list[0])
                        else:
                            self.logger.record("adaptive/student_gt_score", student_score)
                        self.logger.record("events/promotions_total",        self._promotions_total)
                        self.logger.record("events/first_promotion_step",    self._first_promotion_step)
                        self.logger.record("events/episodes_completed",      self._episodes_completed)
                        self.logger.record("adaptive/has_promotions",        int(self.teacher_buffer._has_promotions))
                        # Expert score stats
                        tl = threshold_list  # the active threshold list
                        if tl:
                            self.logger.record("adaptive/expert_scores_min",  float(min(tl)))
                            self.logger.record("adaptive/expert_scores_max",  float(max(tl)))
                            self.logger.record("adaptive/expert_scores_mean", float(sum(tl)/len(tl)))
                        # Teacher buffer ring stats
                        if self.teacher_buffer.max_size is not None:
                            cap = self.teacher_buffer.max_size
                            cur = stats['current_size']
                            self.logger.record("adaptive/teacher_buffer_capacity",   cap)
                            self.logger.record("adaptive/teacher_buffer_fill_ratio", cur / max(cap, 1))

                # Always log episodes_completed (not just on promotion)
                self.logger.record("events/episodes_completed", self._episodes_completed)

                # QPREF student-source: add every completed episode to student pref pool
                # using the online RM J label. No GT fallback: episodes are skipped
                # until the RM is active so the pool is always RM-scored.
                if self.qpref_source == "student" and obs_ep is not None and len(obs_ep) > 0:
                    if (self.online_rm_manager is not None
                            and self.online_rm_manager.is_active):
                        try:
                            r = self.online_rm_manager.reward(obs_ep, acs_ep)
                            rm_j = float(np.sum(r))
                            self.teacher_buffer.add_student_episode(obs_ep, acs_ep, J=rm_j)
                        except Exception as e:
                            pass  # silent skip; pool grows only when RM can score
                    # Before RM activation: skip entirely (no GT fallback)

                # Soft-TAC student pool: add every episode (no quality filter).
                # Pass J=gt_score so the method works even without an offline pref_rm.
                if self.soft_tac and obs_ep is not None and len(obs_ep) > 0:
                    self.teacher_buffer.add_soft_tac_student_episode(obs_ep, acs_ep, J=gt_score)

                # Online RM: feed student episode to segment store
                if (self.online_rm_manager is not None
                        and obs_ep is not None and len(obs_ep) > 0):
                    rew_ep = np.array([t[2] for t in traj_list], dtype=np.float32) \
                             if traj_list else None
                    if rew_ep is not None:
                        self.online_rm_manager.add_student_episode(obs_ep, acs_ep, rew_ep)

                # Reset buffer for the next episode in this env
                self.episode_buffer.reset()

        # Update current observation for the next step across all envs
        self._current_obs = new_obs.copy()
        
        return True

class DiscriminatorLoggingCallback(BaseCallback):
    """
    An SB3 callback to log discriminator metrics (loss, accuracy, custom reward values)
    to TensorBoard or Weights & Biases during the training rollouts.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Placeholder for discriminator-specific logging
        return True
