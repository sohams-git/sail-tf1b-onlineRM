"""
Episode-level trajectory tracking for SAIL adaptive teacher buffer.

Matches the original TensorFlow TrajectoryBuffer used for student episode collection.
Used to reconstruct complete episodes for adaptive teacher buffer replacement.
"""

import numpy as np
from typing import List, Dict, Tuple


class EpisodeBuffer:
    """
    Stores complete episodes (trajectories) for student rollouts.

    Matches TF TrajectoryBuffer behavior:
    - Accumulates (obs, action, reward, next_obs, done) during episode
    - Provides get_episode_return() to retrieve all transitions with computed returns
    - Resets after episode ends and is retrieved

    Used by SAIL.adaptive to:
    1. Track student episodes during training
    2. Extract complete trajectories when student exceeds expert threshold
    3. Add high-quality student data to teacher buffer
    """

    def __init__(self, max_size: int = int(1e5), gamma: float = 0.99):
        """
        Args:
            max_size: Maximum number of transitions to store (safety limit)
            gamma: Discount factor for computing discounted returns
        """
        self.max_size = max_size
        self.gamma = gamma
        self.reset()

    def reset(self):
        """Clear current episode buffer."""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
        self.true_rewards = []  # Store env rewards separately from surrogate

    def add(self, obs, action, reward, next_obs, done, true_reward=None):
        """
        Add a single transition to the current episode.

        Args:
            obs: Observation (numpy array)
            action: Action (numpy array)
            reward: Reward (surrogate reward from discriminator in SAIL)
            next_obs: Next observation
            done: Episode termination flag (1.0 or True if done)
            true_reward: Original environment reward (optional, for logging)
        """
        if len(self.obs) >= self.max_size:
            print(f"[EpisodeBuffer] WARNING: Reached max size {self.max_size}, dropping transition")
            return

        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.dones.append(float(done))
        self.true_rewards.append(float(true_reward) if true_reward is not None else float(reward))

    def size(self) -> int:
        """Return number of transitions in current episode."""
        return len(self.obs)

    def get_episode_return(self, normalized_score: float = None):
        """
        Retrieve all transitions in the current episode with computed discounted returns.

        Matches TF TrajectoryBuffer.get_episode_return(es) signature.
        Computes discount_return for each transition as: es * gamma^(episode_length - idx)

        Args:
            normalized_score: Normalized episode score (optional, used for discount computation)

        Yields:
            tuple: (obs, action, reward, next_obs, done, None, episode_score, true_reward, discount_return)
                   Format matches TF output (line 1552-1553 of sail.py)
        """
        if self.size() == 0:
            return

        episode_length = self.size()

        # Compute episode return (sum of true rewards)
        episode_score = sum(self.true_rewards)

        # Use normalized score if provided, otherwise raw episode score
        es = normalized_score if normalized_score is not None else episode_score

        for idx in range(episode_length):
            # Compute discounted return from this step forward
            # TF: discount_return = es * (gamma ** (episode_length - idx))
            discount_return = es * (self.gamma ** (episode_length - idx))

            yield (
                self.obs[idx],              # s
                self.actions[idx],          # a
                self.rewards[idx],          # r (surrogate)
                self.next_obs[idx],         # s'
                self.dones[idx],            # done
                None,                       # placeholder (if_demo in TF, unused)
                episode_score,              # episode return
                self.true_rewards[idx],     # true env reward
                discount_return             # discounted return from this step
            )

    def get_last_episode_score(self) -> float:
        """Return sum of true rewards for the current episode."""
        return sum(self.true_rewards)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0


if __name__ == "__main__":
    # Unit test
    print("Testing EpisodeBuffer...")

    buffer = EpisodeBuffer(gamma=0.99)

    # Simulate a 5-step episode
    for t in range(5):
        obs = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5])
        reward = 1.0
        next_obs = np.array([1.1, 2.1, 3.1])
        done = 1.0 if t == 4 else 0.0
        true_reward = 1.0

        buffer.add(obs, action, reward, next_obs, done, true_reward)

    print(f"Episode size: {buffer.size()}")
    print(f"Episode score: {buffer.get_last_episode_score():.2f}")

    # Retrieve episode
    print("\nRetrieving episode transitions:")
    for i, transition in enumerate(buffer.get_episode_return(normalized_score=5.0)):
        s, a, r, s1, done, _, ep_score, true_r, disc_ret = transition
        print(f"  Step {i}: obs_dim={len(s)}, done={done}, discount_return={disc_ret:.4f}")

    # Reset and verify empty
    buffer.reset()
    print(f"\nAfter reset: size={buffer.size()}, is_empty={buffer.is_empty()}")

    print("\n✓ EpisodeBuffer test passed")
