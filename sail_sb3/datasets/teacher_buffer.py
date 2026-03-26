import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sail_sb3.datasets.expert_loader import load_expert_npz

class TeacherBuffer(Dataset):
    """
    A PyTorch Dataset that loads expert/teacher demonstrations into memory.
    Provides an interface to sample minibatches of (state, action) for the discriminator.
    """
    def __init__(self, data_path: str, device: torch.device, pref_rm_path: str = None, expect_obs_dim: int = 17):
        self.data_path = data_path
        self.device = device
        self.pref_rm_path = pref_rm_path
        
        # Load structured dictionary from the NPZ utility
        parsed_data = load_expert_npz(data_path)
        
        # Convert numpy arrays to PyTorch tensors and move to device
        self.states = torch.tensor(parsed_data['observations'], dtype=torch.float32).to(self.device)
        self.actions = torch.tensor(parsed_data['actions'], dtype=torch.float32).to(self.device)
        self.num_transitions = len(self.states)
        
        if 'dones' in parsed_data:
            self.dones = torch.tensor(parsed_data['dones'], dtype=torch.float32).to(self.device)
        else:
            self.dones = torch.zeros(self.num_transitions, dtype=torch.float32).to(self.device)

        if 'rewards' in parsed_data:
            self.rewards = torch.tensor(parsed_data['rewards'], dtype=torch.float32).to(self.device)
        else:
            self.rewards = torch.zeros(self.num_transitions, dtype=torch.float32).to(self.device)
            
        self.pref_episodes = []
        if pref_rm_path:
            import os
            if os.path.exists(pref_rm_path):
                from sail_sb3.reward_models.pref_rm_eval import PrefRewardModel
                self.pref_rm = PrefRewardModel(pref_rm_path, device=str(self.device), expect_obs_dim=expect_obs_dim)
                self._build_pref_episodes()
            else:
                print(f"[TeacherBuffer] WARNING: RM path {pref_rm_path} does not exist.")
            
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
        """
        indices = torch.randint(0, self.num_transitions, (batch_size,), device=self.device)
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'dones': self.dones[indices]
        }

    def _build_pref_episodes(self):
        dones_np = self.dones.cpu().numpy()
        done_idx = np.where(dones_np == 1)[0]
        if len(done_idx) == 0:
            print("[TeacherBuffer] No episode boundaries found for preference ranking.")
            return
            
        start = 0
        for ep_i, last in enumerate(done_idx):
            obs_ep = self.states[start:last+1]
            acs_ep = self.actions[start:last+1]
            
            # Score episode offline
            r_pref = self.pref_rm.reward(obs_ep.cpu().numpy(), acs_ep.cpu().numpy())
            J = float(np.sum(r_pref))
            
            self.pref_episodes.append({
                'obs': obs_ep,
                'acs': acs_ep,
                'J': J
            })
            start = last + 1
            
        # Debug/startup print
        scores = [ep['J'] for ep in self.pref_episodes]
        print(f"[TeacherBuffer] Built {len(self.pref_episodes)} preference episodes using RM {self.pref_rm_path}")
        if scores:
            print(f"[TeacherBuffer] RM scores: mean={np.mean(scores):.1f} min={np.min(scores):.1f} max={np.max(scores):.1f}")

    def sample_pref_pairs(self, batch_size: int):
        import random
        N = len(self.pref_episodes)
        if N < 2:
            raise ValueError("Not enough expert episodes for preference ranking")
            
        batch_pos_obs, batch_pos_acs = [], []
        batch_neg_obs, batch_neg_acs = [], []
        
        for _ in range(batch_size):
            a, b = random.sample(range(N), 2)
            # Find a pair that the RM strictly distinguishes
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
        
        return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask

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
