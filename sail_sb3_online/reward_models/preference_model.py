import torch
import torch.nn as nn
import torch.nn.functional as F

class PreferenceRewardModel(nn.Module):
    """
    Learns a scalar reward function from human (or synthetic) trajectory preferences.
    Scores state-action pairs or sequences of state-action pairs.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256, 256), dropout_prob=0.0):
        super(PreferenceRewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        input_dim = state_dim + action_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
            input_dim = h
            
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns the predicted reward for state-action pairs.
        Can handle batch: (batch_size, state_dim) -> (batch_size, 1)
        Can handle sequences: (batch_size, seq_len, state_dim) -> (batch_size, seq_len, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
        
    def score_segment(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Scores a sequential trajectory segment by summing rewards across the sequence.
        Input shapes expected: (batch_size, seq_len, dim)
        Output shape: (batch_size, 1)
        """
        if state.dim() != 3 or action.dim() != 3:
            raise ValueError("Expected 3D tensors for segments: (batch_size, seq_len, dim)")
            
        step_rewards = self.forward(state, action) # (batch_size, seq_len, 1)
        segment_returns = torch.sum(step_rewards, dim=1) # (batch_size, 1)
        return segment_returns

    def compute_preference_loss(self, seg1_state: torch.Tensor, seg1_action: torch.Tensor,
                                      seg2_state: torch.Tensor, seg2_action: torch.Tensor, 
                                      preference_label: torch.Tensor) -> torch.Tensor:
        """
        Compute Bradley-Terry preference loss based on trajectory returns.
        
        preference_label shape: (batch_size,), containing soft or hard probabilities
        e.g., 1.0 means seg1 preferred, 0.0 means seg2 preferred, 0.5 means tie.
        
        Both seg1 and seg2 tensors should have shape: (batch_size, seq_len, dim)
        """
        ret1 = self.score_segment(seg1_state, seg1_action).squeeze(-1) # (batch_size,)
        ret2 = self.score_segment(seg2_state, seg2_action).squeeze(-1) # (batch_size,)
        
        # P(seg1 > seg2) = sigmoid(ret1 - ret2)
        # Using BCEWithLogitsLoss where logits = ret1 - ret2.
        # BCE with logits computes: -[label * log(sigmoid(x)) + (1-label) * log(1-sigmoid(x))]
        logits = ret1 - ret2
        return F.binary_cross_entropy_with_logits(logits, preference_label)

if __name__ == "__main__":
    print("--- Testing Preference Reward Model ---")
    state_dim, action_dim = 18, 6
    batch_size, seq_len = 4, 50
    
    model = PreferenceRewardModel(state_dim, action_dim)
    
    # Dummy tensors for a sequence segment
    seg1_s = torch.randn((batch_size, seq_len, state_dim))
    seg1_a = torch.randn((batch_size, seq_len, action_dim))
    seg2_s = torch.randn((batch_size, seq_len, state_dim))
    seg2_a = torch.randn((batch_size, seq_len, action_dim))
    
    # 1 prefers seg1, 0 prefers seg2, 0.5 is tie
    pref_labels = torch.tensor([1.0, 0.0, 0.5, 1.0])
    
    # 1. Forward step rewards
    step_rewards = model(seg1_s, seg1_a)
    print(f"Step rewards shape: {step_rewards.shape} (Expected: {batch_size}, {seq_len}, 1)")
    
    # 2. Score segment
    seg_score = model.score_segment(seg1_s, seg1_a)
    print(f"Segment score shape: {seg_score.shape} (Expected: {batch_size}, 1)")
    
    # 3. Compute loss
    loss = model.compute_preference_loss(seg1_s, seg1_a, seg2_s, seg2_a, pref_labels)
    print(f"Preference Bradley-Terry Loss: {loss.item():.4f}")
