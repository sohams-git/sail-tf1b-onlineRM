import numpy as np
import os

def load_expert_npz(filepath: str):
    """
    Utility mechanism to load expert trajectories from .npz files saved by the old codebase.
    Translates format to standard dictionaries or arrays for the TeacherBuffer.
    
    Expected keys in Stable-Baselines npz:
    - 'obs'
    - 'actions'
    - 'rewards'
    - 'episode_returns'
    - 'episode_starts'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expert data file not found: {filepath}")
        
    try:
        data = np.load(filepath, allow_pickle=True)
        keys = data.files
    except Exception as e:
        raise ValueError(f"Failed to load .npz file at {filepath}. Error: {e}")

    # Inspect keys and standardize
    parsed_data = {}
    
    # Map common aliases for observations
    if 'obs' in keys:
        parsed_data['observations'] = data['obs']
    elif 'observations' in keys:
        parsed_data['observations'] = data['observations']
    else:
        raise KeyError(f"Could not find observation key in {keys}. Expected 'obs' or 'observations'.")
        
    # Map actions
    if 'actions' in keys:
        parsed_data['actions'] = data['actions']
    else:
        raise KeyError(f"Could not find 'actions' key in {keys}.")
        
    # Optional fields
    if 'rewards' in keys:
        parsed_data['rewards'] = data['rewards']
    if 'episode_starts' in keys:
        parsed_data['episode_starts'] = data['episode_starts']

    # Ensure shape consistency
    num_transitions = len(parsed_data['observations'])
    if len(parsed_data['actions']) != num_transitions:
        raise ValueError("Mismatched number of observations and actions.")

    # Compute terminal flags (dones) based on episode_starts if available
    if 'episode_starts' in parsed_data:
        # A state 'i' is terminal if 'i+1' is the start of a new episode
        dones = np.zeros(num_transitions, dtype=bool)
        ep_starts = parsed_data['episode_starts']
        for i in range(num_transitions - 1):
            if ep_starts[i + 1]:
                dones[i] = True
        dones[-1] = True # The very last transition is always terminal
        parsed_data['dones'] = dones

        # Construct next_observations for LfD mixing (Bellman target requires s').
        # next_obs[i] = obs[i+1] for non-terminal transitions.
        # At terminal transitions (done=True), next_obs is irrelevant because the
        # Bellman target uses (1 - done) * gamma * Q(s', a'), so we use a self-loop
        # (next_obs = obs) as a safe placeholder.
        obs_np = parsed_data['observations']
        next_obs_np = np.empty_like(obs_np)
        next_obs_np[:-1] = obs_np[1:]
        next_obs_np[-1] = obs_np[-1]            # last step: self-loop
        terminal_idx = np.where(dones)[0]
        for idx in terminal_idx:
            next_obs_np[idx] = obs_np[idx]      # episode boundary: self-loop
        parsed_data['next_observations'] = next_obs_np

    return parsed_data

if __name__ == "__main__":
    # Small self-test
    import glob
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    test_files = glob.glob(os.path.join(project_root, "**/*.npz"), recursive=True)
    if test_files:
        test_file = test_files[0]
        print(f"--- Testing expert_loader.py on {test_file} ---")
        try:
            raw_data = np.load(test_file, allow_pickle=True)
            print("Raw NPZ keys:", raw_data.files)
            
            parsed = load_expert_npz(test_file)
            print("Parsed Expert Data Shapes:")
            for k, v in parsed.items():
                print(f"  {k}: {v.shape}")
        except Exception as e:
            print("Test failed:", e)
    else:
        print("No .npz files found in the project to test.")
