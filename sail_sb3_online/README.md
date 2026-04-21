# SAIL PyTorch Migration (SB3)

This directory contains the clean PyTorch migration of the SAIL repository, structured using `stable-baselines3` as an external dependency.

## Structure
- `algorithms/`: SB3 subclasses (e.g., SAIL extending TD3).
- `reward_models/`: Standalone PyTorch models (Discriminator, Preference Model).
- `datasets/`: PyTorch Dataset/DataLoader implementations for expert/teacher data.
- `scripts/`: Clean entry points for training and evaluation.
- `utils/`: SB3 callbacks and environment wrappers.

*Note: This structure preserves compatibility with the original OpenAI Gym and mujoco-py stack by utilizing SB3 v1.8.0.*
