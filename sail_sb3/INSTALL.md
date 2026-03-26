# Installation

1. Create a clean Conda environment:
   ```bash
   conda create -n sail_sb3_env python=3.8
   conda activate sail_sb3_env
   ```
2. Downgrade `setuptools` to fix a known bug installing `gym==0.21.0`:
   ```bash
   pip install wheel setuptools==65.5.0
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your PYTHONPATH to include the project root (if needed):
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
5. Verify PyTorch and Gym:
   ```python
   python -c "import torch, gym, stable_baselines3; print('Success!')"
   ```
