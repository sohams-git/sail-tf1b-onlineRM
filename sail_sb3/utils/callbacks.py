from stable_baselines3.common.callbacks import BaseCallback

class DiscriminatorLoggingCallback(BaseCallback):
    """
    An SB3 callback to log discriminator metrics (loss, accuracy, custom reward values)
    to TensorBoard or Weights & Biases during the training rollouts.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # TODO: Add specific logging for custom network parameters
        # e.g., self.logger.record("reward/discriminator", value)
        return True
