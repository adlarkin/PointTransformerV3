import torch
from torch import nn


class StabilityPredictor(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            # Input layer -> 1st Hidden layer (128 neurons)
            # Input layer is feature_dim * 2 because there are 2 pointclouds
            # of size feature_dim
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            # 1st Hidden layer (128) -> 2nd Hidden layer (128)
            nn.Linear(128, 128),
            nn.ReLU(),
            # 2nd Hidden layer (128) -> Output layer (1 neuron)
            nn.Linear(128, 1),
        )

    def forward(self, features_a, features_b):
        combined = torch.cat([features_a, features_b], dim=-1)
        return self.layers(combined)

    def device(self):
        return next(self.parameters()).device


def load_with_weights(weights_file: str, eval: bool) -> StabilityPredictor:
    """Load a StabilityPredictor network with initialized weights.

    Args:
        weights_file: Full path to the file containing the model weights.
        eval: Whether the model should be in eval mode (true) or not (false).

    Returns:
        A StabilityPredictor instance in EVAL mode with weights initialized via `weights_file`.
    """
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    # Load the existing model weights onto the CPU first, and then move the model (with the
    # loaded weights) to the current device. This handles the scenario of loading model weights
    # that were saved to a different device.
    model = StabilityPredictor()
    state_dict = torch.load(
        weights_file,
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    if eval:
        model.eval()
    return model
