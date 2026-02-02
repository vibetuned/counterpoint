import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin, MultiCategoricalMixin

# Wrapper to flatten Action Space
class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Hand(52) + 5x Fingers(2) + 5x Black(2)
        nvec = [52] + [2]*5 + [2]*5
        self.action_space = gym.spaces.MultiDiscrete(nvec)
        
    def action(self, action):
        # Convert MultiDiscrete array to Dict
        # action is array of [hand, f0, f1, f2, f3, f4, b0, b1, b2, b3, b4]
        return {
            "hand_position": action[0],
            "fingers": action[1:6],
            "fingers_black": action[6:11]
        }
    
    def reverse_action(self, action):
        # Not needed for training usually
        pass

class SimpleBaselinePolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob)

        # Input is Dict
        # Manually calc size: (2*52*10) + 1 = 1041
        self.input_size = 1042  # (2*52*10) + 1 + 1 = grid + hand_state + relative_target
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, sum(action_space.nvec)) # Logits for all branches concatenated
        )

    def compute(self, inputs, role=""):
        # inputs["states"] is already flattened by SKRL wrapper if using Gymnasium
        # Shape: (batch, 1041)
        x = inputs["states"]
        return self.net(x), {}

# Value Model
class SimpleBaselineValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.input_size = 1042  # (2*52*10) + 1 + 1 = grid + hand_state + relative_target
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]
        return self.net(x), {}
