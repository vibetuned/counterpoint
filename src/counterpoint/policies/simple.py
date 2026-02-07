import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin

from .heads import PriorityHead
from .mixins import MaskedMultiCategoricalMixin

class SimpleBaselinePolicy(MaskedMultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        MaskedMultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        # Handle observation space shape
        if hasattr(observation_space, "shape") and observation_space.shape is not None:
             self.feature_size = observation_space.shape[0] - 6 
        else:
             # Fallback for Dict space or when shape is None (1048 total - 6 mask)
             self.feature_size = 1048 - 6
             
        self.mask_size = 6 # fingers_black(5) + num_notes(1)

        # Simple MLP
        self.net = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Priority head: selects which finger to use
        self.priority_head = PriorityHead(input_size=256, hidden_size=64)
        
        # Action head: outputs logits for all actions
        self.action_head = nn.Linear(256, sum(action_space.nvec))
        
    def set_tau(self, tau):
        """Update Gumbel-Softmax temperature."""
        if hasattr(self, "priority_head"):
            self.priority_head.set_tau(tau)

    def compute(self, inputs, role=""):
        x = inputs["states"]
        
        # Split features and mask
        # Action Mask is at the BEGINNING (first 6 elements)
        action_mask = x[:, :self.mask_size]
        num_notes = action_mask[:, -1].int()
        
        # Features are everything else
        features = x[:, self.mask_size:]
        
        # Shared features
        shared_features = self.net(features)
        
        # Get finger mask from priority head using dynamic num_notes
        max_num_notes = max(1, int(num_notes.max().item()))
        finger_mask, _ = self.priority_head.get_finger_mask(
            shared_features, 
            num_fingers=max_num_notes,
            training=self.training
        )
        
        # Get action logits
        logits = self.action_head(shared_features)
        
        # Apply mask
        masked_logits = self._apply_action_mask(logits, action_mask, finger_mask=finger_mask)
        
        # Debugging / Analysis could go here (e.g. returning masks)
        return masked_logits, {}


# Value Model (doesn't need masking - just uses features without mask)
class SimpleBaselineValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        if hasattr(observation_space, "shape") and observation_space.shape is not None:
             self.feature_size = observation_space.shape[0] - 6
        else:
             self.feature_size = 1048 - 6
             
        self.mask_size = 6
        self.net = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]
        # Remove mask features (first 6)
        features = x[:, self.mask_size:]
        return self.net(features), {}
