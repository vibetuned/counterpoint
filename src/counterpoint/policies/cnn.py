import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin

from .heads import PriorityHead
from .mixins import MaskedMultiCategoricalMixin

# =============================================================================
# Convolutional Policy & Value
# =============================================================================
# Treats the grid as a 2D spatial structure with lookahead as channels
# Grid shape: (batch, 2, 52, 10) -> reshape to (batch, 10, 2, 52) for conv

class ConvPolicy(MaskedMultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        MaskedMultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        # Grid: 2 rows x 52 cols x 10 lookahead -> treat as 10 channels, 2x52 spatial
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2  # hand_state + relative_target
        self.mask_size = 6  # fingers_black(5) + num_notes(1)
        self.hidden_size = 128  # Shared hidden size
        
        # Convolutional layers (input: 10 channels, 2 height, 52 width)
        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=(2, 5), stride=1, padding=(0, 2)),  # -> (32, 1, 52)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1, padding=(0, 2)),  # -> (64, 1, 52)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 5), stride=2, padding=(0, 2)),  # -> (64, 1, 26)
            nn.ReLU(),
            nn.Flatten(),  # -> 64 * 26 = 1664
        )
        
        # Shared feature extraction
        self.feature_fc = nn.Sequential(
            nn.Linear(1664 + self.extra_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.ReLU(),
        )
        
        # Priority head: selects which finger to use
        self.priority_head = PriorityHead(input_size=self.hidden_size, hidden_size=64)
        
        # Action head
        self.action_head = nn.Linear(self.hidden_size, sum(action_space.nvec))
        
    def set_tau(self, tau):
        """Update Gumbel-Softmax temperature."""
        if hasattr(self, "priority_head"):
            self.priority_head.set_tau(tau)

    def compute(self, inputs, role=""):
        x = inputs["states"]  # (batch, 1048)
        batch_size = x.shape[0]
        
        # Mask is at the BEGINNING
        action_mask = x[:, :self.mask_size]  # (batch, 6)
        num_notes = action_mask[:, -1].int()  # (batch,)
        
        # Grid follows mask
        grid_start = self.mask_size
        grid_end = grid_start + self.grid_size
        grid_flat = x[:, grid_start:grid_end]  # (batch, 1040)
        
        # Extra features are at the END
        extra = x[:, grid_end:]  # (batch, 2)
        
        # Reshape grid: (batch, 2, 52, 10) -> (batch, 10, 2, 52)
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        
        # Convolutional features
        conv_features = self.conv(grid)  # (batch, 1664)
        
        # Concatenate and extract shared features
        combined = torch.cat([conv_features, extra], dim=1)
        shared_features = self.feature_fc(combined)  # (batch, hidden_size)
        
        # Get finger mask from priority head using dynamic num_notes
        max_num_notes = max(1, int(num_notes.max().item()))
        finger_mask, _ = self.priority_head.get_finger_mask(
            shared_features, 
            num_fingers=max_num_notes,
            training=self.training
        )
        
        # Get action logits
        logits = self.action_head(shared_features)
        masked_logits = self._apply_action_mask(logits, action_mask, finger_mask=finger_mask)
        
        return masked_logits, {}


class ConvValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        self.mask_size = 6
        
        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=(2, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 5), stride=2, padding=(0, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1664 + self.extra_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]
        batch_size = x.shape[0]
        
        # Mask is first
        grid_start = self.mask_size
        grid_end = grid_start + self.grid_size
        
        grid_flat = x[:, grid_start:grid_end]
        extra = x[:, grid_end:]
        
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        conv_features = self.conv(grid)
        combined = torch.cat([conv_features, extra], dim=1)
        
        return self.fc(combined), {}
