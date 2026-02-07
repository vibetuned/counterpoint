import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin

from .heads import PriorityHead
from .mixins import MaskedMultiCategoricalMixin
from .layers import PositionalEncoding

# =============================================================================
# Transformer Policy & Value
# =============================================================================
# Treats each lookahead timestep as a token (10 tokens of dim 104 each)
# Each token = 2*52 = 104 features (one slice of the grid)
# Sequence is REVERSED: token 0 = furthest ahead, token 9 = current note
# This allows causal attention to flow from future context to current decision

class TransformerPolicy(MaskedMultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        MaskedMultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        self.mask_size = 6  # fingers_black(5) + num_notes(1)
        self.lookahead = 10
        self.token_dim = 2 * 52  # 104 features per timestep
        self.d_model = d_model
        self.hidden_size = 128
        
        # Project each token to d_model dimensions
        self.token_embed = nn.Linear(self.token_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.lookahead)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Shared feature extraction
        self.feature_fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, self.hidden_size),
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
        
        # Observation structure (alphabetical keys from Dict space):
        # 0: action_mask (6)
        # 1: grid (1040)
        # 2: hand_state (1)
        # 3: relative_target (1)
        
        # Slicing
        # Mask is at the BEGINNING now
        action_mask = x[:, :self.mask_size]  # (batch, 6)
        num_notes = action_mask[:, -1].int()  # (batch,)
        
        # Grid follows mask
        grid_start = self.mask_size
        grid_end = grid_start + self.grid_size
        grid_flat = x[:, grid_start:grid_end]  # (batch, 1040)
        
        # Extra features (hand_state, relative_target) are at the END
        extra = x[:, grid_end:]  # (batch, 2)
        
        # Reshape to (batch, 10, 104) - 10 tokens of 104 features
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)  # (batch, 10, 2, 52)
        tokens = grid.reshape(batch_size, self.lookahead, -1)  # (batch, 10, 104)
        
        # Reverse sequence: token 0 becomes furthest ahead, token 9 becomes current
        tokens = torch.flip(tokens, dims=[1])
        
        # Embed tokens and add positional encoding
        tokens = self.token_embed(tokens)  # (batch, 10, d_model)
        tokens = self.pos_encoding(tokens)
        
        # Transformer encoding
        encoded = self.transformer(tokens)  # (batch, 10, d_model)
        
        # Use last token (current note after reversal) as output representation
        output = encoded[:, -1, :]  # (batch, d_model)
        
        # Concatenate with extra features and extract shared features
        combined = torch.cat([output, extra], dim=1)
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


class TransformerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        self.mask_size = 6
        self.lookahead = 10
        self.token_dim = 2 * 52
        self.d_model = d_model
        
        self.token_embed = nn.Linear(self.token_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.lookahead)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]
        batch_size = x.shape[0]
        
        # Observation structure (alphabetical keys from Dict space):
        # [action_mask(6), grid(1040), hand_state(1), relative_target(1)]
        
        # Mask is first (we don't need it for Value, but need to skip it)
        grid_start = self.mask_size
        grid_end = grid_start + self.grid_size
        
        grid_flat = x[:, grid_start:grid_end]
        extra = x[:, grid_end:]
        
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        tokens = grid.reshape(batch_size, self.lookahead, -1)
        
        # Reverse sequence: token 0 becomes furthest ahead, token 9 becomes current
        tokens = torch.flip(tokens, dims=[1])
        
        tokens = self.token_embed(tokens)
        tokens = self.pos_encoding(tokens)
        
        encoded = self.transformer(tokens)
        # Use last token (current note after reversal)
        output = encoded[:, -1, :]
        
        combined = torch.cat([output, extra], dim=1)
        
        return self.fc(combined), {}
