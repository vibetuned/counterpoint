import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin

from .heads import PriorityHead
from .mixins import MaskedMultiCategoricalMixin
from .layers import PositionalEncoding

# =============================================================================
# Decoder-based Transformer Policy & Value
# =============================================================================
# Uses TransformerDecoder with causal (masked) attention
# Each position can only attend to earlier positions (autoregressive)
# Sequence is reversed: future notes first, current note last

class DecoderPolicy(MaskedMultiCategoricalMixin, Model):
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
        
        # Transformer decoder layer with causal masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Learnable memory/context for the decoder (no encoder in this setup)
        self.memory = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Shared feature extraction
        self.feature_fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, self.hidden_size),
            nn.ReLU(),
        )
        
        # Priority head: selects which finger to use
        self.priority_head = PriorityHead(input_size=self.hidden_size, hidden_size=64)
        
        # Action head
        self.action_head = nn.Linear(self.hidden_size, sum(action_space.nvec))
        
        # Pre-compute causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(self.lookahead))
    
    def _generate_causal_mask(self, size):
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

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
        
        # Reshape to (batch, 10, 104) - 10 tokens of 104 features
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)  # (batch, 10, 2, 52)
        tokens = grid.reshape(batch_size, self.lookahead, -1)  # (batch, 10, 104)
        
        # Reverse sequence: token 0 becomes furthest ahead, token 9 becomes current
        tokens = torch.flip(tokens, dims=[1])
        
        # Embed tokens and add positional encoding
        tokens = self.token_embed(tokens)  # (batch, 10, d_model)
        tokens = self.pos_encoding(tokens)
        
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # Decoder with causal masking
        decoded = self.decoder(tokens, memory, tgt_mask=self.causal_mask)  # (batch, 10, d_model)
        
        # Use last token (current note after reversal) as output representation
        output = decoded[:, -1, :]  # (batch, d_model)
        
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


class DecoderValue(DeterministicMixin, Model):
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
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.memory = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.register_buffer('causal_mask', self._generate_causal_mask(self.lookahead))
    
    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def compute(self, inputs, role=""):
        x = inputs["states"]
        batch_size = x.shape[0]
        
        # Mask is first
        grid_start = self.mask_size
        grid_end = grid_start + self.grid_size
        
        grid_flat = x[:, grid_start:grid_end]
        extra = x[:, grid_end:]
        
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        tokens = grid.reshape(batch_size, self.lookahead, -1)
        
        # Reverse sequence
        tokens = torch.flip(tokens, dims=[1])
        
        tokens = self.token_embed(tokens)
        tokens = self.pos_encoding(tokens)
        
        memory = self.memory.expand(batch_size, -1, -1)
        decoded = self.decoder(tokens, memory, tgt_mask=self.causal_mask)
        
        # Use last token (current note)
        output = decoded[:, -1, :]
        
        combined = torch.cat([output, extra], dim=1)
        
        return self.fc(combined), {}
