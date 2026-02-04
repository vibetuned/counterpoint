import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin, MultiCategoricalMixin

# Wrapper to flatten Action Space
class FlattenActionWrapper(gym.ActionWrapper):
    """Flatten Dict action space to MultiDiscrete for SKRL compatibility."""
    def __init__(self, env):
        super().__init__(env)
        # 5x Fingers(2) + 5x Black(2) = 10 binary choices
        # No hand_position - it's derived automatically in the env
        nvec = [2]*5 + [2]*5
        self.action_space = gym.spaces.MultiDiscrete(nvec)
        
    def action(self, action):
        # Convert MultiDiscrete array to Dict
        # action is array of [f0, f1, f2, f3, f4, b0, b1, b2, b3, b4]
        return {
            "fingers": action[0:5],
            "fingers_black": action[5:10]
        }
    
    def reverse_action(self, action):
        # Not needed for training
        pass

class SimpleBaselinePolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob)

        # Input is Dict, flattened by SKRL
        # (2*52*10) + 1 + 1 = 1042 (grid + hand_state + relative_target)
        self.input_size = 1042
        
        # Output: 10 binary actions (5 fingers + 5 black keys)
        # sum(nvec) = 2*5 + 2*5 = 20 logits (2 per binary choice)
        
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


# =============================================================================
# Convolutional Policy & Value
# =============================================================================
# Treats the grid as a 2D spatial structure with lookahead as channels
# Grid shape: (batch, 2, 52, 10) -> reshape to (batch, 10, 2, 52) for conv

class ConvPolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        # Grid: 2 rows x 52 cols x 10 lookahead -> treat as 10 channels, 2x52 spatial
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2  # hand_state + relative_target
        
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
        
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(1664 + self.extra_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, sum(action_space.nvec))
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]  # (batch, 1042)
        batch_size = x.shape[0]
        
        # Split grid from extra features
        grid_flat = x[:, :self.grid_size]  # (batch, 1040)
        extra = x[:, self.grid_size:]  # (batch, 2)
        
        # Reshape grid: (batch, 2, 52, 10) -> (batch, 10, 2, 52)
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        
        # Convolutional features
        conv_features = self.conv(grid)  # (batch, 1664)
        
        # Concatenate with extra features
        combined = torch.cat([conv_features, extra], dim=1)
        
        return self.fc(combined), {}


class ConvValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        
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
        
        grid_flat = x[:, :self.grid_size]
        extra = x[:, self.grid_size:]
        
        grid = grid_flat.view(batch_size, 2, 52, 10).permute(0, 3, 1, 2)
        conv_features = self.conv(grid)
        combined = torch.cat([conv_features, extra], dim=1)
        
        return self.fc(combined), {}


# =============================================================================
# Transformer Policy & Value
# =============================================================================
# Treats each lookahead timestep as a token (10 tokens of dim 104 each)
# Each token = 2*52 = 104 features (one slice of the grid)
# Sequence is REVERSED: token 0 = furthest ahead, token 9 = current note
# This allows causal attention to flow from future context to current decision

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerPolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        self.lookahead = 10
        self.token_dim = 2 * 52  # 104 features per timestep
        self.d_model = d_model
        
        # Project each token to d_model dimensions
        self.token_embed = nn.Linear(self.token_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.lookahead)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, sum(action_space.nvec))
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]  # (batch, 1042)
        batch_size = x.shape[0]
        
        # Split grid from extra features
        grid_flat = x[:, :self.grid_size]  # (batch, 1040)
        extra = x[:, self.grid_size:]  # (batch, 2)
        
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
        
        # Concatenate with extra features
        combined = torch.cat([output, extra], dim=1)
        
        return self.fc(combined), {}


class TransformerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
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
        
        grid_flat = x[:, :self.grid_size]
        extra = x[:, self.grid_size:]
        
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


# =============================================================================
# Decoder-based Transformer Policy & Value
# =============================================================================
# Uses TransformerDecoder with causal (masked) attention
# Each position can only attend to earlier positions (autoregressive)
# Sequence is reversed: future notes first, current note last

class DecoderPolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
        self.lookahead = 10
        self.token_dim = 2 * 52  # 104 features per timestep
        self.d_model = d_model
        
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
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, sum(action_space.nvec))
        )
        
        # Pre-compute causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(self.lookahead))
    
    def _generate_causal_mask(self, size):
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def compute(self, inputs, role=""):
        x = inputs["states"]  # (batch, 1042)
        batch_size = x.shape[0]
        
        # Split grid from extra features
        grid_flat = x[:, :self.grid_size]  # (batch, 1040)
        extra = x[:, self.grid_size:]  # (batch, 2)
        
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
        
        # Concatenate with extra features
        combined = torch.cat([output, extra], dim=1)
        
        return self.fc(combined), {}


class DecoderValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 d_model=128, nhead=4, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.grid_size = 2 * 52 * 10
        self.extra_features = 2
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
        
        grid_flat = x[:, :self.grid_size]
        extra = x[:, self.grid_size:]
        
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
