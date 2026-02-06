
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class RNDNetwork(nn.Module):
    """
    Random Network Distillation for intrinsic curiosity.
    
    Uses a fixed random target network and a trainable predictor network.
    The prediction error serves as an intrinsic reward for novel states.
    
    Reference: Burda et al. "Exploration by Random Network Distillation" (2018)
    """
    
    def __init__(self, obs_dim: int = 1048, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.mask_size = 6
        self.feature_dim = obs_dim - self.mask_size
        
        # Fixed random target network (not trained)
        self.target = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target and predictor outputs.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            
        Returns:
            (target_features, predictor_features)
        """
        # Slice off action mask (first 6 elements)
        # RND should only see state features (grid + hand_state + relative_target)
        # obs shape is 1048, we want last 1042
        features = obs[:, 6:]
        
        with torch.no_grad():
            target_features = self.target(features)
        predictor_features = self.predictor(features)
        return target_features, predictor_features
    
    def intrinsic_reward(self, obs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute intrinsic reward as prediction error.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            normalize: Whether to normalize rewards
            
        Returns:
            Intrinsic rewards of shape (batch,)
        """
        target_features, predictor_features = self.forward(obs)
        
        # MSE between target and predictor (per sample)
        rewards = ((target_features - predictor_features) ** 2).mean(dim=1)
        
        if normalize:
            # Update running statistics
            batch_mean = rewards.mean().item()
            # Use unbiased=False to avoid NaN when batch_size=1
            batch_std = rewards.std(unbiased=False).item() + 1e-8
            
            self.reward_count += 1
            alpha = 1.0 / self.reward_count
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
            self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std
            
            # Normalize
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        
        return rewards
    
    def compute_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute RND predictor loss for training.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            
        Returns:
            MSE loss between predictor and target
        """
        target_features, predictor_features = self.forward(obs)
        return F.mse_loss(predictor_features, target_features.detach())
