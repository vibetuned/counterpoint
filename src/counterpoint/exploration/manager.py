
import torch
import torch.nn as nn

from .rnd import RNDNetwork
from .bc import BehaviorCloningLoss

class ExplorationManager:
    """
    Manages both RND and BC exploration mechanisms.
    
    Provides a unified interface for training integration.
    """
    
    def __init__(self,
                 use_bc: bool = True,
                 use_rnd: bool = True,
                 bc_coefficient: float = 0.5,
                 bc_decay_rate: float = 0.995,
                 bc_min_coefficient: float = 0.05,
                 rnd_coefficient: float = 0.1,
                 obs_dim: int = 1042,
                 device: str = "cpu"):
        
        self.use_bc = use_bc
        self.use_rnd = use_rnd
        self.rnd_coefficient = rnd_coefficient
        self.device = device
        
        # Initialize BC
        self.bc = None
        if use_bc:
            self.bc = BehaviorCloningLoss(
                coefficient=bc_coefficient,
                decay_rate=bc_decay_rate,
                min_coefficient=bc_min_coefficient,
                device=device
            )
        
        # Initialize RND
        self.rnd = None
        if use_rnd:
            self.rnd = RNDNetwork(obs_dim=obs_dim).to(device)
            self.rnd_optimizer = torch.optim.Adam(
                self.rnd.predictor.parameters(), 
                lr=1e-4
            )
    
    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute RND intrinsic reward if enabled."""
        if not self.use_rnd or self.rnd is None:
            return torch.zeros(obs.shape[0], device=self.device)
        
        return self.rnd.intrinsic_reward(obs) * self.rnd_coefficient
    
    def compute_bc_loss(self, policy: nn.Module, batch_size: int = 64) -> torch.Tensor:
        """Compute BC loss if enabled."""
        if not self.use_bc or self.bc is None:
            return torch.tensor(0.0, device=self.device)
        
        return self.bc.compute_loss(policy, batch_size)
    
    def update_rnd(self, obs: torch.Tensor):
        """Update RND predictor network."""
        if not self.use_rnd or self.rnd is None:
            return
        
        self.rnd_optimizer.zero_grad()
        rnd_loss = self.rnd.compute_loss(obs)
        rnd_loss.backward()
        self.rnd_optimizer.step()
    
    def decay_bc(self):
        """Decay BC coefficient (call once per epoch)."""
        if self.bc is not None:
            self.bc.decay_coefficient()
    
    def get_stats(self) -> dict:
        """Get exploration stats for logging."""
        stats = {}
        if self.bc is not None:
            stats.update(self.bc.get_stats())
        return stats
