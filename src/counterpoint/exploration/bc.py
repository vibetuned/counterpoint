
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from counterpoint.data.demonstrations import DemonstrationGenerator


class BehaviorCloningLoss:
    """
    Auxiliary behavior cloning loss for PPO training.
    
    Samples expert demonstrations and computes cross-entropy loss
    between policy logits and expert actions.
    """
    
    def __init__(self, 
                 coefficient: float = 0.5,
                 decay_rate: float = 0.995,
                 min_coefficient: float = 0.05,
                 device: str = "cpu"):
        """
        Args:
            coefficient: Initial BC loss weight
            decay_rate: Multiplicative decay per epoch
            min_coefficient: Minimum BC coefficient
            device: Torch device
        """
        self.coefficient = coefficient
        self.initial_coefficient = coefficient
        self.decay_rate = decay_rate
        self.min_coefficient = min_coefficient
        self.device = device
        
        self.demo_generator = DemonstrationGenerator()
        self.epoch = 0
    
    def decay_coefficient(self):
        """Apply decay to BC coefficient (call once per epoch)."""
        self.epoch += 1
        self.coefficient = max(
            self.min_coefficient,
            self.initial_coefficient * (self.decay_rate ** self.epoch)
        )
    
    def reset_coefficient(self):
        """Reset coefficient to initial value."""
        self.coefficient = self.initial_coefficient
        self.epoch = 0
    
    def sample_demonstrations(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample expert demonstrations as tensors.
        
        Returns:
            (observations, actions) as torch tensors on self.device
        """
        obs, actions = self.demo_generator.sample_batch_flat(batch_size)
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        return obs_tensor, actions_tensor
    
    def compute_loss(self, policy: nn.Module, batch_size: int = 64) -> torch.Tensor:
        """
        Compute behavior cloning loss.
        
        Args:
            policy: Policy network that takes observations and returns logits
            batch_size: Number of demonstration samples
            
        Returns:
            Weighted BC loss
        """
        if self.coefficient <= 0:
            return torch.tensor(0.0, device=self.device)
        
        obs, expert_actions = self.sample_demonstrations(batch_size)
        
        # Get policy logits
        # Policy expects {"states": obs} and returns (logits, {})
        policy_output, _ = policy.compute({"states": obs})
        
        # policy_output is concatenated logits for all action branches
        # Shape: (batch, 20) = 10 branches x 2 logits each
        # expert_actions is shape (batch, 10) with values 0 or 1
        
        # Split logits into 10 branches of 2 logits each
        logits_split = policy_output.view(-1, 10, 2)  # (batch, 10, 2)
        
        # Compute cross-entropy for each branch
        total_loss = 0.0
        for i in range(10):
            branch_logits = logits_split[:, i, :]  # (batch, 2)
            branch_targets = expert_actions[:, i]  # (batch,)
            total_loss += F.cross_entropy(branch_logits, branch_targets)
        
        # Average over branches and apply coefficient
        bc_loss = (total_loss / 10) * self.coefficient
        
        return bc_loss
    
    def get_stats(self) -> dict:
        """Return current BC stats for logging."""
        return {
            "bc/coefficient": self.coefficient,
            "bc/epoch": self.epoch
        }
