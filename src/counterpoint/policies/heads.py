import torch
import torch.nn as nn

class PriorityHead(nn.Module):
    """
    Neural network that selects which finger to use.
    
    Takes observation features and outputs a distribution over 5 fingers.
    During inference, samples one finger and masks the others.
    """
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size), # Normalize features before final projection
            nn.Linear(hidden_size, 5) # Logits for 5 fingers
        )
        self.tau = 1.0 # Temperature for Gumbel-Softmax (lower = more deterministic)
        
    def set_tau(self, tau):
        self.tau = tau
        
    def forward(self, features):
        """
        Args:
            features: (batch, input_size) - observation features
            
        Returns:
            finger_logits: (batch, 5) - logits for each finger
        """
        return self.net(features)
        
    def get_finger_mask(self, features, num_fingers=1, training=True, env_finger_mask=None):
        """
        Get finger mask from priority head.
        
        Args:
            features: (batch, input_size)
            num_fingers: how many fingers to activate (default=1 for single notes)
            training: if True, samples from distribution; if False, takes argmax
            env_finger_mask: (batch, 5) optional - 1=allowed, 0=forbidden by env
            
        Returns:
            finger_mask: (batch, 5) - 1 for selected finger(s), 0 for others
        """
        logits = self.forward(features)
        
        # Apply env_finger_mask to logits BEFORE sampling
        # This ensures we only sample from allowed fingers
        if env_finger_mask is not None:
            # Mask forbidden fingers with large negative value
            # env_finger_mask: 1=allowed, 0=forbidden
            # We want to add -inf to forbidden fingers
            mask_value = -1e8
            forbidden_mask = (1 - env_finger_mask) * mask_value  # 0 for allowed, -1e8 for forbidden
            logits = logits + forbidden_mask
        
        if training:
            if num_fingers == 1:
                # Use Gumbel-Softmax (Straight-Through) to allow gradient flow
                # hard=True returns one-hot tensor, but gradients flow through softmax
                mask = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
            else:
                # For >1 fingers, use Gumbel-TopK (stochastic top-k)
                # Gumbel-Max trick for sampling without replacement
                u = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(u + 1e-9) + 1e-9)
                perturbed_logits = logits + gumbel
                
                # We can't easily make this differentiable for TopK > 1 with ST-estimator
                # without custom implementation. But for now, stochastic sampling is key.
                # If we really need gradients here, we'd need a relaxed k-hot estimator.
                # Assuming standard Gumbel-Max is okay for exploration even if gradients are broken?
                # No, if gradients assume broken, it won't learn.
                # But typically num_notes=1.
                
                _, indices = torch.topk(perturbed_logits, k=num_fingers, dim=-1)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, indices, 1.0)
                
                # To maintain some gradient flow approximation could use soft top-k,
                # but let's stick to hard sampling for now as num_fingers=1 is dominant.
                
        else:
            # Deterministic: Top-K (which matches argmax for k=1)
            probs = torch.softmax(logits, dim=-1)
            _, indices = torch.topk(probs, k=num_fingers, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, indices, 1.0)
            
        return mask, logits
