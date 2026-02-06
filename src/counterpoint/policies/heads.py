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
            nn.Linear(hidden_size, 5) # Logits for 5 fingers
        )
        
    def forward(self, features):
        """
        Args:
            features: (batch, input_size) - observation features
            
        Returns:
            finger_logits: (batch, 5) - logits for each finger
        """
        return self.net(features)
        
    def get_finger_mask(self, features, num_fingers=1, training=True):
        """
        Get finger mask from priority head.
        
        Args:
            features: (batch, input_size)
            num_fingers: how many fingers to activate (default=1 for single notes)
            training: if True, samples from distribution; if False, takes argmax
            
        Returns:
            finger_mask: (batch, 5) - 1 for selected finger(s), 0 for others
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        
        if training:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs)
            # For multi-finger, we might want multiple samples without replacement
            # But Categorical only gives one.
            # If num_fingers > 1, we can use Gumbel-Softmax or iterative sampling
            # For now, let's assume num_fingers is usually 1, or use Top-K for deterministic
            
            # Simple sampling for 1 finger:
            if num_fingers == 1:
                indices = dist.sample() # (batch,)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, indices.unsqueeze(1), 1.0)
            else:
                # If we need multiple distinct fingers, sampling is trickier.
                # A simple approximation: Take top-k samples or use TopK for now.
                # Realistically, for chords, we want 'num_fingers' distinct fingers.
                
                # Let's stick to Top-K sampling for stability or just Top-K
                # If training=True, maybe adds noise?
                # For now, let's just use Top-K (deterministic given noise if added)
                # To make it stochastic, we could add Gumbel noise to logits before Top-K
                
                # Gumbel-Max trick
                u = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(u + 1e-9) + 1e-9)
                perturbed_logits = logits + gumbel
                
                _, indices = torch.topk(perturbed_logits, k=num_fingers, dim=-1)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, indices, 1.0)
                
        else:
            # Deterministic: Top-K
            _, indices = torch.topk(probs, k=num_fingers, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, indices, 1.0)
            
        return mask, logits
