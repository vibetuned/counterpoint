import torch
from skrl.models.torch import MultiCategoricalMixin

# Large negative value for masking (avoids numerical issues with -inf)
MASK_VALUE = -1e8

class MaskedMultiCategoricalMixin(MultiCategoricalMixin):
    """
    MultiCategorical mixin with action masking and priority head support.
    
    Expects action_mask in the last 6 positions of flattened observations.
    Mask format: [fingers_black_mask(5), num_notes(1)]
    
    Logic:
    1. Priority Head selects which finger(s) to use (e.g., Index)
    2. Env Mask tells us valid actions for that finger (e.g., White Key Only)
    3. Final Mask = PriorityMask & EnvMask
    """
    def __init__(self, unnormalized_log_prob=True, reduction="sum"):
        super().__init__(unnormalized_log_prob, reduction)
    
    def _apply_action_mask(self, logits, action_mask, finger_mask):
        """
        Apply action mask to logits.
        
        Args:
            logits: (batch, 20) - 2 logits per binary choice
            action_mask: (batch, 6) - env mask [fingers_black_mask(5), num_notes(1)]
            finger_mask: (batch, 5) - priority head mask (which fingers to use)
            
        Returns:
            masked_logits: (batch, 20)
        """
        # action_mask contains:
        # 0-4: fingers_black_mask (1 if black key is forced, 0 if white/flexible)
        # 5: num_notes (count of notes to play)
        
        env_black_mask = action_mask[:, :5] # (batch, 5)
        
        # We need to constructing a mask for the 20 logits (10 binary actions x 2 classes)
        # Actions: [F1_on/off, F2_on/off, ..., F5_on/off, B1_yes/no, ..., B5_yes/no]
        # BUT wait, our FlattenActionWrapper defines nvec = [2]*10
        # Indices:
        # 0: F1 (0=off, 1=on)
        # 1: F2
        # ...
        # 4: F5
        # 5: B1 (0=white, 1=black)
        # ...
        # 9: B5
        
        # Logits shape (batch, 20).
        # Pairs: (0,1) -> F1, (2,3) -> F2, ..., (18,19) -> B5
        
        final_mask = torch.zeros_like(logits)
        
        # 1. Finger Open/Close Actions (Indices 0-9 in logits aka 0-4 in action space)
        # Driven by PriorityHead (finger_mask)
        # finger_mask[i] == 1 => Finger i MUST be ON (Action 1)
        # finger_mask[i] == 0 => Finger i MUST be OFF (Action 0)
        
        for i in range(5):
            # Logit indices for Finger i
            idx_off = 2 * i
            idx_on = 2 * i + 1
            
            # finger_mask[:, i] is 1 if selected, 0 if not
            selected = finger_mask[:, i] # (batch,)
            
            # If selected (1): Mask OFF(0) -> -inf
            # If not selected (0): Mask ON(1) -> -inf
            
            # We want to subtract MASK_VALUE from invalid options? 
            # Or just set them to MASK_VALUE? 
            # SKRL mixin usually adds the mask? No, usually we just manipulate logits.
            
            # If selected: allow ON, forbid OFF
            final_mask[:, idx_off] = torch.where(selected > 0, torch.tensor(MASK_VALUE).to(logits.device), torch.tensor(0.0).to(logits.device))
            
            # If NOT selected: allow OFF, forbid ON
            final_mask[:, idx_on] = torch.where(selected == 0, torch.tensor(MASK_VALUE).to(logits.device), torch.tensor(0.0).to(logits.device))

        # 2. Black Key Actions (Indices 10-19 in logits aka 5-9 in action space)
        # Driven by env_black_mask
        # values in env_black_mask:
        # 0 -> White Key (Force Black=0)
        # 1 -> Black Key (Force Black=1)
        # 0.5 -> Flexible (Allow both) - Wait, previous logic was binary?
        # Let's check environment. Usually 0/1/0.5 or -1. 
        # Assuming from 'fingers_black_mask' name it might be specific.
        # In this specific env:
        # 0.0 -> White key
        # 1.0 -> Black key
        # 0.5 -> Flexible (e.g. key is not pressed, or doesn't matter? Actually if finger is OFF, this action is irrelevant but we still mask it for consistency)
        
        for i in range(5):
            # Logit indices for Black Key i
            # Action space index: 5+i
            # Logit indices: 2*(5+i), 2*(5+i)+1 -> 10+2i, 11+2i
            idx_white = 10 + 2 * i
            idx_black = 11 + 2 * i
            
            req = env_black_mask[:, i] # (batch,)
            
            # If req == 0.0 (White): Forbid Black (idx_black)
            final_mask[:, idx_black] += torch.where(req == 0.0, torch.tensor(MASK_VALUE).to(logits.device), torch.tensor(0.0).to(logits.device))
            
            # If req == 1.0 (Black): Forbid White (idx_white)
            final_mask[:, idx_white] += torch.where(req == 1.0, torch.tensor(MASK_VALUE).to(logits.device), torch.tensor(0.0).to(logits.device))
            
            # If req == 0.5 (Flexible), add 0 (allow both)
            
        return logits + final_mask
