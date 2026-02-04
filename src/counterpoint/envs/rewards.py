import numpy as np

class RewardComponent:
    def calculate(self, env, action, **kwargs):
        raise NotImplementedError

class MovementPenalty(RewardComponent):
    def calculate(self, env, action, **kwargs):
        target_hand_pos = action["hand_position"]
        current_hand_pos = env._hand_pos
        
        dist = abs(target_hand_pos - current_hand_pos)
        
        # Allow free movement for the first step (initial positioning)
        if env._current_step == 0:
            return 0.0
        elif dist == 0:
            return 0.0
        elif dist == 1:
            return -1.0
        else:
            #return -(1.0 + (dist - 1.0) / 6.0)
            return -dist * 10

class KeyChangePenalty(RewardComponent):
    def calculate(self, env, action, **kwargs):
        # Allow free setup for the first step
        if env._current_step == 0:
            return 0.0
            
        fingers_black = action["fingers_black"]
        if env._last_action is not None:
             prev_blacks = env._last_action["fingers_black"]
        else:
             prev_blacks = np.zeros(5, dtype=int)
             
        changed_count = np.sum(np.bitwise_xor(fingers_black, prev_blacks))
        return -(changed_count * 2)

class WrongColorPenalty(RewardComponent):
    """Penalize pressing wrong color key for current target."""
    def calculate(self, env, action, **kwargs):
        if env._current_step >= len(env._score_targets):
            return 0.0
        
        target_note, target_is_black = env._score_targets[env._current_step]
        fingers_pressed = action["fingers"]
        fingers_black = action["fingers_black"]
        
        penalty = 0.0
        for i, pressed in enumerate(fingers_pressed):
            if pressed == 1:
                is_black_pressed = fingers_black[i] == 1
                if is_black_pressed != bool(target_is_black):
                    penalty -= 2.0  # Each wrong-color finger: -2
        return penalty


class FingerRepetitionPenalty(RewardComponent):
    """
    Penalize using the same finger twice in a row when the note changes.
    
    This encourages proper finger alternation which is essential for
    good piano technique.
    """
    def __init__(self, penalty: float = -2.0):
        self.penalty = penalty
    
    def calculate(self, env, action, **kwargs):
        # Skip first step - no previous note to compare
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        # Get current and previous notes
        if env._current_step >= len(env._score_targets):
            return 0.0
        if env._current_step - 1 < 0:
            return 0.0
            
        current_note, _ = env._score_targets[env._current_step]
        prev_note, _ = env._score_targets[env._current_step - 1]
        
        # If same note, no penalty for same finger
        if current_note == prev_note:
            return 0.0
        
        # Check if any finger is repeated
        current_fingers = action["fingers"]
        prev_fingers = env._last_action["fingers"]
        
        penalty = 0.0
        for i in range(5):
            if current_fingers[i] == 1 and prev_fingers[i] == 1:
                # Same finger used for different notes
                penalty += self.penalty
        
        return penalty


class AccuracyReward(RewardComponent):
    def calculate(self, env, action, **kwargs):
        if env._current_step >= len(env._score_targets):
             return 0.0
             
        target_note, target_is_black = env._score_targets[env._current_step]
        
        fingers_pressed = action["fingers"]
        fingers_black = action["fingers_black"]
        target_hand_pos = action["hand_position"]
        
        fingers_active_indices = [i for i, x in enumerate(fingers_pressed) if x == 1]
        
        correct_note_played = False
        false_note_played = False
        
        for finger_idx in fingers_active_indices:
            actual_note_played = target_hand_pos + finger_idx
            is_black_pressed = fingers_black[finger_idx] == 1
            
            note_match = (actual_note_played == target_note)
            color_match = (is_black_pressed == bool(target_is_black))
            
            if note_match and color_match:
                correct_note_played = True
            else:
                false_note_played = True
        
        if correct_note_played and not false_note_played:
            return 10.0, True # Success
        else:
            return -10.0, False # Failure (reduced from -100)


class CompletionReward(RewardComponent):
    """Reward for completing the entire score."""
    
    def __init__(self, bonus: float = 50.0):
        self.bonus = bonus
    
    def calculate(self, env, action, **kwargs):
        # Only give bonus on the last note of the score
        # The AccuracyReward handles success/failure - this just adds extra incentive
        is_last_note = (env._current_step == len(env._score_targets) - 1)
        if is_last_note:
            return self.bonus
        return 0.0

class RewardMixing:
    def __init__(self):
        self.components = []
        self.completion_components = []
        
    def add(self, component):
        if isinstance(component, CompletionReward):
            self.completion_components.append(component)
        else:
            self.components.append(component)
        
    def calculate(self, env, action):
        total_reward = 0.0
        success = False
        
        # Calculate standard rewards
        for comp in self.components:
            res = comp.calculate(env, action)
            if isinstance(res, tuple):
                rew, succ = res
                total_reward += rew
                if succ:
                    success = True
            else:
                total_reward += res
        
        # Only add completion rewards if the step was successful
        if success:
            for comp in self.completion_components:
                total_reward += comp.calculate(env, action)
                
        return total_reward, success
