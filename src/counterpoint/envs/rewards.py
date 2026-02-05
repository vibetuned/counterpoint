import numpy as np

class RewardComponent:
    def calculate(self, env, action, **kwargs):
        raise NotImplementedError

class MovementPenalty(RewardComponent):
    """Penalize large hand movements with sublinear scaling."""
    def calculate(self, env, action, **kwargs):
        target_hand_pos = action["hand_position"]
        current_hand_pos = env._hand_pos
        
        dist = abs(target_hand_pos - current_hand_pos)
        
        # Allow free movement for the first step (initial positioning)
        if env._current_step == 0:
            return 0.0
        elif dist == 0:
            return 0.0
        elif dist <= 2:
            # Small movements are cheap
            return -0.5 * dist
        else:
            # Sublinear scaling for larger movements (logarithmic growth)
            return -1.0 - 0.5 * np.log(dist)

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
    """Reward correct note playing with asymmetric success/failure rewards."""
    def __init__(self, success_reward: float = 5.0, failure_penalty: float = -1.0):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
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
            return self.success_reward, True  # Success - larger positive
        else:
            return self.failure_penalty, False  # Failure - smaller negative


class ArpeggioReward(RewardComponent):
    """
    Reward proper finger patterns that enable smooth arpeggios.
    
    Encourages using adjacent fingers for adjacent notes (1-2-3-4-5 pattern)
    rather than always using the same finger. This is essential for good
    piano technique and enables faster, smoother playing.
    """
    def __init__(self, good_pattern_bonus: float = 2.0, variety_bonus: float = 1.0):
        self.good_pattern_bonus = good_pattern_bonus
        self.variety_bonus = variety_bonus
    
    def calculate(self, env, action, **kwargs):
        # Skip first step - no previous finger to compare
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        # Get current and previous active fingers
        current_fingers = action["fingers"]
        prev_fingers = env._last_action["fingers"]
        
        current_active = [i for i, f in enumerate(current_fingers) if f == 1]
        prev_active = [i for i, f in enumerate(prev_fingers) if f == 1]
        
        if not current_active or not prev_active:
            return 0.0
        
        # Use the primary (first pressed) finger for comparison
        current_finger = current_active[0]
        prev_finger = prev_active[0]
        
        reward = 0.0
        
        # Reward for using different fingers (variety)
        if current_finger != prev_finger:
            reward += self.variety_bonus
            
            # Extra bonus for adjacent finger pattern (proper arpeggio technique)
            finger_distance = abs(current_finger - prev_finger)
            if finger_distance == 1:
                # Perfect! Using adjacent fingers (e.g., 1->2->3)
                reward += self.good_pattern_bonus
            elif finger_distance == 2:
                # Acceptable skip (e.g., 1->3)
                reward += self.good_pattern_bonus * 0.5
        
        return reward


class NoteProgressReward(RewardComponent):
    """
    Small intermediate reward for successfully advancing to the next note.
    
    This provides a denser learning signal than just completion bonus,
    helping the agent understand that progress is valuable.
    """
    def __init__(self, bonus: float = 0.5):
        self.bonus = bonus
        self._last_step = -1
    
    def calculate(self, env, action, **kwargs):
        # Check if we advanced (this is called before step increment)
        # We'll use a simple heuristic: reward is returned along with success
        # So this component just adds a small bonus on success
        # The actual logic is handled by RewardMixing based on success flag
        return self.bonus  # Only added when accuracy check passes


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
        self.success_components = []  # Components that only apply on success
        
    def add(self, component):
        if isinstance(component, (CompletionReward, NoteProgressReward)):
            self.success_components.append(component)
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
        
        # Only add success-conditional rewards if the step was successful
        if success:
            for comp in self.success_components:
                total_reward += comp.calculate(env, action)
                
        return total_reward, success

