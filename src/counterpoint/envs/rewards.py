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
            return -1.5 * dist
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

class FingerNotRepetitionReward(RewardComponent):
    """
    Reward using different fingers for different notes.
    
    This encourages proper finger alternation which is essential for
    good piano technique.
    """
    def __init__(self, reward: float = 2.0):
        self.reward = reward
    
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
        
        return np.sum(np.bitwise_xor(current_fingers, prev_fingers)) * self.reward

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
    Reward proper finger patterns that match melodic direction.
    
    Proper piano technique requires:
    - Ascending notes → fingers should progress upward (1→2→3→4→5)
    - Descending notes → fingers should progress downward (5→4→3→2→1)
    
    This prevents reward hacking with simple 2-finger alternation.
    """
    def __init__(self, direction_match_bonus: float = 2.0, variety_bonus: float = 2.0):
        self.direction_match_bonus = direction_match_bonus
        self.variety_bonus = variety_bonus
    
    def calculate(self, env, action, **kwargs):
        # Skip first step - no previous note/finger to compare
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        # Need at least 2 notes in score to compare
        if env._current_step >= len(env._score_targets) or env._current_step < 1:
            return 0.0
        
        # Get note direction
        current_note, _ = env._score_targets[env._current_step]
        prev_note, _ = env._score_targets[env._current_step - 1]
        note_direction = current_note - prev_note  # positive = ascending
        
        # Same note - no arpeggio pattern expected
        if note_direction == 0:
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
        finger_direction = current_finger - prev_finger  # positive = higher finger
        
        reward = 0.0
        
        # Small reward for using different fingers (prevents single-finger spam)
        if current_finger != prev_finger:
            reward += self.variety_bonus
        
        # Main reward: finger direction matches note direction
        # Ascending notes should use ascending fingers, descending should use descending
        directions_match = (note_direction > 0 and finger_direction > 0) or \
                          (note_direction < 0 and finger_direction < 0)
        
        if directions_match:
            finger_distance = abs(finger_direction)
            if finger_distance == 1:
                # Perfect! Adjacent finger in correct direction
                reward += self.direction_match_bonus
            elif finger_distance == 2:
                # Acceptable skip in correct direction
                reward += self.direction_match_bonus * 0.5
            else:
                # Large jump but still correct direction
                reward += self.direction_match_bonus * 0.25
        
        return reward

class ArpeggioReward2(RewardComponent):
    """
    Reward proper finger patterns that match melodic direction.
    
    Proper piano technique requires:
    - Ascending notes → fingers should progress upward (1→2→3→4→5)
    - Descending notes → fingers should progress downward (5→4→3→2→1)
    
    This prevents reward hacking with simple 2-finger alternation.
    """
    def __init__(self, direction_match_bonus: float = 2.0, variety_bonus: float = 2.0):
        self.direction_match_bonus = direction_match_bonus
        self.variety_bonus = variety_bonus
    
    def calculate(self, env, action, **kwargs):
        # Skip first step - no previous note/finger to compare
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        # Need at least 2 notes in score to compare
        if env._current_step >= len(env._score_targets) or env._current_step < 1:
            return 0.0
        
        # Get note direction
        current_note, _ = env._score_targets[env._current_step]
        prev_note, _ = env._score_targets[env._current_step - 1]
        note_direction = current_note - prev_note  # positive = ascending
        
        # Same note - no arpeggio pattern expected
        if note_direction == 0:
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
        finger_direction = current_finger - prev_finger  # positive = higher finger
        
        reward = 0.0
        
        # Small reward for using different fingers (prevents single-finger spam)
        if current_finger != prev_finger:
            reward += self.variety_bonus
        
        # Main reward: finger direction matches note direction
        # Ascending notes should use ascending fingers, descending should use descending
        directions_match = (note_direction > 0 and finger_direction > 0) or \
                          (note_direction < 0 and finger_direction < 0)
        
        if directions_match:
            finger_distance = abs(finger_direction)
            if finger_distance == 1:
                # Perfect! Adjacent finger in correct direction
                reward += self.direction_match_bonus
            elif finger_distance == 2:
                # Acceptable skip in correct direction
                reward += self.direction_match_bonus * 0.5
            else:
                # Large jump but still correct direction
                reward += self.direction_match_bonus * 0.25
        
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
        current_fingers = action["fingers"]
        
        current_active = [i for i, f in enumerate(current_fingers) if f == 1]

        if len(current_active) < 1:
            return -5.0

        
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
        # These rewards only apply when the agent successfully plays the correct note
        self.components.append(component)
        #if isinstance(component, (CompletionReward, NoteProgressReward, ArpeggioReward)):
        #    self.success_components.append(component)
        #else:
        #    self.components.append(component)
        
    def calculate(self, env, action):
        total_reward = 0.0
        success = False
        
        # Calculate standard rewards
        for comp in self.components:
            res = comp.calculate(env, action)
            #if isinstance(res, tuple):
            #    rew, succ = res
            #    total_reward += rew
            #    if succ:
            #        success = True
            #else:
            total_reward += res
        
        # Only add success-conditional rewards if the step was successful
        #if success:
        #    for comp in self.success_components:
        #        total_reward += comp.calculate(env, action)
                
        return total_reward, True

