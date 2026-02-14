"""
Parncutt 1997 Rules as RL Reward Components.

These wrap the rule functions from counterpoint.rules.parncutt97 as
RewardComponent classes for use with the RewardMixing system.

Each component returns a NEGATIVE value (penalty) when the rule is violated.
"""

import numpy as np
from typing import Optional, List, Tuple

from counterpoint.envs.rewards import RewardComponent
from counterpoint.rules.parncutt97 import (
    lattice_span_to_semitones,
    is_playable,
    rule1_stretch,
    rule2_small_span,
    rule3_large_span,
    rule4_position_change_count,
    rule5_position_change_size,
    rule6_weak_finger,
    rule7_three_four_five,
    rule8_three_to_four,
    rule9_four_on_black,
    rule10_thumb_on_black,
    rule11_five_on_black,
    rule12_thumb_passing,
    calculate_consecutive_cost,
)


def _get_finger_from_action(action: dict) -> Optional[int]:
    """Extract active finger (1-5) from action dict."""
    fingers = action.get("fingers", None)
    if fingers is None:
        return None
    indices = np.where(np.array(fingers) == 1)[0]
    if len(indices) == 0:
        return None
    return int(indices[0]) + 1  # Convert 0-4 to 1-5


def _get_is_black_for_finger(action: dict, finger_idx: int) -> bool:
    """Get black key status for a specific finger (0-4 index)."""
    fingers_black = action.get("fingers_black", None)
    if fingers_black is None:
        return False
    return bool(fingers_black[finger_idx])


def _get_note_from_env(env, step: int) -> Optional[Tuple[int, bool]]:
    """Get (note, is_black) for a step in the score."""
    base_env = env.unwrapped
    if step < 0 or step >= len(base_env._score_targets):
        return None
    target_notes = base_env._get_target_notes(step)
    if not target_notes:
        return None
    return target_notes[0]  # (column, is_black)


class StretchPenalty(RewardComponent):
    """
    Rule 1: Stretch Rule.
    Penalty for intervals exceeding MaxComf or below MinComf.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        curr_note, curr_is_black = curr_note_info
        prev_note, prev_is_black = prev_note_info
        span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
        
        cost = rule1_stretch(prev_finger, curr_finger, span)
        return -cost * self.weight


class SmallSpanPenalty(RewardComponent):
    """
    Rule 2: Small-Span Rule.
    Penalty when span is smaller than MinRel (cramped position).
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        curr_note, curr_is_black = curr_note_info
        prev_note, prev_is_black = prev_note_info
        span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
        
        cost = rule2_small_span(prev_finger, curr_finger, span)
        return -cost * self.weight


class LargeSpanPenalty(RewardComponent):
    """
    Rule 3: Large-Span Rule.
    Penalty when span exceeds MaxRel.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        curr_note, curr_is_black = curr_note_info
        prev_note, prev_is_black = prev_note_info
        span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
        
        cost = rule3_large_span(prev_finger, curr_finger, span)
        return -cost * self.weight


class PositionChangeCountPenalty(RewardComponent):
    """
    Rule 4: Position-Change-Count Rule.
    Penalty for changes in hand position over 3 consecutive notes.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self._history: List[Tuple[int, int]] = []  # (finger, note) pairs
    
    def calculate(self, env, action, **kwargs) -> float:
        # Reset history on new episode
        if env._current_step == 0:
            self._history = []
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        curr_note_info = _get_note_from_env(env, env._current_step)
        
        if curr_finger is None or curr_note_info is None:
            return 0.0
        
        curr_note, _ = curr_note_info
        
        # Build history from last action if needed
        if len(self._history) < 2 and env._last_action is not None:
            prev_finger = _get_finger_from_action(env._last_action)
            prev_note_info = _get_note_from_env(env, env._current_step - 1)
            if prev_finger is not None and prev_note_info is not None:
                self._history.append((prev_finger, prev_note_info[0]))
        
        # Add current
        self._history.append((curr_finger, curr_note))
        
        # Keep only last 3
        if len(self._history) > 3:
            self._history = self._history[-3:]
        
        # Need 3 notes for this rule
        if len(self._history) < 3:
            return 0.0
        
        f1, n1 = self._history[-3]
        f2, n2 = self._history[-2]
        f3, n3 = self._history[-1]
        
        cost = rule4_position_change_count(f1, n1, f2, n2, f3, n3)
        return -cost * self.weight


class PositionChangeSizePenalty(RewardComponent):
    """
    Rule 5: Position-Change-Size Rule.
    Penalty based on distance traveled during position change (1st to 3rd note).
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self._history: List[Tuple[int, int]] = []  # (finger, note) pairs
    
    def calculate(self, env, action, **kwargs) -> float:
        # Reset history on new episode
        if env._current_step == 0:
            self._history = []
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        curr_note_info = _get_note_from_env(env, env._current_step)
        
        if curr_finger is None or curr_note_info is None:
            return 0.0
        
        curr_note, _ = curr_note_info
        
        # Build history from last action if needed
        if len(self._history) < 2 and env._last_action is not None:
            prev_finger = _get_finger_from_action(env._last_action)
            prev_note_info = _get_note_from_env(env, env._current_step - 1)
            if prev_finger is not None and prev_note_info is not None:
                self._history.append((prev_finger, prev_note_info[0]))
        
        # Add current
        self._history.append((curr_finger, curr_note))
        
        # Keep only last 3
        if len(self._history) > 3:
            self._history = self._history[-3:]
        
        # Need 3 notes for this rule
        if len(self._history) < 3:
            return 0.0
        
        f1, n1 = self._history[-3]
        f3, n3 = self._history[-1]
        
        cost = rule5_position_change_size(f1, n1, f3, n3)
        return -cost * self.weight


class WeakFingerPenalty(RewardComponent):
    """
    Rule 6: Weak-Finger Rule.
    Penalty for using fingers 4 and 5.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        curr_finger = _get_finger_from_action(action)
        if curr_finger is None:
            return 0.0
        
        cost = rule6_weak_finger(curr_finger)
        return -cost * self.weight


class ThreeFourFivePenalty(RewardComponent):
    """
    Rule 7: Three-Four-Five Rule.
    Penalty for using fingers 3, 4, 5 consecutively (any order).
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self._finger_history: List[int] = []
    
    def calculate(self, env, action, **kwargs) -> float:
        # Reset on new episode
        if env._current_step == 0:
            self._finger_history = []
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        if curr_finger is None:
            return 0.0
        
        # Build history
        if len(self._finger_history) < 2 and env._last_action is not None:
            prev_finger = _get_finger_from_action(env._last_action)
            if prev_finger is not None:
                self._finger_history.append(prev_finger)
        
        self._finger_history.append(curr_finger)
        
        # Keep only last 3
        if len(self._finger_history) > 3:
            self._finger_history = self._finger_history[-3:]
        
        if len(self._finger_history) < 3:
            return 0.0
        
        cost = rule7_three_four_five(self._finger_history[-3:])
        return -cost * self.weight


class ThreeToFourPenalty(RewardComponent):
    """
    Rule 8: Three-to-Four Rule.
    Penalty for following finger 3 with finger 4.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        cost = rule8_three_to_four(prev_finger, curr_finger)
        return -cost * self.weight


class FourOnBlackPenalty(RewardComponent):
    """
    Rule 9: Four-on-Black Rule.
    Penalty for 3W→4B or 4B→3W transitions.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        _, curr_is_black = curr_note_info
        _, prev_is_black = prev_note_info
        
        cost = rule9_four_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)
        return -cost * self.weight


class ThumbOnBlackPenalty(RewardComponent):
    """
    Rule 10: Thumb-on-Black Rule.
    Penalty for placing thumb on black key.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        next_note_info = _get_note_from_env(env, env._current_step + 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        _, curr_is_black = curr_note_info
        _, prev_is_black = prev_note_info
        next_is_black = next_note_info[1] if next_note_info else None
        
        cost = rule10_thumb_on_black(
            prev_finger, prev_is_black,
            curr_finger, curr_is_black,
            next_is_black=next_is_black
        )
        return -cost * self.weight


class FiveOnBlackPenalty(RewardComponent):
    """
    Rule 11: Five-on-Black Rule.
    Penalty for placing pinky on black key when surrounded by white keys.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        next_note_info = _get_note_from_env(env, env._current_step + 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        _, curr_is_black = curr_note_info
        _, prev_is_black = prev_note_info
        next_is_black = next_note_info[1] if next_note_info else None
        
        cost = rule11_five_on_black(
            prev_finger, prev_is_black,
            curr_finger, curr_is_black,
            next_is_black=next_is_black
        )
        return -cost * self.weight


class ThumbPassingPenalty(RewardComponent):
    """
    Rule 12: Thumb-Passing Rule.
    Penalty for thumb passing based on key elevation.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        if env._current_step == 0 or env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        _, curr_is_black = curr_note_info
        _, prev_is_black = prev_note_info
        
        cost = rule12_thumb_passing(prev_finger, prev_is_black, curr_finger, curr_is_black)
        return -cost * self.weight


class Parncutt97AllPenalties(RewardComponent):
    """
    Combined penalty using all Parncutt 1997 rules.
    
    Uses calculate_consecutive_cost for rules that only need 2 notes,
    and tracks history for rules that need 3 notes.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self._history: List[Tuple[int, int, bool]] = []  # (finger, note, is_black)
    
    def calculate(self, env, action, **kwargs) -> float:
        # Reset on new episode
        if env._current_step == 0:
            self._history = []
            return 0.0
        
        if env._last_action is None:
            return 0.0
        
        curr_finger = _get_finger_from_action(action)
        prev_finger = _get_finger_from_action(env._last_action)
        
        if curr_finger is None or prev_finger is None:
            return 0.0
        
        curr_note_info = _get_note_from_env(env, env._current_step)
        prev_note_info = _get_note_from_env(env, env._current_step - 1)
        next_note_info = _get_note_from_env(env, env._current_step + 1)
        
        if curr_note_info is None or prev_note_info is None:
            return 0.0
        
        curr_note, curr_is_black = curr_note_info
        prev_note, prev_is_black = prev_note_info
        next_is_black = next_note_info[1] if next_note_info else None
        
        # Playability filter: moderate penalty for RL (not 1e9)
        span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
        if not is_playable(prev_finger, curr_finger, span):
            return -20.0 * self.weight
        
        # Build history
        if len(self._history) < 2:
            self._history.append((prev_finger, prev_note, prev_is_black))
        self._history.append((curr_finger, curr_note, curr_is_black))
        
        # Keep only last 3
        if len(self._history) > 3:
            self._history = self._history[-3:]
        
        # Calculate consecutive cost
        cost = calculate_consecutive_cost(
            prev_finger, prev_note, prev_is_black,
            curr_finger, curr_note, curr_is_black
        )
        
        # Add rules 10, 11 with next note context
        cost += rule10_thumb_on_black(
            prev_finger, prev_is_black,
            curr_finger, curr_is_black,
            next_is_black=next_is_black
        )
        cost += rule11_five_on_black(
            prev_finger, prev_is_black,
            curr_finger, curr_is_black,
            next_is_black=next_is_black
        )
        
        # Add 3-note rules if we have enough history
        if len(self._history) >= 3:
            f1, n1, _ = self._history[-3]
            f2, n2, _ = self._history[-2]
            f3, n3, _ = self._history[-1]
            
            cost += rule4_position_change_count(f1, n1, f2, n2, f3, n3)
            cost += rule5_position_change_size(f1, n1, f3, n3)
            cost += rule7_three_four_five([f1, f2, f3])
        
        return -cost * self.weight
