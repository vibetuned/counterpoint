"""
Jacobs 2001 Rules as RL Reward Components.

These wrap the rule functions from counterpoint.rules.jacobs01 as
RewardComponent classes for use with the RewardMixing system.

Each component returns a NEGATIVE value (penalty) when the rule is violated.

For unchanged rules (4, 5, 8, 9, 10, 11, 12), import directly from
parncutt_rewards since the logic is identical.
"""

import numpy as np
from typing import Optional, List, Tuple

from counterpoint.envs.rewards import RewardComponent
from counterpoint.rules.jacobs01 import (
    jacobs_stretch,
    jacobs_small_span,
    jacobs_large_span,
    jacobs_weak_finger,
    jacobs_three_four_five,
    calculate_jacobs_consecutive_cost,
    lattice_span_to_semitones,
    is_playable,
    rule4_position_change_count,
    rule5_position_change_size,
    rule8_three_to_four,
    rule10_thumb_on_black,
    rule11_five_on_black,
)

# Re-export unchanged reward components from parncutt_rewards
from counterpoint.envs.parncutt_rewards import (
    _get_finger_from_action,
    _get_note_from_env,
    PositionChangeCountPenalty,   # Rule 4: unchanged
    PositionChangeSizePenalty,    # Rule 5: unchanged
    ThreeToFourPenalty,           # Rule 8: unchanged
    FourOnBlackPenalty,           # Rule 9: unchanged
    ThumbOnBlackPenalty,          # Rule 10: unchanged
    FiveOnBlackPenalty,           # Rule 11: unchanged
    ThumbPassingPenalty,          # Rule 12: unchanged
)


class JacobsStretchPenalty(RewardComponent):
    """
    Rule 1 + Rule A (Jacobs): Stretch with physical distance mapping.
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
        
        cost = jacobs_stretch(prev_finger, curr_finger, span)
        return -cost * self.weight


class JacobsSmallSpanPenalty(RewardComponent):
    """
    Rule 2 + Rule A (Jacobs): Small-Span with physical distance mapping.
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
        
        cost = jacobs_small_span(prev_finger, curr_finger, span)
        return -cost * self.weight


class JacobsLargeSpanPenalty(RewardComponent):
    """
    Rule C (Jacobs): Unified Large-Span (1x for ALL finger pairs).
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
        
        cost = jacobs_large_span(prev_finger, curr_finger, span)
        return -cost * self.weight


class JacobsWeakFingerPenalty(RewardComponent):
    """
    Rule 6 (Jacobs): Only finger 4 is weak (finger 5 is NOT penalized).
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        curr_finger = _get_finger_from_action(action)
        if curr_finger is None:
            return 0.0
        
        cost = jacobs_weak_finger(curr_finger)
        return -cost * self.weight


class JacobsThreeFourFivePenalty(RewardComponent):
    """
    Rule 7 (Jacobs): DISABLED. Always returns 0.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def calculate(self, env, action, **kwargs) -> float:
        return 0.0  # Disabled per Jacobs


class Jacobs01AllPenalties(RewardComponent):
    """
    Combined penalty using all Jacobs 2001 rules.
    
    Uses Jacobs-modified rules for span (physical distance mapping),
    large-span (unified 1x), weak finger (only 4), and disabled 3-4-5.
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
        
        # Calculate consecutive cost (Jacobs-modified)
        cost = calculate_jacobs_consecutive_cost(
            prev_finger, prev_note, prev_is_black,
            curr_finger, curr_note, curr_is_black
        )
        
        # Add rules 10, 11 with next note context (unchanged from Parncutt)
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
            # Rule 7: DISABLED in Jacobs (no call needed)
        
        return -cost * self.weight
