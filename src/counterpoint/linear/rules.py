
"""
Rules for the Linear Agent cost function.

Supports two ergonomic models:
  - Parncutt 1997 (default): calculate_transition_cost
  - Jacobs 2001 (refined):   calculate_jacobs_transition_cost

Legacy rules are kept for backwards compatibility.
"""

from counterpoint.rules.parncutt97 import (
    calculate_consecutive_cost,
    lattice_span_to_semitones,
    rule1_stretch,
    rule2_small_span,
    rule3_large_span,
    rule6_weak_finger,
    rule8_three_to_four,
    rule9_four_on_black,
    rule10_thumb_on_black,
    rule11_five_on_black,
    rule12_thumb_passing,
)

from counterpoint.rules.jacobs01 import (
    calculate_jacobs_consecutive_cost,
)


# =============================================================================
# LEGACY RULES (kept for backwards compatibility)
# =============================================================================

def base_distance_cost(prev_note, curr_note):
    """
    Base Distance cost: 1 if note changes, 0 if same note.
    """
    return 1.0 if prev_note != curr_note else 0.0

def rule_5_fourth_finger(curr_finger):
    """
    Rule 5 (legacy): For every use of the fourth finger.
    Now uses Parncutt Rule 6 (weak finger).
    """
    return rule6_weak_finger(curr_finger)

def rule_6_sequential_3_4(prev_finger, curr_finger):
    """
    Rule 6 (legacy): For the use of the third and the fourth finger consecutively.
    Now uses Parncutt Rule 8 (three-to-four).
    """
    return rule8_three_to_four(prev_finger, curr_finger)

def rule_7_white_3_black_4(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 7 (legacy): For 3W→4B transitions.
    Now uses Parncutt Rule 9 (four-on-black).
    """
    return rule9_four_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)

def rule_8_thumb_black(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 8 (legacy): Thumb (1) on Black Key.
    Now uses Parncutt Rule 10 (thumb-on-black).
    """
    return rule10_thumb_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)

def rule_9_pinky_black(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 9 (legacy): Fifth (5) on Black Key.
    Now uses Parncutt Rule 11 (five-on-black).
    """
    return rule11_five_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)

def rule_position_change(prev_finger, prev_note, prev_is_black, curr_finger, curr_note, curr_is_black):
    """
    Rule for position change size (legacy).
    Now computed via span rules in Parncutt module.
    """
    span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
    # Combine stretch and span rules
    cost = rule1_stretch(prev_finger, curr_finger, span)
    cost += rule2_small_span(prev_finger, curr_finger, span)
    cost += rule3_large_span(prev_finger, curr_finger, span)
    return cost


# =============================================================================
# MAIN COST FUNCTIONS
# =============================================================================

def calculate_transition_cost(prev_finger, prev_note, prev_is_black, curr_finger, curr_note, curr_is_black):
    """
    Aggregates all cost rules for a transition using Parncutt 1997 model.
    
    This function uses the shared calculate_consecutive_cost from the
    parncutt97 module, which includes rules 1-3, 6, 8-12.
    
    Note: Rules 4, 5, 7 require 3-note context and are not included here.
    The LinearAgent's Dijkstra search optimizes for consecutive transitions.
    """
    return calculate_consecutive_cost(
        prev_finger, prev_note, prev_is_black,
        curr_finger, curr_note, curr_is_black
    )


def calculate_jacobs_transition_cost(prev_finger, prev_note, prev_is_black, curr_finger, curr_note, curr_is_black):
    """
    Aggregates all cost rules for a transition using Jacobs 2001 model.
    
    Key differences from Parncutt 1997:
    - Span rules use physical distance mapping (Rule A)
    - Large-span uses 1x multiplier for ALL pairs (Rule C)
    - Only finger 4 is considered weak (Rule 6 mod)
    - Three-four-five rule is disabled (Rule 7 disc)
    
    Note: Rules 4, 5 require 3-note context and are not included here.
    """
    return calculate_jacobs_consecutive_cost(
        prev_finger, prev_note, prev_is_black,
        curr_finger, curr_note, curr_is_black
    )

