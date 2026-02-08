"""
Counterpoint Rules Module.

Contains implementations of ergonomic fingering rules for piano.
"""

from counterpoint.rules.parncutt97 import (
    # Finger span utilities
    get_finger_span_limits,
    involves_thumb,
    FINGER_SPANS,
    
    # Individual rules
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
    
    # Aggregate functions
    calculate_parncutt_cost,
    calculate_consecutive_cost,
)

__all__ = [
    'get_finger_span_limits',
    'involves_thumb',
    'FINGER_SPANS',
    'rule1_stretch',
    'rule2_small_span',
    'rule3_large_span',
    'rule4_position_change_count',
    'rule5_position_change_size',
    'rule6_weak_finger',
    'rule7_three_four_five',
    'rule8_three_to_four',
    'rule9_four_on_black',
    'rule10_thumb_on_black',
    'rule11_five_on_black',
    'rule12_thumb_passing',
    'calculate_parncutt_cost',
    'calculate_consecutive_cost',
]
