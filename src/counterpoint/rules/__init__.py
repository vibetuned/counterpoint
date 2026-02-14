"""
Counterpoint Rules Module.

Contains implementations of ergonomic fingering rules for piano.
Supports two models:
  - Parncutt 1997 (original)
  - Jacobs 2001 (refinements)
"""

from counterpoint.rules.parncutt97 import (
    # Finger span utilities
    get_finger_span_limits,
    involves_thumb,
    FINGER_SPANS,
    UNPLAYABLE_COST,
    is_playable,
    
    # Lattice-to-semitone conversion
    lattice_to_semitone,
    lattice_span_to_semitones,
    
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

from counterpoint.rules.jacobs01 import (
    # Physical distance mapping
    physical_distance_to_effective_semitone_span,
    
    # Jacobs-specific rules
    jacobs_stretch,
    jacobs_small_span,
    jacobs_large_span,
    jacobs_weak_finger,
    jacobs_three_four_five,
    
    # Aggregate functions
    calculate_jacobs_cost,
    calculate_jacobs_consecutive_cost,
)

__all__ = [
    'get_finger_span_limits',
    'involves_thumb',
    'FINGER_SPANS',
    'UNPLAYABLE_COST',
    'is_playable',
    'lattice_to_semitone',
    'lattice_span_to_semitones',
    
    # Parncutt 1997 rules
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
    
    # Jacobs 2001 rules
    'physical_distance_to_effective_semitone_span',
    'jacobs_stretch',
    'jacobs_small_span',
    'jacobs_large_span',
    'jacobs_weak_finger',
    'jacobs_three_four_five',
    'calculate_jacobs_cost',
    'calculate_jacobs_consecutive_cost',
]

