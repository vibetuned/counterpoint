"""
Jacobs 2001 Refinements to Parncutt 1997 Ergonomic Fingering Rules.

Based on "Refinements to the Ergonomic Model for Keyboard Fingering
of Parncutt, Sloboda, Clarke, Raekallio, and Desain" (2001).

Key changes from Parncutt 1997:
  - Rule A: Physical distance mapping (mm-based bins instead of raw semitone count)
  - Rule C: Unified large-span rule (1 pt/step for ALL finger pairs)
  - Rule 6 mod: Only finger 4 is weak (finger 5 removed)
  - Rule 7 disc: Three-four-five rule disabled
  - Rules 10/11: Kept same as Parncutt (re-justified ergonomically)

All values are adapted for a lattice model where 1 step = 2 semitones.
"""

import math
from typing import Optional, List

# Re-use unchanged rules from Parncutt 1997
from counterpoint.rules.parncutt97 import (
    # Span tables (same in both models)
    FINGER_SPANS,
    MIN_PRAC, MIN_COMF, MIN_REL, MAX_REL, MAX_COMF, MAX_PRAC,
    get_finger_span_limits,
    involves_thumb,
    
    # Unchanged rules (imported directly)
    rule1_stretch,
    rule2_small_span,
    rule4_position_change_count,
    rule5_position_change_size,
    rule8_three_to_four,
    rule9_four_on_black,
    rule10_thumb_on_black,
    rule11_five_on_black,
    rule12_thumb_passing,
    
    # Hand position helper
    _compute_hand_position,
)

# =============================================================================
# PHYSICAL DISTANCE CONSTANTS
# =============================================================================

# Average inter-key distance per semitone on a standard keyboard
# Octave width = 165 mm, 12 semitones → ~13.75 mm per semitone
SEMITONE_DISTANCE_MM = 13.75

# In our lattice model, 1 step = 2 semitones
LATTICE_STEP_MM = SEMITONE_DISTANCE_MM * 2  # ~27.5 mm per lattice step


# =============================================================================
# RULE A: PHYSICAL DISTANCE MAPPING
# Convert physical key distance to effective semitone bin index.
#
# Jacobs' approach:
# 1. Measure physical distance D (mm) between two key centers
# 2. Define reference distances: d_i = SEMITONE_DISTANCE_MM * i (for semitone i)
# 3. Create bin boundaries: B_lower(i) = (d_i + d_{i-1}) / 2
#                           B_upper(i) = (d_{i+1} + d_i) / 2
# 4. If D falls in bin i, the effective span = i semitones
#
# In our lattice:
# - The column difference IS already physical distance (uniform grid)
# - 1 lattice step = 2 semitones = ~27.5 mm
# - We convert lattice span → mm → find semitone bin → convert back to
#   lattice units (÷2) for comparison against FINGER_SPANS table
# =============================================================================

def lattice_span_to_physical_mm(lattice_span: float) -> float:
    """Convert lattice span to physical distance in millimeters."""
    return abs(lattice_span) * LATTICE_STEP_MM


def physical_mm_to_semitone_bin(distance_mm: float) -> int:
    """
    Map a physical distance (mm) to its semitone bin index.
    
    A distance D falls in bin i if:
        (d_{i-1} + d_i) / 2 <= D < (d_i + d_{i+1}) / 2
    where d_i = SEMITONE_DISTANCE_MM * i
    
    This is equivalent to rounding D / SEMITONE_DISTANCE_MM to the nearest integer.
    
    Returns:
        Semitone bin index (non-negative integer)
    """
    if distance_mm < 0:
        distance_mm = abs(distance_mm)
    
    # The bin boundaries are at (i - 0.5) * SEMITONE_DISTANCE_MM
    # So finding the bin is equivalent to rounding
    return round(distance_mm / SEMITONE_DISTANCE_MM)


def physical_distance_to_effective_lattice_span(lattice_span: float) -> float:
    """
    Rule A: Convert lattice span to effective lattice span via physical distance bins.
    
    Pipeline: lattice_span → physical mm → semitone bin → effective lattice span
    
    In our lattice model this is mostly a pass-through since the grid is uniform,
    but we implement it faithfully for correctness.
    
    Args:
        lattice_span: Raw span in lattice steps (signed)
    
    Returns:
        Effective span in lattice steps (unsigned) after physical distance mapping
    """
    physical_mm = lattice_span_to_physical_mm(lattice_span)
    semitone_bin = physical_mm_to_semitone_bin(physical_mm)
    # Convert semitone bin back to lattice units
    return semitone_bin / 2.0


# =============================================================================
# RULE C: UNIFIED LARGE-SPAN RULE
# Jacobs removes the extra penalty for non-thumb pairs.
# 1 point per lattice step exceeding MaxRel for ALL finger pairs.
# =============================================================================

def jacobs_large_span(finger1: int, finger2: int, span: float) -> float:
    """
    Rule C (Jacobs): Unified Large-Span Rule.
    
    Unlike Parncutt (2x for non-thumb pairs), Jacobs uses 1 point per step
    for ALL finger pairs. Jacobs argues stretching non-thumb fingers is not
    inherently more difficult than stretching the thumb.
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in lattice steps (signed)
    
    Returns:
        Cost: 1 point per lattice step above MaxRel (for ALL pairs)
    """
    limits = get_finger_span_limits(finger1, finger2)
    max_rel = limits[MAX_REL]
    
    # Use effective span via physical distance mapping (Rule A)
    effective_span = physical_distance_to_effective_lattice_span(span)
    
    if effective_span > max_rel:
        return 1.0 * (effective_span - max_rel)  # Always 1x, never 2x
    
    return 0.0


# =============================================================================
# RULE 6 (MODIFIED): WEAK-FINGER RULE
# Only finger 4 is penalized. Finger 5 is NOT weak per Jacobs.
# Modern pedagogy recognizes the 5th finger as strong due to specialized
# outer-hand muscles.
# =============================================================================

def jacobs_weak_finger(finger: int) -> float:
    """
    Rule 6 (Jacobs): Modified Weak-Finger Rule.
    
    Only penalizes finger 4. Finger 5 is NOT considered weak.
    
    Args:
        finger: Finger being used (1-5)
    
    Returns:
        Cost: 1 point for finger 4 only
    """
    return 1.0 if finger == 4 else 0.0


# =============================================================================
# RULE 7 (DISABLED): THREE-FOUR-FIVE RULE
# Jacobs disables this rule as redundant with other rules.
# =============================================================================

def jacobs_three_four_five(finger_sequence: List[int]) -> float:
    """
    Rule 7 (Jacobs): DISABLED.
    
    Jacobs argues this rule is redundant since other rules (like 3-to-4
    transitions) already account for the coordination difficulty.
    
    Returns:
        Always 0.0
    """
    return 0.0


# =============================================================================
# JACOBS-MODIFIED STRETCH AND SMALL-SPAN RULES
# These use the physical distance mapping (Rule A) before applying
# the original Parncutt logic.
# =============================================================================

def jacobs_stretch(finger1: int, finger2: int, span: float) -> float:
    """
    Rule 1 with Jacobs' Rule A: Stretch with physical distance mapping.
    
    Same as Parncutt's Rule 1 but uses effective lattice span from
    physical distance bins.
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in lattice steps (signed)
    
    Returns:
        Cost: 2 points per lattice step outside [MinComf, MaxComf]
    """
    limits = get_finger_span_limits(finger1, finger2)
    min_comf, max_comf = limits[MIN_COMF], limits[MAX_COMF]
    
    effective_span = physical_distance_to_effective_lattice_span(span)
    
    if effective_span > max_comf:
        return 2.0 * (effective_span - max_comf)
    elif effective_span < min_comf:
        return 2.0 * (min_comf - effective_span)
    
    return 0.0


def jacobs_small_span(finger1: int, finger2: int, span: float) -> float:
    """
    Rule 2 with Jacobs' Rule A: Small-Span with physical distance mapping.
    
    Same as Parncutt's Rule 2 but uses effective lattice span from
    physical distance bins.
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in lattice steps (signed)
    
    Returns:
        Cost: 1 point per step (thumb) or 2 points per step (non-thumb)
    """
    limits = get_finger_span_limits(finger1, finger2)
    min_rel = limits[MIN_REL]
    
    effective_span = physical_distance_to_effective_lattice_span(span)
    
    if effective_span < min_rel:
        multiplier = 1.0 if involves_thumb(finger1, finger2) else 2.0
        return multiplier * (min_rel - effective_span)
    
    return 0.0


# =============================================================================
# AGGREGATE COST FUNCTIONS
# =============================================================================

def calculate_jacobs_cost(
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool,
    next_finger: Optional[int] = None, next_note: Optional[int] = None, next_is_black: Optional[bool] = None,
    prev_prev_finger: Optional[int] = None, prev_prev_note: Optional[int] = None
) -> float:
    """
    Calculate total Jacobs 2001 cost for a fingering transition.
    
    This aggregates all applicable rules with Jacobs modifications:
    - Rules 1, 2 use physical distance mapping (Rule A)
    - Rule 3 uses unified multiplier (Rule C) 
    - Rule 6 only penalizes finger 4
    - Rule 7 is disabled
    
    Args:
        prev_finger, prev_note, prev_is_black: Previous note state
        curr_finger, curr_note, curr_is_black: Current note state
        next_*: Next note state (optional, for rules 10, 11)
        prev_prev_*: Note before previous (optional, for rules 4, 5, 7)
    
    Returns:
        Total cost as float
    """
    span = curr_note - prev_note  # Signed span in lattice steps
    
    cost = 0.0
    
    # Consecutive span rules with physical distance mapping (1, 2, 3/C)
    cost += jacobs_stretch(prev_finger, curr_finger, span)
    cost += jacobs_small_span(prev_finger, curr_finger, span)
    cost += jacobs_large_span(prev_finger, curr_finger, span)
    
    # Position change rules (4, 5) - require prev_prev context (unchanged)
    if prev_prev_finger is not None and prev_prev_note is not None:
        cost += rule4_position_change_count(
            prev_prev_finger, prev_prev_note,
            prev_finger, prev_note,
            curr_finger, curr_note
        )
        cost += rule5_position_change_size(
            prev_prev_finger, prev_prev_note,
            curr_finger, curr_note
        )
    
    # Finger strength rules
    cost += jacobs_weak_finger(curr_finger)  # Rule 6 mod: only finger 4
    cost += rule8_three_to_four(prev_finger, curr_finger)  # Unchanged
    
    # Rule 7: DISABLED in Jacobs
    # (jacobs_three_four_five always returns 0.0)
    
    # Key color rules (9, 10, 11, 12) - all unchanged
    cost += rule9_four_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)
    cost += rule10_thumb_on_black(
        prev_finger, prev_is_black,
        curr_finger, curr_is_black,
        next_finger, next_is_black
    )
    cost += rule11_five_on_black(
        prev_finger, prev_is_black,
        curr_finger, curr_is_black,
        next_finger, next_is_black
    )
    cost += rule12_thumb_passing(prev_finger, prev_is_black, curr_finger, curr_is_black)
    
    return cost


def calculate_jacobs_consecutive_cost(
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool
) -> float:
    """
    Simplified Jacobs cost for consecutive notes only.
    
    Excludes rules requiring 3-note context (4, 5) and Rule 7 (disabled anyway).
    
    Args:
        prev_finger, prev_note, prev_is_black: Previous note state
        curr_finger, curr_note, curr_is_black: Current note state
    
    Returns:
        Total cost as float
    """
    span = curr_note - prev_note
    
    cost = 0.0
    
    # Span rules with physical distance mapping
    cost += jacobs_stretch(prev_finger, curr_finger, span)
    cost += jacobs_small_span(prev_finger, curr_finger, span)
    cost += jacobs_large_span(prev_finger, curr_finger, span)
    
    # Finger strength (Jacobs modifications)
    cost += jacobs_weak_finger(curr_finger)
    cost += rule8_three_to_four(prev_finger, curr_finger)
    
    # Key color rules (unchanged)
    cost += rule9_four_on_black(prev_finger, prev_is_black, curr_finger, curr_is_black)
    cost += rule10_thumb_on_black(
        prev_finger, prev_is_black,
        curr_finger, curr_is_black
    )
    cost += rule11_five_on_black(
        prev_finger, prev_is_black,
        curr_finger, curr_is_black
    )
    cost += rule12_thumb_passing(prev_finger, prev_is_black, curr_finger, curr_is_black)
    
    return cost
