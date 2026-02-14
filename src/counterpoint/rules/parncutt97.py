"""
Parncutt 1997 Ergonomic Fingering Rules.

Based on "An Ergonomic Model of Keyboard Fingering for Melodic Fragments" (1997).
All span values are in semitones, matching the original paper.
"""

from typing import Tuple, Optional, List

# Cost returned for fingerings outside practical span limits (unplayable).
# Large enough that Dijkstra / shortest-path agents will never select these
# edges, but finite so the graph stays well-defined.
UNPLAYABLE_COST = 1e9

# =============================================================================
# FINGER SPAN TABLES
# Values are in semitones, matching the original paper exactly.
# =============================================================================

# Finger pair key: (min_finger, max_finger), e.g., (1, 2) means thumb-to-index
# Values: (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac) in semitones
FINGER_SPANS = {
    (1, 2): (-5, -3,  1,  5,  8, 10),
    (1, 3): (-4, -2,  3,  7, 10, 12),
    (1, 4): (-3, -1,  5,  9, 12, 14),
    (1, 5): (-1,  1,  7, 11, 13, 15),
    (2, 3): ( 1,  1,  1,  2,  3,  5),
    (2, 4): ( 1,  1,  3,  4,  5,  7),
    (2, 5): ( 2,  2,  5,  6,  8, 10),
    (3, 4): ( 1,  1,  1,  2,  2,  4),
    (3, 5): ( 1,  1,  3,  4,  5,  7),
    (4, 5): ( 1,  1,  1,  2,  3,  5),
}

# Indices for FINGER_SPANS tuple
MIN_PRAC, MIN_COMF, MIN_REL, MAX_REL, MAX_COMF, MAX_PRAC = range(6)


def get_finger_span_limits(finger1: int, finger2: int) -> Tuple[int, int, int, int, int, int]:
    """
    Get span limits for a finger pair.
    
    Args:
        finger1: First finger (1-5, where 1=thumb)
        finger2: Second finger (1-5)
    
    Returns:
        Tuple of (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac) in semitones.
        Returns (0,0,0,0,0,0) if invalid finger pair.
    """
    key = (min(finger1, finger2), max(finger1, finger2))
    return FINGER_SPANS.get(key, (0, 0, 0, 0, 0, 0))


def is_playable(prev_finger: int, curr_finger: int, span: int, hand: int = 1) -> bool:
    """
    Check if a finger transition is physically playable (within practical span).
    
    The FINGER_SPANS table assumes ascending finger order (lower-numbered
    finger first) with positive spans meaning rightward movement on the
    keyboard.  When the higher-numbered finger is played *first* (descending
    finger order), the practical bounds are negated and swapped so that the
    direction is correctly accounted for.
    
    For the Left Hand (hand=2), the span is negated before checking because
    the spatial finger-to-keyboard mapping is mirrored: ascending keyboard
    movement uses descending finger numbers (5→4→3→2→1).
    
    For the same finger on the same note (span=0), always returns True.
    
    Args:
        prev_finger: Previous finger (1-5)
        curr_finger: Current finger (1-5)
        span: Signed semitone distance (positive = ascending on keyboard)
        hand: 1=RH, 2=LH
    
    Returns:
        True if the transition is within practical limits.
    """
    if prev_finger == curr_finger:
        return True  # Same finger on same note is always valid
    
    # LH mirror: ascending keyboard = descending fingers, so negate span
    # to match the RH-oriented table.
    if hand == 2:
        span = -span
    
    limits = get_finger_span_limits(prev_finger, curr_finger)
    min_prac = limits[MIN_PRAC]
    max_prac = limits[MAX_PRAC]
    
    # The table is keyed (min_finger, max_finger) and assumes that the
    # lower-numbered finger is to the LEFT (ascending order).  If the
    # higher-numbered finger was played first, the expected direction
    # is reversed, so we negate and swap the bounds.
    if prev_finger > curr_finger:
        actual_min, actual_max = -max_prac, -min_prac
    else:
        actual_min, actual_max = min_prac, max_prac
    
    return actual_min <= span <= actual_max


def involves_thumb(finger1: int, finger2: int) -> bool:
    """Check if the finger pair involves the thumb."""
    return finger1 == 1 or finger2 == 1


# =============================================================================
# LATTICE-TO-SEMITONE CONVERSION
# The piano environment uses a 52×2 lattice (columns = white keys, row 1 = sharps).
# These functions convert lattice coordinates to absolute semitone values.
# =============================================================================

# Semitone offset of each white key within one octave (7 columns → 12 semitones)
# Col 0=C(0), 1=D(2), 2=E(4), 3=F(5), 4=G(7), 5=A(9), 6=B(11)
_WHITE_KEY_SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def lattice_to_semitone(column: int, is_black: bool) -> int:
    """
    Convert a lattice position to an absolute semitone number.
    
    Args:
        column: White key column index (0–51 in the 52-column lattice)
        is_black: Whether the note is on the accidental row (sharp of that column)
    
    Returns:
        Absolute semitone number (MIDI-like, starting from 0 = C0)
    """
    octave = column // 7
    key_in_octave = column % 7
    semitone = octave * 12 + _WHITE_KEY_SEMITONES[key_in_octave]
    if is_black:
        semitone += 1  # sharp of the white key at that column
    return semitone


def lattice_span_to_semitones(
    col1: int, is_black1: bool,
    col2: int, is_black2: bool
) -> int:
    """
    Compute the signed semitone distance between two lattice positions.
    
    Args:
        col1, is_black1: First note (previous)
        col2, is_black2: Second note (current)
    
    Returns:
        Signed semitone distance (positive = ascending)
    """
    return lattice_to_semitone(col2, is_black2) - lattice_to_semitone(col1, is_black1)


# =============================================================================
# RULE 1: STRETCH RULE
# Penalty for intervals exceeding MaxComf or below MinComf.
# 2 points per semitone outside the range.
# =============================================================================

def rule1_stretch(finger1: int, finger2: int, span: float) -> float:
    """
    Rule 1: Stretch Rule.
    
    Penalty for intervals that exceed the maximum comfortable span (MaxComf)
    or are less than the minimum comfortable span (MinComf).
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in semitones (signed: positive = ascending)
    
    Returns:
        Cost: 2 points per semitone outside [MinComf, MaxComf]
    """
    limits = get_finger_span_limits(finger1, finger2)
    min_comf, max_comf = limits[MIN_COMF], limits[MAX_COMF]
    
    # Use absolute span for comparison
    abs_span = abs(span)
    
    if abs_span > max_comf:
        return 2.0 * (abs_span - max_comf)
    elif abs_span < min_comf:
        return 2.0 * (min_comf - abs_span)
    
    return 0.0


# =============================================================================
# RULE 2: SMALL-SPAN RULE
# Penalty when span is smaller than MinRel (cramped position).
# 1 point per step if thumb involved, 2 points otherwise.
# =============================================================================

def rule2_small_span(finger1: int, finger2: int, span: float) -> float:
    """
    Rule 2: Small-Span Rule.
    
    Penalty when the span between two fingers is smaller than a relaxed span (MinRel).
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in semitones (signed)
    
    Returns:
        Cost: 1 point per semitone (thumb) or 2 points per semitone (non-thumb)
    """
    limits = get_finger_span_limits(finger1, finger2)
    min_rel = limits[MIN_REL]
    
    abs_span = abs(span)
    
    if abs_span < min_rel:
        multiplier = 1.0 if involves_thumb(finger1, finger2) else 2.0
        return multiplier * (min_rel - abs_span)
    
    return 0.0


# =============================================================================
# RULE 3: LARGE-SPAN RULE
# Penalty when span exceeds MaxRel.
# 1 point per step if thumb involved, 2 points otherwise.
# =============================================================================

def rule3_large_span(finger1: int, finger2: int, span: float) -> float:
    """
    Rule 3: Large-Span Rule.
    
    Penalty when spans exceed the maximum relaxed span (MaxRel).
    
    Args:
        finger1: Previous finger (1-5)
        finger2: Current finger (1-5)
        span: Distance in semitones (signed)
    
    Returns:
        Cost: 1 point per semitone (thumb) or 2 points per semitone (non-thumb)
    """
    limits = get_finger_span_limits(finger1, finger2)
    max_rel = limits[MAX_REL]
    
    abs_span = abs(span)
    
    if abs_span > max_rel:
        multiplier = 1.0 if involves_thumb(finger1, finger2) else 2.0
        return multiplier * (abs_span - max_rel)
    
    return 0.0


# =============================================================================
# RULE 4: POSITION-CHANGE-COUNT RULE
# Penalty for changes in hand position.
# 2 points for "full" change, 1 point for "half" change.
# =============================================================================

def _compute_hand_position(finger: int, note: int, hand: int = 1) -> float:
    """Compute hand position from finger and note.
    
    RH: thumb (1) is leftmost → pos = note - (finger - 1)
    LH: pinky (5) is leftmost → pos = note - (5 - finger)
    """
    if hand == 2:  # LH
        return note - (5 - finger)
    return note - (finger - 1)


def rule4_position_change_count(
    finger1: int, note1: int,
    finger2: int, note2: int,
    finger3: int, note3: int,
    hand: int = 1
) -> float:
    """
    Rule 4: Position-Change-Count Rule.
    
    Penalty for changes in hand position, favoring fingerings that stay
    within a single position. Requires three consecutive notes.
    
    A "full" change occurs when the hand moves between note 1→2 AND 2→3.
    A "half" change occurs when the hand moves only between 1→2 OR 2→3.
    
    Args:
        finger1, note1: First note fingering
        finger2, note2: Second note fingering  
        finger3, note3: Third note fingering
        hand: 1=RH, 2=LH
    
    Returns:
        Cost: 2 points for full change, 1 point for half change
    """
    pos1 = _compute_hand_position(finger1, note1, hand)
    pos2 = _compute_hand_position(finger2, note2, hand)
    pos3 = _compute_hand_position(finger3, note3, hand)
    
    change_1_to_2 = pos1 != pos2
    change_2_to_3 = pos2 != pos3
    
    if change_1_to_2 and change_2_to_3:
        return 2.0  # Full change
    elif change_1_to_2 or change_2_to_3:
        return 1.0  # Half change
    
    return 0.0


# =============================================================================
# RULE 5: POSITION-CHANGE-SIZE RULE
# Penalty based on distance traveled during position change.
# 1 point per step that interval between 1st and 3rd notes falls outside [MinComf, MaxComf].
# =============================================================================

def rule5_position_change_size(
    finger1: int, note1: int,
    finger3: int, note3: int
) -> float:
    """
    Rule 5: Position-Change-Size Rule.
    
    Penalty based on the physical distance the hand must travel during a position change.
    Applies to next-to-consecutive notes (1st and 3rd).
    
    Args:
        finger1, note1: First note fingering
        finger3, note3: Third note fingering
    
    Returns:
        Cost: 1 point per step outside [MinComf, MaxComf] range
    """
    span = abs(note3 - note1)
    limits = get_finger_span_limits(finger1, finger3)
    min_comf, max_comf = limits[MIN_COMF], limits[MAX_COMF]
    
    if span > max_comf:
        return span - max_comf
    elif span < min_comf:
        return min_comf - span
    
    return 0.0


# =============================================================================
# RULE 6: WEAK-FINGER RULE
# Penalty for using fingers 4 and 5.
# 1 point per use.
# =============================================================================

def rule6_weak_finger(finger: int) -> float:
    """
    Rule 6: Weak-Finger Rule.
    
    Penalty for using fingers considered less strong and agile (fingers 4 and 5).
    
    Args:
        finger: Finger being used (1-5)
    
    Returns:
        Cost: 1 point for finger 4 or 5
    """
    return 1.0 if finger in (4, 5) else 0.0


# =============================================================================
# RULE 7: THREE-FOUR-FIVE RULE
# Penalty for using fingers 3, 4, 5 consecutively (any order).
# 1 point per occurrence.
# =============================================================================

def rule7_three_four_five(finger_sequence: List[int]) -> float:
    """
    Rule 7: Three-Four-Five Rule.
    
    Penalty for using fingers 3, 4, and 5 consecutively, which are difficult
    to coordinate on the weak side of the hand.
    
    Args:
        finger_sequence: List of 3 consecutive fingers used
    
    Returns:
        Cost: 1 point if sequence contains {3, 4, 5} in any order
    """
    if len(finger_sequence) < 3:
        return 0.0
    
    finger_set = set(finger_sequence[:3])
    if finger_set == {3, 4, 5}:
        return 1.0
    
    return 0.0


# =============================================================================
# RULE 8: THREE-TO-FOUR RULE
# Penalty for following finger 3 with finger 4.
# 1 point per occurrence.
# =============================================================================

def rule8_three_to_four(prev_finger: int, curr_finger: int) -> float:
    """
    Rule 8: Three-to-Four Rule.
    
    Penalty for following finger 3 immediately with finger 4,
    a difficult transition due to tendon limitations.
    
    Args:
        prev_finger: Previous finger (1-5)
        curr_finger: Current finger (1-5)
    
    Returns:
        Cost: 1 point for 3→4 transition
    """
    if prev_finger == 3 and curr_finger == 4:
        return 1.0
    return 0.0


# =============================================================================
# RULE 9: FOUR-ON-BLACK RULE
# Penalty for 3-4 transition where 3 is on white and 4 is on black (or reverse).
# 1 point per occurrence.
# =============================================================================

def rule9_four_on_black(
    prev_finger: int, prev_is_black: bool,
    curr_finger: int, curr_is_black: bool
) -> float:
    """
    Rule 9: Four-on-Black Rule.
    
    Penalty for a consecutive transition between fingers 3 and 4 where
    finger 3 is on white and finger 4 is on black.
    
    Args:
        prev_finger: Previous finger (1-5)
        prev_is_black: Whether previous note is black key
        curr_finger: Current finger (1-5)
        curr_is_black: Whether current note is black key
    
    Returns:
        Cost: 1 point for 3W→4B or 4B→3W transitions
    """
    # 3 on white, 4 on black
    is_3w_4b = (prev_finger == 3 and not prev_is_black and 
                curr_finger == 4 and curr_is_black)
    
    # 4 on black, 3 on white
    is_4b_3w = (prev_finger == 4 and prev_is_black and 
                curr_finger == 3 and not curr_is_black)
    
    return 1.0 if is_3w_4b or is_4b_3w else 0.0


# =============================================================================
# RULE 10: THUMB-ON-BLACK RULE
# Penalty for placing thumb on black key.
# 1 base point + 2 if preceding is white + 2 if following is white.
# =============================================================================

def rule10_thumb_on_black(
    prev_finger: int, prev_is_black: bool,
    curr_finger: int, curr_is_black: bool,
    next_finger: Optional[int] = None, next_is_black: Optional[bool] = None
) -> float:
    """
    Rule 10: Thumb-on-Black Rule.
    
    Penalty for placing the thumb on a black key, which can displace
    the hand from a comfortable position.
    
    Args:
        prev_finger: Previous finger (1-5)
        prev_is_black: Whether previous note is black key
        curr_finger: Current finger (1-5)
        curr_is_black: Whether current note is black key
        next_finger: Next finger (1-5), if known
        next_is_black: Whether next note is black key, if known
    
    Returns:
        Cost: 1 base + 2 if prev is white + 2 if next is white
    """
    if curr_finger != 1 or not curr_is_black:
        return 0.0
    
    cost = 1.0  # Base point for thumb on black
    
    if not prev_is_black:
        cost += 2.0  # Previous note is white
    
    if next_is_black is not None and not next_is_black:
        cost += 2.0  # Next note is white
    
    return cost


# =============================================================================
# RULE 11: FIVE-ON-BLACK RULE
# Penalty for placing pinky on black key when surrounded by white keys.
# 2 if preceding is white + 2 if following is white.
# =============================================================================

def rule11_five_on_black(
    prev_finger: int, prev_is_black: bool,
    curr_finger: int, curr_is_black: bool,
    next_finger: Optional[int] = None, next_is_black: Optional[bool] = None
) -> float:
    """
    Rule 11: Five-on-Black Rule.
    
    Penalty for placing the little finger on a black key when
    surrounded by white keys.
    
    Args:
        prev_finger: Previous finger (1-5)
        prev_is_black: Whether previous note is black key
        curr_finger: Current finger (1-5)
        curr_is_black: Whether current note is black key
        next_finger: Next finger (1-5), if known
        next_is_black: Whether next note is black key, if known
    
    Returns:
        Cost: 2 if prev is white + 2 if next is white (0 if both neighbors are black)
    """
    if curr_finger != 5 or not curr_is_black:
        return 0.0
    
    cost = 0.0
    
    if not prev_is_black:
        cost += 2.0  # Previous note is white
    
    if next_is_black is not None and not next_is_black:
        cost += 2.0  # Next note is white
    
    return cost


# =============================================================================
# RULE 12: THUMB-PASSING RULE
# Penalty based on difficulty of passing thumb under/over another finger.
# 1 point for same level, 3 points for white-to-black passes involving thumb.
# =============================================================================

def rule12_thumb_passing(
    prev_finger: int, prev_is_black: bool,
    curr_finger: int, curr_is_black: bool
) -> float:
    """
    Rule 12: Thumb-Passing Rule.
    
    Penalty based on the difficulty of passing the thumb under/over
    another finger relative to key elevation.
    
    Args:
        prev_finger: Previous finger (1-5)
        prev_is_black: Whether previous note is black key
        curr_finger: Current finger (1-5)
        curr_is_black: Whether current note is black key
    
    Returns:
        Cost: 1 point for same level, 3 points for white-to-black thumb passes
    """
    # Check if this is a thumb pass (thumb to non-thumb or vice versa)
    is_thumb_pass = ((prev_finger == 1 and curr_finger != 1) or 
                     (prev_finger != 1 and curr_finger == 1))
    
    if not is_thumb_pass:
        return 0.0
    
    # Same key color (same level)
    if prev_is_black == curr_is_black:
        return 1.0
    
    # Different levels (white-to-black or black-to-white)
    return 3.0


# =============================================================================
# AGGREGATE COST FUNCTION
# =============================================================================

def calculate_parncutt_cost(
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool,
    next_finger: Optional[int] = None, next_note: Optional[int] = None, next_is_black: Optional[bool] = None,
    prev_prev_finger: Optional[int] = None, prev_prev_note: Optional[int] = None,
    hand: int = 1
) -> float:
    """
    Calculate total Parncutt cost for a fingering transition.
    
    This aggregates all applicable rules for a single transition.
    Some rules (4, 5, 7) require context from neighboring notes.
    
    Args:
        prev_finger, prev_note, prev_is_black: Previous note state
        curr_finger, curr_note, curr_is_black: Current note state
        next_*: Next note state (optional, for rules 10, 11)
        prev_prev_*: Note before previous (optional, for rules 4, 5, 7)
    
    Returns:
        Total cost as float
    """
    span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
    
    # Enumeration filter: reject fingerings outside practical span limits
    if not is_playable(prev_finger, curr_finger, span, hand):
        return UNPLAYABLE_COST
    
    cost = 0.0
    
    # Consecutive span rules (1, 2, 3)
    cost += rule1_stretch(prev_finger, curr_finger, span)
    cost += rule2_small_span(prev_finger, curr_finger, span)
    cost += rule3_large_span(prev_finger, curr_finger, span)
    
    # Position change rules (4, 5) - require prev_prev context
    if prev_prev_finger is not None and prev_prev_note is not None:
        cost += rule4_position_change_count(
            prev_prev_finger, prev_prev_note,
            prev_finger, prev_note,
            curr_finger, curr_note,
            hand=hand
        )
        cost += rule5_position_change_size(
            prev_prev_finger, prev_prev_note,
            curr_finger, curr_note
        )
    
    # Finger strength rules (6, 8)
    cost += rule6_weak_finger(curr_finger)
    cost += rule8_three_to_four(prev_finger, curr_finger)
    
    # Rule 7 (345 sequence) - require prev_prev context
    if prev_prev_finger is not None:
        cost += rule7_three_four_five([prev_prev_finger, prev_finger, curr_finger])
    
    # Key color rules (9, 10, 11, 12)
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


def calculate_consecutive_cost(
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool
) -> float:
    """
    Simplified cost calculation for consecutive notes only.
    
    This excludes rules that require context beyond two notes:
    - Rules 4, 5 (position change - need 3 notes)
    - Rule 7 (345 sequence - need 3 notes)
    - Rules 10, 11 (thumb/five on black - need next note for full accuracy)
    
    Args:
        prev_finger, prev_note, prev_is_black: Previous note state
        curr_finger, curr_note, curr_is_black: Current note state
    
    Returns:
        Total cost as float
    """
    span = lattice_span_to_semitones(prev_note, prev_is_black, curr_note, curr_is_black)
    
    # Enumeration filter: reject fingerings outside practical span limits
    if not is_playable(prev_finger, curr_finger, span):
        return UNPLAYABLE_COST
    
    cost = 0.0
    
    # Consecutive span rules
    cost += rule1_stretch(prev_finger, curr_finger, span)
    cost += rule2_small_span(prev_finger, curr_finger, span)
    cost += rule3_large_span(prev_finger, curr_finger, span)
    
    # Finger strength rules (per-note and pair)
    cost += rule6_weak_finger(curr_finger)
    cost += rule8_three_to_four(prev_finger, curr_finger)
    
    # Key color rules (pair-based)
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
