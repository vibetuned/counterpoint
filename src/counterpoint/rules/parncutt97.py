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

def normalize_span(finger1: int, finger2: int, span: int) -> int:
    """
    Normalize span to be relative to ascending finger order (low->high).
    
    If finger1 < finger2 (e.g. 1->2), returns span as is.
    If finger1 > finger2 (e.g. 2->1), returns -span (equivalent to low(2)->high(1)? No, low(1)->high(2)).
      Distance 2->1 is X. Distance 1->2 is -X.
      So if we have transition 2->1 with span +1.
      We want to compare to table (1,2).
      Table assumes 1->2.
      So we effectively want the distance 1->2.
      Distance(1->2) = -Distance(2->1) = -1.
    """
    if finger1 > finger2:
        return -span
    return span


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
    
    norm_span = normalize_span(finger1, finger2, span)
    
    if norm_span > max_comf:
        return 2.0 * (norm_span - max_comf)
    elif norm_span < min_comf:
        return 2.0 * (min_comf - norm_span)
    
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
    
    norm_span = normalize_span(finger1, finger2, span)
    
    if norm_span < min_rel:
        multiplier = 1.0 if involves_thumb(finger1, finger2) else 2.0
        return multiplier * (min_rel - norm_span)
    
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
    
    norm_span = normalize_span(finger1, finger2, span)
    
    if norm_span > max_rel:
        multiplier = 1.0 if involves_thumb(finger1, finger2) else 2.0
        return multiplier * (norm_span - max_rel)
    
    return 0.0


# =============================================================================
# RULE 4: POSITION-CHANGE-COUNT RULE
# Penalty for changes in hand position.
# 2 points for "full" change, 1 point for "half" change.
# =============================================================================

def _semitone_to_diatonic_step(note: int) -> int:
    """
    Convert absolute semitone to diatonic step index (0-6).
    Assumes C Major / A Minor context for white keys.
    C/C# -> 0
    D/D# -> 1
    E    -> 2
    F/F# -> 3
    G/G# -> 4
    A/A# -> 5
    B    -> 6
    """
    # Pitch class 0..11
    pc = note % 12
    # Map map
    _map = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2,
        5: 3, 6: 3,
        7: 4, 8: 4,
        9: 5, 10: 5,
        11: 6
    }
    octave = note // 12
    return octave * 7 + _map.get(pc, 0)

def _compute_hand_position(finger: int, note: int, hand: int = 1) -> float:
    """Compute hand position from finger and note (Diatonic)."""
    step = _semitone_to_diatonic_step(note)
    
    # Fingers 1..5 map to relative steps 0, 1, 2, 3, 4
    # (Thumb is 0, Index 1, etc.)
    # Pos = Step - (Finger - 1)
    
    if hand == 2: # LH
        # Mirror: 5 is 0. 1 is 4.
        # 5->0, 4->1, 3->2, 2->3, 1->4
        f_shift = 5 - finger
        return step - f_shift
        
    f_shift = finger - 1
    return step - f_shift


def _get_valid_thumb_positions(finger: int, note_semi: int, is_black: bool) -> set:
    """
    Get the set of valid thumb positions (in semitones) that allow the
    given finger to play the note within RELAXED span limits.
    
    If the non-thumb finger plays a Black Key, we apply a heuristic bonus
    to the relaxed span (widening the valid range), as the hand can reach
    further/closer due to the key elevation and forward position.
    """
    if finger == 1:
        return {note_semi}
        
    limits = get_finger_span_limits(1, finger)
    min_rel = limits[MIN_REL]
    max_rel = limits[MAX_REL]
    
    # Heuristic: If finger is on Black Key, relax constraints significantly.
    # Parncutt suggests "plenty of room" for Thumb-White / Other-Black.
    # We relax MinRel by up to 3 semitones (e.g. 1-5 MinRel 7 -> 4).
    if is_black:
        min_rel = max(0, min_rel - 3)
        max_rel += 1
    
    # Valid Thumbs = Note - [MinRel...MaxRel]
    return set(range(note_semi - max_rel, note_semi - min_rel + 1))

# ... (Rule 4 logic remains, just updating helper above)

# =============================================================================
# RULE 7: THREE-FOUR-FIVE RULE
# Penalty for using fingers 3, 4, 5 consecutively (any order).
# 1 point per occurrence.
# =============================================================================

def rule7_three_four_five(finger_sequence: List[int]) -> float:
    """
    Rule 7: Three-Four-Five Rule.
    
    Checks if fingers 3, 4, and 5 appear within the sequence using two sub-rules:
    1. Contiguous 3-4-5: The last 3 fingers form a permutation of {3,4,5}.
    2. Interrupted 3-4-5: The last 4 fingers contain {3,4,5} interrupted by a strong finger.
       (e.g. 4-2-3-5).
    """
    if len(finger_sequence) < 3:
        return 0.0

    # 1. Check strict contiguous 3-window
    last_three = finger_sequence[-3:]
    weak_3 = [f for f in last_three if f in (3, 4, 5)]
    # Note: Using len(set) == 3 ensures we have distinct 3, 4, 5. 
    # {3,4,5} must be subset of checked fingers.
    # If len(weak_3) < 3, we can't have 3 uniques.
    if len(set(weak_3)) == 3:
        return 1.0

    # 2. Check interrupted 4-window if available
    if len(finger_sequence) >= 4:
        last_four = finger_sequence[-4:]
        weak_4_indices = [i for i, f in enumerate(last_four) if f in (3, 4, 5)]
        weak_4_values = [last_four[i] for i in weak_4_indices]
        
        # Must contain all three weak fingers
        if len(set(weak_4_values)) == 3:
            # Check for gaps (interruption by strong finger)
            # If indices are contiguous (e.g. 0,1,2 or 1,2,3), range = len-1.
            # If range > len-1, there is a gap.
            idx_range = max(weak_4_indices) - min(weak_4_indices)
            if idx_range > len(weak_4_indices) - 1:
                return 1.0

    return 0.0

def rule4_position_change_count(
    prev_prev_finger: int, prev_prev_note: int, prev_prev_is_black: bool,
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool,
    hand: int = 1
) -> float:
    """
    Rule 4: Position-Change-Count Rule.
    
    Detects if a transition requires a shift in hand position.
    Uses 2-step history (PP -> P -> C) to establish the 'Existing Position' anchor.
    """
    # 1. Crossing Exception
    span = curr_note - prev_note
    if span != 0 and (curr_finger - prev_finger) * span < 0:
        return 0.0

    # 2. Determine Anchor at P
    prev_valid = _get_valid_thumb_positions(prev_finger, prev_note, prev_is_black)
    
    # Refine P anchor using PP if available
    # If PP->P was a hold (overlap), we are constrained to the intersection.
    # If PP->P was a shift (no overlap), we reset to P_valid.
    if prev_prev_finger is not None:
        # Heuristic: If we return to the exact same state as PP (Finger + Note), 
        # assume we recovered the position for free.
        if curr_finger == prev_prev_finger and curr_note == prev_prev_note:
            return 0.0

        pp_valid = _get_valid_thumb_positions(prev_prev_finger, prev_prev_note, prev_prev_is_black)
        overlap_pp_p = pp_valid.intersection(prev_valid)
        if overlap_pp_p:
            prev_valid = overlap_pp_p
            
    # 3. Check Overlap with C
    curr_valid = _get_valid_thumb_positions(curr_finger, curr_note, curr_is_black)
    common = prev_valid.intersection(curr_valid)
    
    if common:
        return 0.0
        
    return 1.0


# =============================================================================
# RULE 5: POSITION-CHANGE-SIZE RULE
# Penalty based on distance traveled during position change.
# 1 point per step that interval between 1st and 3rd notes falls outside [MinComf, MaxComf].
# =============================================================================

def rule5_position_change_size(
    finger1: int, note1: int,
    finger3: int, note3: int
) -> float:
    """Rule 5: Position-Change-Size Rule."""
    # For size, we use the raw semitone distance? Or diatonic?
    # Usually strictly physical stretch -> Semitones. 
    # Current implementation uses raw semitones. Keeping as is unless tested otherwise.
    span = note3 - note1
    limits = get_finger_span_limits(finger1, finger3)
    min_comf, max_comf = limits[MIN_COMF], limits[MAX_COMF]
    
    norm_span = normalize_span(finger1, finger3, span)
    
    if norm_span > max_comf:
        return norm_span - max_comf
    elif norm_span < min_comf:
        return min_comf - norm_span
    
    return 0.0


# =============================================================================
# RULE 6: WEAK-FINGER RULE
# Penalty for using fingers 4 and 5.
# 1 point per use.
# =============================================================================

def rule6_weak_finger(finger: int) -> float:
    """Rule 6: Weak-Finger Rule."""
    return 1.0 if finger in (4, 5) else 0.0





# =============================================================================
# RULE 8: THREE-TO-FOUR RULE
# Penalty for following finger 3 with finger 4.
# 1 point per occurrence.
# =============================================================================

def rule8_three_to_four(prev_finger: int, curr_finger: int) -> float:
    """Rule 8: Three-to-Four Rule."""
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
    
    Penalty for placing finger 4 on a black key.
    """
    # DEBUG Trigger
    # if curr_finger == 4 and not curr_is_black:
    #     print(f"DEBUG R9: 4 on White? {prev_finger}->{curr_finger}")
    
    if curr_finger == 4 and curr_is_black:
        return 1.0
    
    return 0.0


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
    
    # If next context is missing (end of phrase), assume white key continuation/release
    if next_is_black is None or not next_is_black:
        cost += 2.0  # Next note is white
    
    return cost


# =============================================================================
# RULE 12: THUMB-PASSING RULE
# Penalty based on difficulty of passing thumb under/over another finger.
# 1 point for same level, 3 points for white-to-black passes involving thumb.
# =============================================================================

def rule12_thumb_passing(
    prev_finger: int, prev_is_black: bool,
    curr_finger: int, curr_is_black: bool,
    span: int = 0
) -> float:
    """
    Rule 12: Thumb-Passing Rule.
    
    Penalty based on the difficulty of passing the thumb under/over
    another finger relative to key elevation.
    
    A "pass" only occurs when the thumb physically crosses the hand
    boundary.  Two conditions must be met:
    
    1. Direction check — the finger-number ordering matches the span
       direction (the thumb is moving past the other fingers, not
       reaching within its natural range).
    2. Relaxed-range check — if the interval sits inside the relaxed
       span [MinRel, MaxRel], the fingers can reach comfortably
       without passing, so Rule 12 does not fire.
    
    Key-elevation penalty (when the rule fires):
    - Same level (both white or both black): 1 point
    - Thumb on black, non-thumb on white (hardest): 3 points
    - Thumb on white, non-thumb on black ("plenty of room"): 1 point
    
    Note: This rule fires independently of Rule 10 (Thumb-on-Black).
    
    Args:
        prev_finger: Previous finger (1-5)
        prev_is_black: Whether previous note is black key
        curr_finger: Current finger (1-5)
        curr_is_black: Whether current note is black key
        span: Signed semitone distance (positive = ascending)
    
    Returns:
        Cost: 0, 1, or 3 points
    """
    # Gate 1: Must involve the thumb
    is_thumb_pass = ((prev_finger == 1 and curr_finger != 1) or 
                     (prev_finger != 1 and curr_finger == 1))
    if not is_thumb_pass:
        return 0.0
    
    # Gate 2: Explicit Direction Check.
    # If the fingers move in the same lateral direction as the keys, it is a
    # "Reach" (extension/contraction), not a "Pass".
    # A Pass requires the physical order of fingers to contradict the order of keys.
    # Case 1: Ascending Keys (span > 0). Natural: F1 < F2 (Thumb Left of Finger). Pass: F1 > F2.
    # Case 2: Descending Keys (span < 0). Natural: F1 > F2. Pass: F1 < F2.
    # So if (F2 - F1) has same sign as span, it's consistent -> NO PASS.
    if (curr_finger - prev_finger) * span > 0:
        return 0.0
    
    # Gate 3: Directional relaxed-range check.
    # (Existing logic to catch cases that are technically crosses but within relaxed reach?)
    # Parncutt implies minimal crosses might be comfortable.
    limits = get_finger_span_limits(prev_finger, curr_finger)
    min_rel = limits[MIN_REL]
    max_rel = limits[MAX_REL]
    
    if prev_finger > curr_finger:
        actual_min_rel, actual_max_rel = -max_rel, -min_rel
    else:
        actual_min_rel, actual_max_rel = min_rel, max_rel
    
    if actual_min_rel <= span <= actual_max_rel:
        return 0.0
    
    # --- The transition IS a pass — compute key-elevation penalty ---
    
    # Same key color (same level)
    if prev_is_black == curr_is_black:
        return 1.0
    
    # Most difficult: thumb on black key, non-thumb on white key
    thumb_on_black = (
        (prev_finger != 1 and not prev_is_black and curr_finger == 1 and curr_is_black) or
        (prev_finger == 1 and prev_is_black and curr_finger != 1 and not curr_is_black)
    )
    if thumb_on_black:
        return 3.0
    
    # Easiest: thumb on white, non-thumb on black ("plenty of room")
    # Parncutt specifies 0 points for this advantageous crossing.
    return 0.0


# =============================================================================
# AGGREGATE COST FUNCTION
# =============================================================================

def calculate_parncutt_cost(
    prev_finger: int, prev_note: int, prev_is_black: bool,
    curr_finger: int, curr_note: int, curr_is_black: bool,
    next_finger: Optional[int] = None, next_note: Optional[int] = None, next_is_black: Optional[bool] = None,
    prev_prev_finger: Optional[int] = None, prev_prev_note: Optional[int] = None, prev_prev_is_black: Optional[bool] = None,
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
    if prev_prev_finger is not None and prev_prev_note is not None and prev_prev_is_black is not None:
        # Convert to semitones for position rules (chromatic distance)
        pp_semi = lattice_to_semitone(prev_prev_note, prev_prev_is_black)
        p_semi = lattice_to_semitone(prev_note, prev_is_black)
        c_semi = lattice_to_semitone(curr_note, curr_is_black)
        
        r4 = rule4_position_change_count(
            prev_prev_finger, pp_semi, prev_prev_is_black,
            prev_finger, p_semi, prev_is_black,
            curr_finger, c_semi, curr_is_black,
            hand=hand
        )
        cost += r4
        
        # Rule 5 applies only if there is a position change (R4 > 0)
        if r4 > 0:
            cost += rule5_position_change_size(
                prev_prev_finger, pp_semi,
                curr_finger, c_semi
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
    cost += rule12_thumb_passing(prev_finger, prev_is_black, curr_finger, curr_is_black, span)
    
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
    cost += rule12_thumb_passing(prev_finger, prev_is_black, curr_finger, curr_is_black, span)
    
    return cost
