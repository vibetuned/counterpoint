
"""
Rules for the Linear Agent cost function.
Based on rules.md.
"""

def base_distance_cost(prev_note, curr_note):
    """
    Base Distance cost: 1 if note changes, 0 if same note.
    """
    return 1.0 if prev_note != curr_note else 0.0

def rule_5_fourth_finger(curr_finger):
    """
    Rule 5: For every use of the fourth finger.
    """
    return 1.0 if curr_finger == 4 else 0.0

def rule_6_sequential_3_4(prev_finger, curr_finger):
    """
    Rule 6: For the use of the third and the fourth finger consecutively.
    (3->4 or 4->3)
    """
    if (prev_finger == 3 and curr_finger == 4) or (prev_finger == 4 and curr_finger == 3):
        return 1.0
    return 0.0

def rule_7_white_3_black_4(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 7: For the use of the third finger on a white key and the fourth finger on a black key consecutively.
    """
    is_3W_4B = (prev_finger == 3 and not prev_is_black) and (curr_finger == 4 and curr_is_black)
    is_4B_3W = (prev_finger == 4 and prev_is_black) and (curr_finger == 3 and not curr_is_black)
    
    if is_3W_4B or is_4B_3W:
        return 1.0
    return 0.0

def rule_8_thumb_black(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 8: Thumb (1) on Black Key.
    """
    cost = 0.0
    
    # Case A: Curr is Thumb on Black.
    if curr_finger == 1 and curr_is_black:
        cost += 0.5
        # "different finger used on a white key just before"
        if prev_finger != 1 and not prev_is_black:
            cost += 1.0
    
    # Case B: Prev was Thumb on Black.
    if prev_finger == 1 and prev_is_black:
        # "one extra for one just after" (different finger on white key)
        if curr_finger != 1 and not curr_is_black:
            cost += 1.0
            
    return cost

def rule_9_pinky_black(prev_finger, prev_is_black, curr_finger, curr_is_black):
    """
    Rule 9: Fifth (5) on Black Key.
    """
    cost = 0.0
    
    # Case A: Curr is 5 on Black.
    if curr_finger == 5 and curr_is_black:
        # Base cost 0.
        # "different finger used on a white key just before"
        if prev_finger != 5 and not prev_is_black:
            cost += 1.0
    
    # Case B: Prev was 5 on Black.
    if prev_finger == 5 and prev_is_black:
        # "one extra for one just after"
        if curr_finger != 5 and not curr_is_black:
            cost += 1.0
            
    return cost

def rule_position_change(prev_finger, prev_note, curr_finger, curr_note):
    """
    Rule 5 (from rules_paslclrade97.md): Position-Change-Size Rule.
    Penalty based on the physical distance the hand must travel during a position change.
    1 point per semitone.
    
    Hand Position = Note - (FingerIndex). FingerIndex is 0-4.
    Input fingers are 1-5.
    """
    prev_hand_pos = prev_note - (prev_finger - 1)
    curr_hand_pos = curr_note - (curr_finger - 1)
    
    return float(abs(curr_hand_pos - prev_hand_pos)) * 2.0

def calculate_transition_cost(prev_finger, prev_note, prev_is_black, curr_finger, curr_note, curr_is_black):
    """
    Aggregates all cost rules for a transition.
    """
    cost = 0.0
    cost += base_distance_cost(prev_note, curr_note)
    cost += rule_5_fourth_finger(curr_finger)
    cost += rule_6_sequential_3_4(prev_finger, curr_finger)
    cost += rule_7_white_3_black_4(prev_finger, prev_is_black, curr_finger, curr_is_black)
    cost += rule_8_thumb_black(prev_finger, prev_is_black, curr_finger, curr_is_black)
    cost += rule_9_pinky_black(prev_finger, prev_is_black, curr_finger, curr_is_black)
    cost += rule_position_change(prev_finger, prev_note, curr_finger, curr_note)
    return cost
