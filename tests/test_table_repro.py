
import pytest
from counterpoint.rules.parncutt97 import (
    calculate_parncutt_cost,
    lattice_to_semitone,
    lattice_span_to_semitones,
    rule1_stretch, rule2_small_span, rule3_large_span,
    rule4_position_change_count, rule5_position_change_size,
    rule6_weak_finger, rule7_three_four_five, rule8_three_to_four,
    rule9_four_on_black, rule10_thumb_on_black, rule11_five_on_black,
    rule12_thumb_passing, is_playable, UNPLAYABLE_COST
)

# Piece A: E G F G E F D E
# E4, G4, F4, G4, E4, F4, D4, E4
# Lattice cols: E=2, G=4, F=3, D=1. All White=False.
NOTES_A = [
    (2, False),   # E
    (4, False),   # G
    (3, False),   # F
    (4, False),   # G
    (2, False),   # E
    (3, False),   # F
    (1, False),   # D
    (2, False),   # E
]

# Table A Data: Rank, Fingering, [R1...R12], Total
TABLE_A_DATA = [
    (1, [2, 4, 3, 4, 2, 3, 1, 3], [0, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0], 6),
    (2, [1, 3, 2, 3, 1, 2, 1, 3], [0, 1, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0], 7),
    (3, [1, 4, 3, 4, 2, 3, 1, 3], [0, 3, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0], 8),
    (4, [3, 5, 4, 5, 3, 4, 2, 3], [0, 0, 0, 0, 0, 4, 3, 1, 0, 0, 0, 0], 8),
    (5, [1, 3, 2, 3, 1, 3, 2, 3], [0, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0], 9),
    (6, [1, 3, 2, 3, 1, 4, 2, 3], [0, 4, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0], 9),
    (7, [1, 3, 2, 4, 2, 3, 1, 3], [0, 3, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0], 9),
    (8, [1, 2, 1, 4, 2, 3, 1, 3], [0, 4, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0], 10),
    (9, [1, 3, 2, 3, 2, 3, 1, 3], [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0], 10),
    (10, [2, 4, 3, 4, 2, 3, 2, 3], [0, 0, 2, 2, 3, 2, 0, 1, 0, 0, 0, 0], 10),
]

# Piece B: E-C#-D-F#
NOTES_B = [
    (2, False),   # E
    (0, True),    # C#
    (1, False),   # D
    (3, True),    # F#
]


# Table B Data: Rank, Fingering, [R1...R12], Total
TABLE_B_DATA = [
    (1, [3, 1, 2, 4], [0, 0, 0, 0, 0, 1, 0, 0, 1, 5, 0, 0], 7),
    (2, [3, 2, 1, 4], [0, 3, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0], 7),
    (3, [4, 2, 3, 5], [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 4, 0], 7),
    (4, [4, 2, 1, 5], [0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0], 11),
    (5, [1, 2, 3, 5], [0, 9, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0], 14),
    (6, [2, 1, 2, 4], [0, 2, 0, 2, 4, 1, 0, 0, 0, 5, 0, 0], 14),
    (7, [3, 1, 2, 5], [0, 4, 0, 0, 0, 1, 0, 0, 0, 5, 4, 0], 14),
    (8, [3, 2, 1, 5], [0, 7, 2, 0, 0, 1, 0, 0, 0, 0, 4, 0], 14),
    (9, [3, 2, 3, 5], [0, 2, 2, 2, 4, 1, 0, 0, 0, 0, 4, 0], 15),
    (10, [4, 1, 2, 5], [0, 4, 0, 0, 0, 2, 0, 0, 0, 5, 4, 0], 15),
    (10, [4, 1, 2, 5], [0, 4, 0, 0, 0, 2, 0, 0, 0, 5, 4, 0], 15),
]

# Piece C: E-C#-E-A-E
NOTES_C = [
    (2, False),   # E
    (0, True),    # C#
    (2, False),   # E
    (5, False),   # A
    (2, False),   # E
]

# Table C Data: Rank, Fingering, [R1...R12], Total
TABLE_C_DATA = [
    (1, [2, 1, 2, 4, 1], [0, 0, 2, 1, 0, 1, 0, 0, 0, 5, 0, 0], 9),
    (2, [2, 1, 2, 5, 1], [0, 2, 0, 1, 0, 1, 0, 0, 0, 5, 0, 0], 9),
    (3, [3, 1, 2, 4, 1], [0, 0, 2, 2, 0, 1, 0, 0, 0, 5, 0, 0], 10),
    (4, [3, 1, 2, 5, 1], [0, 2, 0, 2, 0, 1, 0, 0, 0, 5, 0, 0], 10),
    (5, [3, 2, 3, 5, 1], [0, 2, 6, 1, 0, 1, 0, 0, 0, 0, 0, 0], 10),
    (6, [4, 2, 3, 5, 1], [0, 2, 4, 2, 0, 2, 0, 0, 0, 0, 0, 0], 10),
    (7, [3, 1, 3, 5, 1], [0, 2, 2, 1, 0, 1, 0, 0, 0, 5, 0, 0], 11),
    (8, [4, 2, 1, 5, 1], [0, 8, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 11),
    (9, [2, 1, 3, 5, 1], [0, 2, 2, 2, 0, 1, 0, 0, 0, 5, 0, 0], 12),
    (10, [3, 2, 1, 5, 1], [0, 8, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0], 12),
]

def compute_full_cost(notes, fingers, hand=1):
    total = 0.0
    details = []
    
    # Accumulators for checking against table
    rule_sums = [0.0] * 12  # R1-R12

    for i in range(1, len(notes)):
        prev_col, prev_black = notes[i - 1]
        curr_col, curr_black = notes[i]
        prev_f = fingers[i - 1]
        curr_f = fingers[i]

        prev_prev_f = fingers[i - 2] if i >= 2 else None
        prev_prev_col = notes[i - 2][0] if i >= 2 else None
        prev_prev_prev_f = fingers[i - 3] if i >= 3 else None
        
        # Next context for R10/R11
        next_f = fingers[i + 1] if i + 1 < len(fingers) else None
        next_black = notes[i + 1][1] if i + 1 < len(notes) else None

        span = lattice_span_to_semitones(prev_col, prev_black, curr_col, curr_black)

        if not is_playable(prev_f, curr_f, span, hand):
            total = UNPLAYABLE_COST
            break

        r1 = rule1_stretch(prev_f, curr_f, span)
        r2 = rule2_small_span(prev_f, curr_f, span)
        r3 = rule3_large_span(prev_f, curr_f, span)
        r6 = rule6_weak_finger(curr_f)
        r8 = rule8_three_to_four(prev_f, curr_f)
        r9 = rule9_four_on_black(prev_f, prev_black, curr_f, curr_black)
        r10 = rule10_thumb_on_black(prev_f, prev_black, curr_f, curr_black, next_f, next_black)
        r11 = rule11_five_on_black(prev_f, prev_black, curr_f, curr_black, next_f, next_black)
        r12 = rule12_thumb_passing(prev_f, prev_black, curr_f, curr_black, span)

        r4 = 0.0
        r5 = 0.0
        r7 = 0.0
        if prev_prev_f is not None and prev_prev_col is not None:
            prev_prev_black = notes[i - 2][1]
            # Use semitones for position rules
            pp_semi = lattice_to_semitone(prev_prev_col, prev_prev_black)
            p_semi = lattice_to_semitone(prev_col, prev_black)
            c_semi = lattice_to_semitone(curr_col, curr_black)
            
            r4 = rule4_position_change_count(
                prev_prev_f, pp_semi, prev_prev_black,
                prev_f, p_semi, prev_black,
                curr_f, c_semi, curr_black,
                hand=hand
            )
            
            if r4 > 0:
                r5 = rule5_position_change_size(prev_prev_f, pp_semi, curr_f, c_semi)
            else:
                r5 = 0.0
            
            r7_seq = [prev_prev_f, prev_f, curr_f]
            if prev_prev_prev_f is not None:
                r7_seq.insert(0, prev_prev_prev_f)
            r7 = rule7_three_four_five(r7_seq)

        current_rules = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12]
        
        # DEBUG PRINT
        if sum(current_rules) > 0:
            print(f"  Step {i} ({prev_f}->{curr_f}): {current_rules}")

        for idx, val in enumerate(current_rules):
            rule_sums[idx] += val
            
        step_cost = sum(current_rules)
        total += step_cost
        
    # Check First Note for single-note rules (R6, R9?)
    # Based on Parncutt, some rules apply to single notes.
    # R6 (Weak Finger): Yes.
    # R9 (4 on Black): Yes.
    # R11 (5 on Black): Yes.
    # R10 (Thumb on Black): Yes.
    # Our loop starts at 1, so we missed note 0.
    f0 = fingers[0]
    n0_col, n0_black = notes[0]
    # R6
    r6_0 = rule6_weak_finger(f0)
    # R9
    r9_0 = rule9_four_on_black(0, False, f0, n0_black) # prev doesn't matter for new R9
    # R11
    r11_0 = rule11_five_on_black(0, False, f0, n0_black, fingers[1], notes[1][1])
    # R10
    r10_0 = rule10_thumb_on_black(0, False, f0, n0_black, fingers[1], notes[1][1])
    
    first_note_cost = r6_0 + r9_0 + r11_0 + r10_0
    rule_sums[5] += r6_0
    rule_sums[8] += r9_0
    rule_sums[9] += r10_0
    rule_sums[10] += r11_0
    total += first_note_cost
    
    if first_note_cost > 0:
        print(f"  Step 0 ({f0}): R6={r6_0} R9={r9_0} R10={r10_0} R11={r11_0}")

    return total, rule_sums

@pytest.mark.parametrize("rank, fingers, expected_rules, expected_total", TABLE_A_DATA)
def test_table_a_row(rank, fingers, expected_rules, expected_total):
    total, rule_sums = compute_full_cost(NOTES_A, fingers)
    
    print(f"\n[Table A] Rank {rank} Fingers {fingers}")
    print(f"Expected Total: {expected_total}, Got: {total}")
    print(f"Expected Rules: {expected_rules}")
    print(f"Got Rules:      {rule_sums}")
    
    # Assert Total
    assert total == pytest.approx(float(expected_total)), f"[Table A] Rank {rank}: Total mismatch"

@pytest.mark.parametrize("rank, fingers, expected_rules, expected_total", TABLE_B_DATA)
def test_table_b_row(rank, fingers, expected_rules, expected_total):
    total, rule_sums = compute_full_cost(NOTES_B, fingers)
    
    print(f"\n[Table B] Rank {rank} Fingers {fingers}")
    print(f"Expected Total: {expected_total}, Got: {total}")
    print(f"Expected Rules: {expected_rules}")
    print(f"Got Rules:      {rule_sums}")
    
    # Assert Total
    # Assert Total
    assert total == pytest.approx(float(expected_total)), f"[Table B] Rank {rank}: Total mismatch"

@pytest.mark.parametrize("rank, fingers, expected_rules, expected_total", TABLE_C_DATA)
def test_table_c_row(rank, fingers, expected_rules, expected_total):
    total, rule_sums = compute_full_cost(NOTES_C, fingers)
    
    print(f"\n[Table C] Rank {rank} Fingers {fingers}")
    print(f"Expected Total: {expected_total}, Got: {total}")
    print(f"Expected Rules: {expected_rules}")
    print(f"Got Rules:      {rule_sums}")
    
    # Assert Total
    assert total == pytest.approx(float(expected_total)), f"[Table C] Rank {rank}: Total mismatch"
