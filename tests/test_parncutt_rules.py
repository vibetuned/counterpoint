"""
Tests for Parncutt 1997 ergonomic fingering rules.
"""

import pytest
from counterpoint.rules.parncutt97 import (
    get_finger_span_limits,
    involves_thumb,
    lattice_to_semitone,
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
    calculate_consecutive_cost,
    FINGER_SPANS,
    MIN_COMF, MAX_COMF, MIN_REL, MAX_REL,
)


class TestFingerSpanTables:
    """Test finger span lookup tables match the paper exactly."""
    
    def test_thumb_index_limits(self):
        """Test thumb-index (1-2) span limits match paper."""
        limits = get_finger_span_limits(1, 2)
        # Paper: MinPrac=-5, MinComf=-3, MinRel=1, MaxRel=5, MaxComf=8, MaxPrac=10
        assert limits == (-5, -3, 1, 5, 8, 10)
    
    def test_thumb_middle_limits(self):
        """Test thumb-middle (1-3) span limits match paper."""
        assert FINGER_SPANS[(1, 3)] == (-4, -2, 3, 7, 10, 12)
    
    def test_thumb_ring_limits(self):
        """Test thumb-ring (1-4) span limits match paper."""
        assert FINGER_SPANS[(1, 4)] == (-3, -1, 5, 9, 12, 14)
    
    def test_thumb_pinky_limits(self):
        """Test thumb-pinky (1-5) span limits match paper."""
        assert FINGER_SPANS[(1, 5)] == (-1, 1, 7, 11, 13, 15)
    
    def test_non_thumb_pairs(self):
        """Test non-thumb pair span limits match paper."""
        assert FINGER_SPANS[(2, 3)] == (1, 1, 1, 2, 3, 5)
        assert FINGER_SPANS[(2, 4)] == (1, 1, 3, 4, 5, 7)
        assert FINGER_SPANS[(2, 5)] == (2, 2, 5, 6, 8, 10)
        assert FINGER_SPANS[(3, 4)] == (1, 1, 1, 2, 2, 4)
        assert FINGER_SPANS[(3, 5)] == (1, 1, 3, 4, 5, 7)
        assert FINGER_SPANS[(4, 5)] == (1, 1, 1, 2, 3, 5)
    
    def test_symmetric_lookup(self):
        """Finger pairs should work in either order."""
        assert get_finger_span_limits(1, 3) == get_finger_span_limits(3, 1)
        assert get_finger_span_limits(2, 4) == get_finger_span_limits(4, 2)
    
    def test_involves_thumb(self):
        """Test thumb detection."""
        assert involves_thumb(1, 2) is True
        assert involves_thumb(2, 1) is True
        assert involves_thumb(2, 3) is False
        assert involves_thumb(4, 5) is False


class TestLatticeToSemitone:
    """Test lattice coordinate to semitone conversion."""
    
    def test_c_natural(self):
        """Column 0, white = C = semitone 0."""
        assert lattice_to_semitone(0, False) == 0
    
    def test_c_sharp(self):
        """Column 0, black = C# = semitone 1."""
        assert lattice_to_semitone(0, True) == 1
    
    def test_d_natural(self):
        """Column 1, white = D = semitone 2."""
        assert lattice_to_semitone(1, False) == 2
    
    def test_e_natural(self):
        """Column 2, white = E = semitone 4."""
        assert lattice_to_semitone(2, False) == 4
    
    def test_f_natural(self):
        """Column 3, white = F = semitone 5."""
        assert lattice_to_semitone(3, False) == 5
    
    def test_g_natural(self):
        """Column 4, white = G = semitone 7."""
        assert lattice_to_semitone(4, False) == 7
    
    def test_a_natural(self):
        """Column 5, white = A = semitone 9."""
        assert lattice_to_semitone(5, False) == 9
    
    def test_b_natural(self):
        """Column 6, white = B = semitone 11."""
        assert lattice_to_semitone(6, False) == 11
    
    def test_one_octave(self):
        """Column 7 = next C, exactly 12 semitones from column 0."""
        assert lattice_to_semitone(7, False) == 12
    
    def test_two_octaves(self):
        """Column 14 = C two octaves up."""
        assert lattice_to_semitone(14, False) == 24


class TestLatticeSpanToSemitones:
    """Test span computation between lattice positions."""
    
    def test_c_to_d(self):
        """C→D = 2 semitones (whole tone)."""
        assert lattice_span_to_semitones(0, False, 1, False) == 2
    
    def test_e_to_f(self):
        """E→F = 1 semitone (half step, no black key between them)."""
        assert lattice_span_to_semitones(2, False, 3, False) == 1
    
    def test_b_to_c(self):
        """B→C = 1 semitone (half step across octave boundary)."""
        assert lattice_span_to_semitones(6, False, 7, False) == 1
    
    def test_one_octave_span(self):
        """C to C an octave up = 12 semitones."""
        assert lattice_span_to_semitones(0, False, 7, False) == 12
    
    def test_descending(self):
        """D→C = -2 semitones (negative = descending)."""
        assert lattice_span_to_semitones(1, False, 0, False) == -2
    
    def test_with_black_keys(self):
        """C→C# = 1 semitone."""
        assert lattice_span_to_semitones(0, False, 0, True) == 1
    
    def test_black_to_white(self):
        """C#→D = 1 semitone."""
        assert lattice_span_to_semitones(0, True, 1, False) == 1
    
    def test_black_to_black(self):
        """C#→D# = 2 semitones."""
        assert lattice_span_to_semitones(0, True, 1, True) == 2


class TestRule1Stretch:
    """Test Rule 1: Stretch penalty (now in semitones)."""
    
    def test_no_penalty_within_comfortable(self):
        """No penalty for spans within comfortable range."""
        # Thumb-index, span=4 is within [MinComf=-3, MaxComf=8]
        assert rule1_stretch(1, 2, 4) == 0.0
    
    def test_penalty_beyond_max_comf(self):
        """Penalty for spans beyond MaxComf."""
        # Thumb-index MaxComf=8, span=12 exceeds by 4
        cost = rule1_stretch(1, 2, 12)
        assert cost == pytest.approx(8.0)  # 2 points per semitone * 4


class TestRule6WeakFinger:
    """Test Rule 6: Weak finger penalty."""
    
    def test_strong_fingers_no_penalty(self):
        """Fingers 1, 2, 3 have no penalty."""
        assert rule6_weak_finger(1) == 0.0
        assert rule6_weak_finger(2) == 0.0
        assert rule6_weak_finger(3) == 0.0
    
    def test_weak_fingers_penalty(self):
        """Fingers 4 and 5 have 1 point penalty."""
        assert rule6_weak_finger(4) == 1.0
        assert rule6_weak_finger(5) == 1.0


class TestRule8ThreeToFour:
    """Test Rule 8: Three-to-four penalty."""
    
    def test_three_to_four_penalty(self):
        """3→4 has 1 point penalty."""
        assert rule8_three_to_four(3, 4) == 1.0
    
    def test_four_to_three_no_penalty(self):
        """4→3 has no penalty (only 3→4)."""
        assert rule8_three_to_four(4, 3) == 0.0
    
    def test_other_transitions_no_penalty(self):
        """Other transitions have no penalty."""
        assert rule8_three_to_four(1, 2) == 0.0
        assert rule8_three_to_four(2, 3) == 0.0


class TestRule10ThumbOnBlack:
    """Test Rule 10: Thumb on black key penalty."""
    
    def test_thumb_on_white_no_penalty(self):
        """Thumb on white key has no penalty."""
        assert rule10_thumb_on_black(2, False, 1, False) == 0.0
    
    def test_thumb_on_black_base_penalty(self):
        """Thumb on black key has base 1 point."""
        # Prev is also black, next not specified
        cost = rule10_thumb_on_black(2, True, 1, True)
        assert cost == 1.0  # Just base point
    
    def test_thumb_on_black_with_white_before(self):
        """Thumb on black with white before adds 2 points."""
        cost = rule10_thumb_on_black(2, False, 1, True)
        assert cost == 3.0  # 1 base + 2 for white before


class TestRule12ThumbPassing:
    """Test Rule 12: Thumb passing penalty with directional relaxed-range gate."""
    
    def test_no_thumb_pass_no_penalty(self):
        """Non-thumb transitions have no penalty regardless of span."""
        assert rule12_thumb_passing(2, False, 3, False, span=-2) == 0.0
    
    def test_pass_outside_relaxed_range_same_level(self):
        """Span outside directional relaxed range, same key color → 1pt."""
        # 1→2, span=+8: pair (1,2) relaxed=[1,5]. 8>5 → pass → 1pt
        assert rule12_thumb_passing(1, False, 2, False, span=+8) == 1.0
        # 2→1, span=-8: directional relaxed=[-5,-1]. -8<-5 → pass → 1pt
        assert rule12_thumb_passing(2, False, 1, False, span=-8) == 1.0
    
    def test_pass_thumb_on_black(self):
        """Thumb on black outside relaxed → 3pts (hardest pass)."""
        # 3→1(black), span=-3: pair (1,3) dir relaxed=[-7,-3]. -3 at boundary → pass
        # But wait: -3 = actual_max_rel(-3), so -7 <= -3 <= -3 → within? 
        # Nope: actual_min=-7, actual_max=-3. -3 <= -3 → within. Hmm.
        # Need span OUTSIDE: use span=-8 instead
        assert rule12_thumb_passing(3, False, 1, True, span=-8) == 3.0
        # 1(black)→2(white), span=+8: outside [1,5] → pass → 3pts
        assert rule12_thumb_passing(1, True, 2, False, span=+8) == 3.0
    
    def test_thumb_on_white_easy(self):
        """Thumb on white, non-thumb on black = easy pass → 1pt."""
        # 1(white)→2(black), span=+8: outside relaxed → 1pt
        assert rule12_thumb_passing(1, False, 2, True, span=+8) == 1.0
    
    def test_within_relaxed_range_is_reach(self):
        """Span within directional relaxed range (inclusive) → reach → 0pts."""
        # 1→2, span=+3: pair (1,2) relaxed=[1,5]. 1<=3<=5 → reach → 0
        assert rule12_thumb_passing(1, False, 2, False, span=+3) == 0.0
        # 2→1, span=-3: dir relaxed=[-5,-1]. -5<=-3<=-1 → reach → 0
        assert rule12_thumb_passing(2, False, 1, False, span=-3) == 0.0
        # At boundaries: 1→2, span=+1 (=MinRel) → within → reach → 0
        assert rule12_thumb_passing(1, False, 2, False, span=+1) == 0.0
        # At boundaries: 1→2, span=+5 (=MaxRel) → within → reach → 0
        assert rule12_thumb_passing(1, False, 2, False, span=+5) == 0.0
    
    def test_reverse_direction_is_pass(self):
        """Span in opposite direction from finger ordering → outside bounds → pass."""
        # 1→2, span=-5: dir relaxed=[1,5]. -5 < 1 → outside → pass → 1pt
        assert rule12_thumb_passing(1, False, 2, False, span=-5) == 1.0
        # 2→1, span=+5: dir relaxed=[-5,-1]. +5 > -1 → outside → pass → 1pt
        assert rule12_thumb_passing(2, False, 1, False, span=+5) == 1.0


class TestConsecutiveCost:
    """Test the aggregate consecutive cost function."""
    
    def test_simple_adjacent_notes(self):
        """Adjacent notes with natural fingering should have low cost."""
        # Thumb to index, white C→D (2 semitones), col 0→1
        # Pair (1,2) MinRel=1, MaxRel=5: span=2 is in (1,5) → reach → R12=0
        cost = calculate_consecutive_cost(1, 0, False, 2, 1, False)
        assert cost == 0.0  # R2: abs(2)>=MinRel(1) → 0, R12: reach → 0
    
    def test_awkward_fingering_high_cost(self):
        """Awkward fingering should have higher cost."""
        # Finger 3 on white, finger 4 on black (next column)
        cost = calculate_consecutive_cost(3, 10, False, 4, 11, True)
        # Should include: four-on-black (1), weak finger (1), etc.
        assert cost >= 2.0


class TestPracticalSpanPruning:
    """Test the practical span enumeration filter (is_playable / UNPLAYABLE_COST)."""

    def test_playable_thumb_index_ascending(self):
        """Thumb(1)→Index(2), C→D (span=+2) is well within practical range."""
        from counterpoint.rules.parncutt97 import is_playable, UNPLAYABLE_COST
        assert is_playable(1, 2, 2) is True
        # Cost should be a normal value, not UNPLAYABLE
        cost = calculate_consecutive_cost(1, 0, False, 2, 1, False)
        assert cost < UNPLAYABLE_COST

    def test_unplayable_non_thumb_crossing(self):
        """Finger 3→2 on E→D (span=−2): unplayable.
        
        Pair (2,3) MinPrac=1, MaxPrac=5.
        Descending finger order (3→2): actual bounds = [-5, -1].
        Span = -2 ⇒ within bounds? Actually -5 <= -2 <= -1 ⇒ playable!
        
        But Finger 2→3 on D→E (span=+2): pair (2,3) ascending, bounds [1, 5] ⇒ playable.
        
        Test the real unplayable case: Finger 3→2 ascending (span=+2).
        Descending finger order (3→2): actual bounds = [-5, -1].
        Span = +2 is NOT in [-5, -1] ⇒ unplayable!
        """
        from counterpoint.rules.parncutt97 import is_playable, UNPLAYABLE_COST
        # Finger 3 then finger 2, but notes ASCENDING (+2 semitones)
        # Higher-numbered finger going to lower-numbered finger on ascending notes 
        # = finger crossing = unplayable
        assert is_playable(3, 2, 2) is False
        # The aggregate cost should return UNPLAYABLE_COST
        # col 0 (C) → col 1 (D) = +2 semitones
        cost = calculate_consecutive_cost(3, 0, False, 2, 1, False)
        assert cost == UNPLAYABLE_COST

    def test_unplayable_excessive_stretch(self):
        """Thumb(1)→Index(2) with span > MaxPrac(10) is unplayable."""
        from counterpoint.rules.parncutt97 import is_playable, UNPLAYABLE_COST
        assert is_playable(1, 2, 11) is False
        assert is_playable(1, 2, 10) is True  # At the boundary = playable

    def test_playable_thumb_under(self):
        """Thumb-under: finger 3→thumb(1), ascending span.
        
        Descending finger order (3→1): pair (1,3), bounds flipped.
        (1,3) MinPrac=-4, MaxPrac=12 → flipped: [-12, 4].
        Span = +3 is in [-12, 4] ⇒ playable.
        """
        from counterpoint.rules.parncutt97 import is_playable
        assert is_playable(3, 1, 3) is True

    def test_same_finger_always_playable(self):
        """Same finger on same note (span=0) is always valid."""
        from counterpoint.rules.parncutt97 import is_playable
        for f in range(1, 6):
            assert is_playable(f, f, 0) is True

    def test_full_cost_returns_unplayable(self):
        """calculate_parncutt_cost also returns UNPLAYABLE_COST for impossible spans."""
        from counterpoint.rules.parncutt97 import calculate_parncutt_cost, UNPLAYABLE_COST
        # Finger 3→2 ascending = crossing = unplayable
        cost = calculate_parncutt_cost(
            prev_finger=3, prev_note=0, prev_is_black=False,
            curr_finger=2, curr_note=1, curr_is_black=False,
        )
        assert cost == UNPLAYABLE_COST

    def test_lh_ascending_span_descending_fingers_playable(self):
        """LH: ascending span + descending fingers (5→4) is natural and playable."""
        from counterpoint.rules.parncutt97 import is_playable
        # LH finger 5 (leftmost) → 4, ascending +2 semitones
        # In LH this is natural movement (pinky left, ring right)
        assert is_playable(5, 4, 2, hand=2) is True

    def test_lh_ascending_span_ascending_fingers_unplayable(self):
        """LH: ascending span + ascending fingers (2→3) = crossing = unplayable."""
        from counterpoint.rules.parncutt97 import is_playable
        # LH finger 2→3 ascending: in LH, finger 2 is to the RIGHT of 3,
        # so ascending notes + ascending fingers = crossing
        assert is_playable(2, 3, 2, hand=2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
