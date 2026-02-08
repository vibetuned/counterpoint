"""
Tests for Parncutt 1997 ergonomic fingering rules.
"""

import pytest
from counterpoint.rules.parncutt97 import (
    get_finger_span_limits,
    involves_thumb,
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
    MIN_COMF, MAX_COMF, MIN_REL, MAX_REL,
)


class TestFingerSpanTables:
    """Test finger span lookup tables."""
    
    def test_thumb_index_limits(self):
        """Test thumb-index (1-2) span limits."""
        limits = get_finger_span_limits(1, 2)
        # MinPrac=-2.5, MinComf=-1.5, MinRel=0.5, MaxRel=2.5, MaxComf=4.0, MaxPrac=5.0
        assert limits[MIN_COMF] == -1.5
        assert limits[MAX_COMF] == 4.0
    
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


class TestRule1Stretch:
    """Test Rule 1: Stretch penalty."""
    
    def test_no_penalty_within_comfortable(self):
        """No penalty for spans within comfortable range."""
        # Thumb-index, span=2 is within [MinComf=-1.5, MaxComf=4.0]
        assert rule1_stretch(1, 2, 2) == 0.0
    
    def test_penalty_beyond_max_comf(self):
        """Penalty for spans beyond MaxComf."""
        # Thumb-index MaxComf=4.0, span=6 exceeds by 2
        cost = rule1_stretch(1, 2, 6)
        assert cost == pytest.approx(4.0)  # 2 points per step


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
    """Test Rule 12: Thumb passing penalty."""
    
    def test_no_thumb_pass_no_penalty(self):
        """Non-thumb transitions have no penalty."""
        assert rule12_thumb_passing(2, False, 3, False) == 0.0
    
    def test_thumb_pass_same_level(self):
        """Thumb pass at same level (both white) has 1 point."""
        assert rule12_thumb_passing(1, False, 2, False) == 1.0
        assert rule12_thumb_passing(2, False, 1, False) == 1.0
    
    def test_thumb_pass_different_levels(self):
        """Thumb pass across levels (white-black) has 3 points."""
        assert rule12_thumb_passing(1, False, 2, True) == 3.0
        assert rule12_thumb_passing(2, True, 1, False) == 3.0


class TestConsecutiveCost:
    """Test the aggregate consecutive cost function."""
    
    def test_simple_adjacent_notes(self):
        """Adjacent notes with natural fingering should have low cost."""
        # Thumb to index, white keys, span=1
        cost = calculate_consecutive_cost(1, 10, False, 2, 11, False)
        # Should include: weak finger (0), thumb pass (1)
        assert cost >= 1.0
    
    def test_awkward_fingering_high_cost(self):
        """Awkward fingering should have higher cost."""
        # Finger 4 on black after 3 on white
        cost = calculate_consecutive_cost(3, 10, False, 4, 11, True)
        # Should include: four-on-black (1), weak finger (1), etc.
        assert cost >= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
