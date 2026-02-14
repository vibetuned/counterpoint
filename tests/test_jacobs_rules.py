"""
Tests for Jacobs 2001 ergonomic fingering rules.

Verifies the key differences from Parncutt 1997:
- Rule A: Physical distance mapping (now semitone-based)
- Rule C: Unified large-span (1x for ALL pairs)
- Rule 6 mod: Only finger 4 is weak
- Rule 7: Disabled
"""

import pytest

from counterpoint.rules.jacobs01 import (
    SEMITONE_DISTANCE_MM,
    semitone_span_to_physical_mm,
    physical_mm_to_semitone_bin,
    physical_distance_to_effective_semitone_span,
    jacobs_stretch,
    jacobs_small_span,
    jacobs_large_span,
    jacobs_weak_finger,
    jacobs_three_four_five,
    calculate_jacobs_consecutive_cost,
    calculate_jacobs_cost,
)

from counterpoint.rules.parncutt97 import (
    rule3_large_span,
    rule6_weak_finger,
    rule7_three_four_five,
    get_finger_span_limits,
    MAX_REL,
)


# =============================================================================
# Rule A: Physical Distance Mapping
# =============================================================================

class TestPhysicalDistanceMapping:
    """Test the physical distance → semitone bin pipeline."""

    def test_semitone_to_mm(self):
        """1 semitone should be ~13.75 mm."""
        mm = semitone_span_to_physical_mm(1.0)
        assert mm == pytest.approx(13.75, abs=0.1)

    def test_semitone_to_mm_negative(self):
        """Should use absolute value for negative spans."""
        assert semitone_span_to_physical_mm(-3.0) == semitone_span_to_physical_mm(3.0)

    def test_zero_distance(self):
        """Zero span should map to bin 0."""
        assert physical_mm_to_semitone_bin(0.0) == 0

    def test_one_semitone_bin(self):
        """~13.75 mm should map to bin 1 (1 semitone)."""
        assert physical_mm_to_semitone_bin(SEMITONE_DISTANCE_MM) == 1

    def test_two_semitone_bin(self):
        """~27.5 mm should map to bin 2 (2 semitones)."""
        assert physical_mm_to_semitone_bin(SEMITONE_DISTANCE_MM * 2) == 2

    def test_boundary_rounding(self):
        """Values at bin boundaries should round correctly."""
        # Midpoint between d_1 and d_2 = (1.5 * SEMITONE_DISTANCE_MM)
        # Values just above should map to 2, just below to 1
        mid = 1.5 * SEMITONE_DISTANCE_MM
        assert physical_mm_to_semitone_bin(mid + 0.1) == 2
        assert physical_mm_to_semitone_bin(mid - 0.1) == 1

    def test_effective_span_passthrough(self):
        """Integer semitone spans should pass through unchanged."""
        # 1 semitone → 13.75mm → bin 1 → 1.0
        assert physical_distance_to_effective_semitone_span(1.0) == 1.0

    def test_effective_span_larger(self):
        """Larger spans should map proportionally."""
        # 5 semitones → ~68.75mm → bin 5 → 5.0
        assert physical_distance_to_effective_semitone_span(5.0) == 5.0


# =============================================================================
# Rule C: Unified Large-Span
# =============================================================================

class TestJacobsLargeSpan:
    """Test unified large-span rule."""

    def test_no_penalty_within_maxrel(self):
        """No penalty for spans within MaxRel."""
        # Thumb-index MaxRel = 5
        assert jacobs_large_span(1, 2, 4) == 0.0

    def test_penalty_1x_for_thumb_pair(self):
        """Thumb pairs use 1x multiplier (same as Parncutt here)."""
        limits = get_finger_span_limits(1, 2)
        max_rel = limits[MAX_REL]  # 5
        # Span 1 semitone beyond max_rel
        cost = jacobs_large_span(1, 2, max_rel + 1)
        assert cost > 0.0

    def test_penalty_1x_for_nonthumb_pair(self):
        """Non-thumb pairs ALSO use 1x (unlike Parncutt's 2x)."""
        limits = get_finger_span_limits(2, 3)
        max_rel = limits[MAX_REL]  # 2
        
        # Compare Jacobs (1x) vs Parncutt (2x) for same span
        span = max_rel + 4  # Well beyond MaxRel
        jacobs_cost = jacobs_large_span(2, 3, span)
        parncutt_cost = rule3_large_span(2, 3, span)
        
        # Jacobs should be less than or equal to Parncutt for non-thumb
        assert jacobs_cost <= parncutt_cost


# =============================================================================
# Rule 6: Modified Weak Finger
# =============================================================================

class TestJacobsWeakFinger:
    """Test modified weak finger rule."""

    def test_finger_4_weak(self):
        """Finger 4 should still be penalized."""
        assert jacobs_weak_finger(4) == 1.0

    def test_finger_5_not_weak(self):
        """Finger 5 should NOT be penalized (key Jacobs difference)."""
        assert jacobs_weak_finger(5) == 0.0

    def test_parncutt_finger_5_is_weak(self):
        """Verify Parncutt DOES penalize finger 5 (for contrast)."""
        assert rule6_weak_finger(5) == 1.0

    def test_strong_fingers_no_penalty(self):
        """Fingers 1-3 should have no penalty."""
        for f in [1, 2, 3]:
            assert jacobs_weak_finger(f) == 0.0


# =============================================================================
# Rule 7: Disabled
# =============================================================================

class TestJacobsThreeFourFiveDisabled:
    """Test that the 3-4-5 rule is disabled."""

    def test_always_zero(self):
        """Should return 0.0 for any input."""
        assert jacobs_three_four_five([3, 4, 5]) == 0.0
        assert jacobs_three_four_five([5, 4, 3]) == 0.0

    def test_parncutt_not_zero(self):
        """Verify Parncutt DOES penalize (for contrast)."""
        assert rule7_three_four_five([3, 4, 5]) == 1.0


# =============================================================================
# Aggregate Cost
# =============================================================================

class TestJacobsConsecutiveCost:
    """Test aggregate cost functions."""

    def test_basic_cost(self):
        """Consecutive cost should be non-negative."""
        cost = calculate_jacobs_consecutive_cost(1, 0, False, 2, 1, False)
        assert cost >= 0.0

    def test_finger5_less_costly_than_parncutt(self):
        """Using finger 5 should be cheaper in Jacobs (not weak)."""
        from counterpoint.rules.parncutt97 import calculate_consecutive_cost
        
        # Same transition but with finger 5: C→G (col 0→4, 7 semitones)
        jacobs = calculate_jacobs_consecutive_cost(1, 0, False, 5, 4, False)
        parncutt = calculate_consecutive_cost(1, 0, False, 5, 4, False)
        
        # Jacobs should be strictly cheaper (no finger-5 penalty)
        assert jacobs < parncutt

    def test_full_cost_with_3note_context(self):
        """Full cost function with 3-note context works."""
        cost = calculate_jacobs_cost(
            prev_finger=1, prev_note=0, prev_is_black=False,
            curr_finger=2, curr_note=1, curr_is_black=False,
            prev_prev_finger=3, prev_prev_note=6,
        )
        assert cost >= 0.0


# =============================================================================
# Integration / Import Tests
# =============================================================================

class TestImports:
    """Test that all imports work correctly."""

    def test_import_jacobs_rules(self):
        """All Jacobs rule functions should be importable."""
        from counterpoint.rules.jacobs01 import (
            jacobs_stretch,
            jacobs_small_span,
            jacobs_large_span,
            jacobs_weak_finger,
            jacobs_three_four_five,
            calculate_jacobs_cost,
            calculate_jacobs_consecutive_cost,
        )

    def test_import_jacobs_rewards(self):
        """All Jacobs reward components should be importable."""
        from counterpoint.envs.jacobs_rewards import (
            JacobsStretchPenalty,
            JacobsSmallSpanPenalty,
            JacobsLargeSpanPenalty,
            JacobsWeakFingerPenalty,
            JacobsThreeFourFivePenalty,
            Jacobs01AllPenalties,
        )

    def test_import_linear_rules(self):
        """Linear agent should have both cost functions."""
        from counterpoint.linear.rules import (
            calculate_transition_cost,
            calculate_jacobs_transition_cost,
        )

    def test_import_from_init(self):
        """Module __init__ should export Jacobs rules."""
        from counterpoint.rules import (
            jacobs_large_span,
            jacobs_weak_finger,
            calculate_jacobs_cost,
        )

    def test_import_lattice_conversion(self):
        """Lattice conversion functions should be importable from rules init."""
        from counterpoint.rules import (
            lattice_to_semitone,
            lattice_span_to_semitones,
        )


class TestJacobsPracticalSpanPruning:
    """Test the practical span enumeration filter in Jacobs cost functions."""

    def test_playable_transition(self):
        """Normal playable transition returns finite cost."""
        from counterpoint.rules.parncutt97 import UNPLAYABLE_COST
        # Thumb→index, C→D (ascending, +2 semitones)
        cost = calculate_jacobs_consecutive_cost(1, 0, False, 2, 1, False)
        assert cost < UNPLAYABLE_COST

    def test_unplayable_finger_crossing(self):
        """Finger 3→2 ascending = crossing = UNPLAYABLE_COST."""
        from counterpoint.rules.parncutt97 import UNPLAYABLE_COST
        cost = calculate_jacobs_consecutive_cost(3, 0, False, 2, 1, False)
        assert cost == UNPLAYABLE_COST

    def test_full_cost_unplayable(self):
        """calculate_jacobs_cost also returns UNPLAYABLE_COST."""
        from counterpoint.rules.parncutt97 import UNPLAYABLE_COST
        cost = calculate_jacobs_cost(
            prev_finger=3, prev_note=0, prev_is_black=False,
            curr_finger=2, curr_note=1, curr_is_black=False,
        )
        assert cost == UNPLAYABLE_COST

