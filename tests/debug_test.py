
import pytest
from tests.test_user_cases import TestUserCase2, compute_full_cost, print_breakdown

def test_debug_piece_a():
    t = TestUserCase2()
    fingers = [3, 5, 4, 5, 3, 4, 2, 3]
    total, details = compute_full_cost(t.NOTES, fingers)
    print_breakdown("DEBUG Piece A", "E-G-F-G-E-F-D-E", fingers, total, details)
    # Fail to see output
    assert False, f"Total: {total}"
