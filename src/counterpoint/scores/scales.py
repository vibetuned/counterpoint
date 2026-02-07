
import numpy as np
from typing import List, Tuple

# Major scale intervals in semitones: W-W-H-W-W-W-H
# But in our grid, columns are white keys (C, D, E, F, G, A, B)
# and row 1 is for accidentals (black keys)
#
# For our 2x52 grid representation:
# - Column index = white key index (0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B per octave)
# - Row 0 = natural (white key), Row 1 = accidental (black key on that column)
#
# Major scales with their patterns:
# Each tuple is (column_offset_from_root, is_black)
# Root is always (0, 0) for white key roots

MAJOR_SCALES = {
    # C Major: C D E F G A B (all white)
    "C": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
    
    # G Major: G A B C D E F# (F# is black on column 3 of next position)
    "G": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1)],  # F# = col 6, black
    
    # D Major: D E F# G A B C# 
    "D": [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0), (5, 0), (6, 1)],  # F#, C#
    
    # A Major: A B C# D E F# G#
    "A": [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0), (5, 1), (6, 1)],  # C#, F#, G#
    
    # E Major: E F# G# A B C# D#
    "E": [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0), (5, 1), (6, 1)],  # F#, G#, C#, D#
    
    # B Major: B C# D# E F# G# A#
    "B": [(0, 0), (1, 1), (2, 1), (3, 0), (4, 1), (5, 1), (6, 1)],  # C#, D#, F#, G#, A#
    
    # F Major: F G A Bb C D E (Bb is black on column 6 of prev octave position)
    "F": [(0, 0), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0)],  # Bb
    
    # Bb Major: Bb C D Eb F G A
    "Bb": [(0, 1), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0)],  # Bb, Eb
    
    # Eb Major: Eb F G Ab Bb C D
    "Eb": [(0, 1), (1, 0), (2, 0), (3, 1), (4, 1), (5, 0), (6, 0)],  # Eb, Ab, Bb
    
    # Ab Major: Ab Bb C Db Eb F G
    "Ab": [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 0), (6, 0)],  # Ab, Bb, Db, Eb
    
    # Db Major: Db Eb F Gb Ab Bb C
    "Db": [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1), (6, 0)],  # Db, Eb, Gb, Ab, Bb
    
    # Gb Major: Gb Ab Bb Cb Db Eb F
    "Gb": [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0)],  # Gb, Ab, Bb, Db, Eb, Cb
}

# Root positions for each scale (which white key column is the root)
# In our grid: 0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B (repeating every 7 columns)
SCALE_ROOTS = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6,
    "Bb": 6, "Eb": 2, "Ab": 5, "Db": 1, "Gb": 4,
}

# Classical 2-octave fingerings for major scales
# Fingers: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
# RH = Right Hand (ascending), LH = Left Hand (ascending)
# These are the standard fingerings taught in classical piano
SCALE_FINGERINGS = {
    # White key scales - standard pattern
    "C":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    "G":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    "D":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    "A":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    "E":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    # B major - different LH pattern
    "B":  {"RH": [1,2,3,1,2,3,4,1,2,3,1,2,3,4,5], "LH": [4,3,2,1,4,3,2,1,3,2,1,4,3,2,1]},
    # F# / Gb - starts on black key
    "Gb": {"RH": [2,3,4,1,2,3,1,2,3,4,1,2,3,1,2], "LH": [4,3,2,1,3,2,1,4,3,2,1,3,2,1,4]},
    # F major - thumb crosses to 4th
    "F":  {"RH": [1,2,3,4,1,2,3,1,2,3,4,1,2,3,4], "LH": [5,4,3,2,1,3,2,1,4,3,2,1,3,2,1]},
    # Bb major - starts with 2
    "Bb": {"RH": [2,1,2,3,1,2,3,4,1,2,3,1,2,3,4], "LH": [3,2,1,4,3,2,1,3,2,1,4,3,2,1,3]},
    # Eb major - starts with 3
    "Eb": {"RH": [3,1,2,3,4,1,2,3,1,2,3,4,1,2,3], "LH": [3,2,1,4,3,2,1,3,2,1,4,3,2,1,3]},
    # Ab major - starts with 3-4
    "Ab": {"RH": [3,4,1,2,3,1,2,3,4,1,2,3,1,2,3], "LH": [3,2,1,4,3,2,1,3,2,1,4,3,2,1,3]},
    # Db major - starts with 2-3
    "Db": {"RH": [2,3,1,2,3,4,1,2,3,1,2,3,4,1,2], "LH": [3,2,1,4,3,2,1,3,2,1,4,3,2,1,3]},
}


class MajorScaleGenerator:
    """Generates random major scale exercises."""
    
    def __init__(self, pitch_range: int = 52):
        self.pitch_range = pitch_range
        self.scale_names = list(MAJOR_SCALES.keys())
    
    def generate(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """
        Generate a random major scale exercise.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            List of (column_index, is_black) tuples representing the score
        """
        # Pick a random scale
        scale_name = rng.choice(self.scale_names)
        scale_pattern = MAJOR_SCALES[scale_name]
        root_offset = SCALE_ROOTS[scale_name]
        
        # Pick a random starting octave (ensure we have room for the scale)
        # Each octave is 7 columns wide
        max_octave = (self.pitch_range - 7) // 7
        start_octave = rng.integers(1, max(2, max_octave))
        base_column = start_octave * 7 + root_offset
        
        # Ensure we stay in bounds
        if base_column + 7 >= self.pitch_range:
            base_column = self.pitch_range - 14  # Move back an octave
        if base_column < 0:
            base_column = 7
        
        # Generate score pattern
        length = rng.integers(5, 13)
        direction = 1
        
        if length > 4:
            reverse_at = rng.integers(2, length - 2)
        else:
            reverse_at = -1
        
        score_targets = []
        scale_index = 0
        
        for i in range(length):
            # Get note from scale pattern
            col_offset, is_black = scale_pattern[scale_index % 7]
            
            # Calculate octave adjustment
            octave_adjust = (scale_index // 7) * 7
            
            # Calculate final column
            column = base_column + col_offset + octave_adjust
            
            # Bounds check
            if 0 <= column < self.pitch_range:
                score_targets.append((column, is_black))
            
            # Reverse direction at specified point
            if i == reverse_at:
                direction *= -1
            
            scale_index += direction
            
            # Keep scale_index in valid range
            if scale_index < 0:
                scale_index = 0
                direction = 1
            elif scale_index > 13:  # Allow up to 2 octaves
                scale_index = 13
                direction = -1
        
        return score_targets


class SimpleScaleGenerator:
    """Generates simple white-key-only scales (original behavior)."""
    
    def __init__(self, pitch_range: int = 52):
        self.pitch_range = pitch_range
    
    def generate(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """Generate a simple ascending/descending white key pattern."""
        length = rng.integers(5, 13)
        start_note = rng.integers(10, self.pitch_range - 15)
        direction = 1
        current_note = start_note
        
        score_targets = []
        if length > 4:
            reverse_at = rng.integers(2, length - 2)
        else:
            reverse_at = -1

        for i in range(length):
            score_targets.append((current_note, 0))  # 0 for Natural
            
            if i == reverse_at:
                direction *= -1
            current_note += direction
        
        return score_targets
