"""
Chord generation module for piano fingering training.

Provides circle of fifths structure, chord types, and generators for:
- Block chord progressions (12 bars)
- Arpeggiated chord patterns (4 bars)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# Circle of Fifths
# =============================================================================

# Circle of fifths with sharp (clockwise) and flat (counter-clockwise) neighbors
# Also includes relative minor for each major key
CIRCLE_OF_FIFTHS = {
    "C":  {"sharp": "G",  "flat": "F",  "relative_minor": "Am"},
    "G":  {"sharp": "D",  "flat": "C",  "relative_minor": "Em"},
    "D":  {"sharp": "A",  "flat": "G",  "relative_minor": "Bm"},
    "A":  {"sharp": "E",  "flat": "D",  "relative_minor": "F#m"},
    "E":  {"sharp": "B",  "flat": "A",  "relative_minor": "C#m"},
    "B":  {"sharp": "F#", "flat": "E",  "relative_minor": "G#m"},
    "F#": {"sharp": "C#", "flat": "B",  "relative_minor": "D#m"},
    "C#": {"sharp": "G#", "flat": "F#", "relative_minor": "A#m"},
    "F":  {"sharp": "C",  "flat": "Bb", "relative_minor": "Dm"},
    "Bb": {"sharp": "F",  "flat": "Eb", "relative_minor": "Gm"},
    "Eb": {"sharp": "Bb", "flat": "Ab", "relative_minor": "Cm"},
    "Ab": {"sharp": "Eb", "flat": "Db", "relative_minor": "Fm"},
    "Db": {"sharp": "Ab", "flat": "Gb", "relative_minor": "Bbm"},
    "Gb": {"sharp": "Db", "flat": "Cb", "relative_minor": "Ebm"},
}

# Chromatic note to semitone offset from C
NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

# Semitone to (white_key_column_in_octave, is_black)
# Column layout: C=0, D=1, E=2, F=3, G=4, A=5, B=6
SEMITONE_TO_GRID = {
    0:  (0, 0),  # C
    1:  (0, 1),  # C#/Db
    2:  (1, 0),  # D
    3:  (1, 1),  # D#/Eb
    4:  (2, 0),  # E
    5:  (3, 0),  # F
    6:  (3, 1),  # F#/Gb
    7:  (4, 0),  # G
    8:  (4, 1),  # G#/Ab
    9:  (5, 0),  # A
    10: (5, 1),  # A#/Bb
    11: (6, 0),  # B
}

# =============================================================================
# Chord Types and Intervals
# =============================================================================

# Chord intervals in semitones from root
# Supports 3-note and 4-note voicings
CHORD_INTERVALS = {
    # Triads (3 notes)
    "major":     [0, 4, 7],
    "minor":     [0, 3, 7],
    "dim":       [0, 3, 6],
    "aug":       [0, 4, 8],
    "sus2":      [0, 2, 7],
    "sus4":      [0, 5, 7],
    
    # Seventh chords (4 notes)
    "maj7":      [0, 4, 7, 11],
    "dom7":      [0, 4, 7, 10],
    "min7":      [0, 3, 7, 10],
    "dim7":      [0, 3, 6, 9],
    "half_dim7": [0, 3, 6, 10],  # m7b5
    
    # Sixth chords (4 notes)
    "maj6":      [0, 4, 7, 9],
    "min6":      [0, 3, 7, 9],
    
    # Add chords (4 notes)
    "add9":      [0, 4, 7, 14],  # Root + major triad + 9th
}

# Diatonic chords for major keys (I, ii, iii, IV, V, vi, vii°)
# Each entry: (scale_degree, chord_quality)
DIATONIC_CHORDS_DEGREES = [
    (0, "major"),   # I
    (2, "minor"),   # ii
    (4, "minor"),   # iii
    (5, "major"),   # IV
    (7, "major"),   # V
    (9, "minor"),   # vi
    (11, "dim"),    # vii°
]

# Roman numeral names for reference
ROMAN_NUMERALS = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]


def get_key_root_semitone(key: str) -> int:
    """Get the semitone offset of a key's root note from C."""
    # Handle minor keys
    root = key.replace("m", "")
    return NOTE_TO_SEMITONE.get(root, 0)


def build_diatonic_chords(key: str) -> List[Tuple[int, str, str]]:
    """
    Build the 7 diatonic chords for a given major key.
    
    Returns:
        List of (root_semitone, chord_quality, roman_numeral)
    """
    key_root = get_key_root_semitone(key)
    chords = []
    for (degree, quality), numeral in zip(DIATONIC_CHORDS_DEGREES, ROMAN_NUMERALS):
        root_semitone = (key_root + degree) % 12
        chords.append((root_semitone, quality, numeral))
    return chords


def get_chord_notes_semitones(root_semitone: int, quality: str, inversion: int = 0) -> List[int]:
    """
    Get chord notes as semitones (0-11).
    
    Args:
        root_semitone: Root note as semitone offset from C
        quality: Chord quality (e.g., "major", "min7")
        inversion: 0=root position, 1=first inversion, etc.
        
    Returns:
        List of semitone values for the chord
    """
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["major"])
    notes = [(root_semitone + interval) % 12 for interval in intervals]
    
    # Apply inversion by rotating the list
    if inversion > 0:
        inversion = inversion % len(notes)
        notes = notes[inversion:] + notes[:inversion]
    
    return notes


def semitones_to_grid(semitones: List[int], octave: int = 3) -> List[Tuple[int, int]]:
    """
    Convert semitone values to grid positions.
    
    Args:
        semitones: List of semitone values (0-11)
        octave: Starting octave (0-indexed)
        
    Returns:
        List of (column, is_black) tuples
    """
    grid_positions = []
    current_octave = octave
    prev_semitone = -1
    
    for semitone in semitones:
        # Handle octave wrapping for ascending notes
        if semitone <= prev_semitone:
            current_octave += 1
        prev_semitone = semitone
        
        col_in_octave, is_black = SEMITONE_TO_GRID[semitone]
        column = current_octave * 7 + col_in_octave
        grid_positions.append((column, is_black))
    
    return grid_positions


def get_secondary_dominant(target_degree: int, key_root: int) -> Tuple[int, str]:
    """
    Get the secondary dominant (V of target chord).
    
    Args:
        target_degree: Target chord's scale degree in semitones
        key_root: Key root in semitones
        
    Returns:
        (root_semitone, quality) for the secondary dominant
    """
    # V of target = target root + 7 semitones (perfect 5th above)
    target_root = (key_root + target_degree) % 12
    secondary_v_root = (target_root + 7) % 12
    return (secondary_v_root, "major")  # Secondary dominants are major


def calculate_voice_leading_distance(chord_a: List[int], chord_b: List[int]) -> int:
    """
    Calculate total voice leading distance between two chords.
    Smaller distance = smoother voice leading.
    """
    if len(chord_a) != len(chord_b):
        # Handle different chord sizes by padding with repeated notes
        max_len = max(len(chord_a), len(chord_b))
        while len(chord_a) < max_len:
            chord_a = chord_a + [chord_a[-1]]
        while len(chord_b) < max_len:
            chord_b = chord_b + [chord_b[-1]]
    
    total_distance = 0
    for a, b in zip(sorted(chord_a), sorted(chord_b)):
        # Calculate minimum distance considering octave wrapping
        dist = min(abs(a - b), 12 - abs(a - b))
        total_distance += dist
    return total_distance


def choose_best_inversion(prev_chord: List[int], next_chord_root: int, 
                          next_chord_quality: str) -> int:
    """
    Choose the inversion that minimizes voice leading distance.
    
    Returns:
        Best inversion number (0, 1, or 2)
    """
    if not prev_chord:
        return 0
    
    num_notes = len(CHORD_INTERVALS.get(next_chord_quality, [0, 4, 7]))
    best_inversion = 0
    best_distance = float('inf')
    
    for inv in range(num_notes):
        candidate = get_chord_notes_semitones(next_chord_root, next_chord_quality, inv)
        distance = calculate_voice_leading_distance(prev_chord, candidate)
        if distance < best_distance:
            best_distance = distance
            best_inversion = inv
    
    return best_inversion


# =============================================================================
# Chord Progression Generator
# =============================================================================

@dataclass
class ChordEvent:
    """Represents a single chord in the progression."""
    root_semitone: int
    quality: str
    inversion: int
    grid_positions: List[Tuple[int, int]]
    roman_numeral: str
    beat: int


class ChordProgressionGenerator:
    """
    Generates block chord progressions for fingering training.
    
    Produces 12 bars of block chords implementing harmonic rules:
    - Pivot chord modulation (Rule 1)
    - Secondary dominants (Rule 2)
    - High harmonic rhythm (Rule 3)
    - Voice leading inversions (Rule 4)
    - 7th chords on V (Rule 5)
    - Sus chord resolutions (Rule 6)
    - Diminished transitions (Rule 7)
    - Smooth bass movement (Rule 9)
    - Standard cadences (Rule 10)
    - Position shifts (Rule 12)
    """
    
    def __init__(self, pitch_range: int = 52, bars: int = 12, beats_per_bar: int = 4, hand: int = 1):
        self.pitch_range = pitch_range
        self.hand = hand  # 1=RH, 2=LH
        self.bars = bars
        self.beats_per_bar = beats_per_bar
        self.total_beats = bars * beats_per_bar
        
        # Available starting keys (common keys for practice)
        self.starting_keys = ["C", "G", "D", "F", "Bb", "A", "E", "Eb", "Ab"]
    
    def generate(self, rng: np.random.Generator) -> List[Tuple[Tuple[int, int], ...]]:
        """
        Generate a chord progression.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            List of chord events, each as tuple of (column, is_black) positions
        """
        # Select starting key
        current_key = rng.choice(self.starting_keys)
        diatonic = build_diatonic_chords(current_key)
        
        progression = []
        beat = 0
        prev_chord_semitones = []
        chords_in_register = 0
        current_octave = rng.integers(1, 3) if self.hand == 2 else rng.integers(2, 4)
        
        while beat < self.total_beats:
            # === Rule 3: Harmonic rhythm (chord every 2 beats) ===
            chord_duration = 2
            
            # Choose chord degree
            degree_idx = self._choose_chord_degree(rng, beat, len(progression))
            root_semitone, quality, roman = diatonic[degree_idx]
            
            # === Rule 5: 50% chance of 7th on V ===
            if degree_idx == 4 and rng.random() < 0.5:
                quality = "dom7"
            
            # === Rule 2: Secondary dominants (15% chance) ===
            if rng.random() < 0.15 and degree_idx not in [0, 4]:
                sec_dom_root, sec_dom_qual = get_secondary_dominant(
                    DIATONIC_CHORDS_DEGREES[degree_idx][0],
                    get_key_root_semitone(current_key)
                )
                # Insert secondary dominant
                inv = choose_best_inversion(prev_chord_semitones, sec_dom_root, sec_dom_qual)
                sec_notes = get_chord_notes_semitones(sec_dom_root, sec_dom_qual, inv)
                grid_pos = semitones_to_grid(sec_notes, current_octave)
                
                if self._is_in_range(grid_pos):
                    progression.append(tuple(grid_pos))
                    prev_chord_semitones = sec_notes
                    beat += 1
                    chord_duration = 1  # Shorter duration for following chord
            
            # === Rule 6: Sus4 resolutions (10% chance) ===
            if rng.random() < 0.10 and quality == "major":
                # Play sus4 first, then resolve
                sus_quality = "sus4"
                inv = choose_best_inversion(prev_chord_semitones, root_semitone, sus_quality)
                sus_notes = get_chord_notes_semitones(root_semitone, sus_quality, inv)
                grid_pos = semitones_to_grid(sus_notes, current_octave)
                
                if self._is_in_range(grid_pos):
                    progression.append(tuple(grid_pos))
                    prev_chord_semitones = sus_notes
                    beat += 1
                    chord_duration = 1
            
            # === Rule 7: Use vii° as V substitute (10% chance) ===
            if degree_idx == 4 and rng.random() < 0.10:
                degree_idx = 6  # Switch to vii°
                root_semitone, quality, roman = diatonic[degree_idx]
            
            # === Rule 4: Voice leading inversions ===
            inversion = choose_best_inversion(prev_chord_semitones, root_semitone, quality)
            
            # Get chord notes
            chord_notes = get_chord_notes_semitones(root_semitone, quality, inversion)
            grid_positions = semitones_to_grid(chord_notes, current_octave)
            
            # === Rule 12: Position shifts ===
            chords_in_register += 1
            if chords_in_register > 6:
                # Force position change
                shift = rng.choice([-1, 1])
                new_octave = current_octave + shift
                if 1 <= new_octave <= 4:
                    current_octave = new_octave
                    grid_positions = semitones_to_grid(chord_notes, current_octave)
                    chords_in_register = 0
            
            # Add chord if in range
            if self._is_in_range(grid_positions):
                progression.append(tuple(grid_positions))
                prev_chord_semitones = chord_notes
            
            beat += chord_duration
            
            # === Rule 10: Cadences at phrase ends ===
            # Every 16 beats (4 bars), ensure we end on I
            bars_complete = beat // self.beats_per_bar
            if bars_complete % 4 == 0 and bars_complete > 0:
                # Force cadence: ii-V-I or just V-I
                if rng.random() < 0.5 and beat < self.total_beats - 4:
                    # ii chord
                    ii_root, ii_qual, _ = diatonic[1]
                    inv = choose_best_inversion(prev_chord_semitones, ii_root, ii_qual)
                    ii_notes = get_chord_notes_semitones(ii_root, ii_qual, inv)
                    grid_pos = semitones_to_grid(ii_notes, current_octave)
                    if self._is_in_range(grid_pos):
                        progression.append(tuple(grid_pos))
                        prev_chord_semitones = ii_notes
                        beat += 2
            
            # === Rule 1: Pivot modulation (5% chance per chord) ===
            if rng.random() < 0.05:
                # Move to neighboring key on circle of fifths
                direction = rng.choice(["sharp", "flat"])
                new_key = CIRCLE_OF_FIFTHS.get(current_key, {}).get(direction)
                if new_key and new_key in CIRCLE_OF_FIFTHS:
                    current_key = new_key
                    diatonic = build_diatonic_chords(current_key)
        
        return progression
    
    def _choose_chord_degree(self, rng: np.random.Generator, beat: int, 
                             progression_length: int) -> int:
        """Choose a chord degree with musical weighting."""
        # Weight towards common progressions
        # I, IV, V are most common
        weights = [0.25, 0.12, 0.10, 0.18, 0.20, 0.12, 0.03]  # I, ii, iii, IV, V, vi, vii°
        
        # Start and end on I
        if progression_length == 0:
            return 0
        
        return rng.choice(7, p=weights)
    
    def _is_in_range(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if all chord positions are within the grid range."""
        return all(0 <= col < self.pitch_range for col, _ in positions)


# =============================================================================
# Arpeggio Generator
# =============================================================================

class ArpeggioPattern(Enum):
    """Arpeggio pattern types (Rule 11)."""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    BROKEN = "broken"           # 1-3-2-3 or similar
    ALBERTI = "alberti"         # 1-3-2-3-1-3-2-3 bass pattern


class ArpeggioGenerator:
    """
    Generates arpeggiated chord patterns for fingering training.
    
    Produces 4 bars of arpeggios with:
    - Multiple arpeggio patterns (Rule 11)
    - Extended chords for richer arpeggios (Rule 8)
    """
    
    def __init__(self, pitch_range: int = 52, bars: int = 4, 
                 notes_per_beat: int = 4, beats_per_bar: int = 4, hand: int = 1):
        self.pitch_range = pitch_range
        self.hand = hand  # 1=RH, 2=LH
        self.bars = bars
        self.notes_per_beat = notes_per_beat
        self.beats_per_bar = beats_per_bar
        self.chord_gen = ChordProgressionGenerator(pitch_range, bars=bars, hand=hand)
        
        # Weight patterns
        self.patterns = list(ArpeggioPattern)
    
    def generate(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """
        Generate an arpeggiated progression.
        
        Args:
            rng: NumPy random generator
            
        Returns:
            List of (column, is_black) tuples representing individual notes
        """
        # Generate underlying chord progression (fewer chords for arpeggios)
        current_key = rng.choice(self.chord_gen.starting_keys)
        diatonic = build_diatonic_chords(current_key)
        
        notes = []
        current_octave = rng.integers(1, 3) if self.hand == 2 else rng.integers(2, 4)
        prev_chord_semitones = []
        
        # One chord per bar for arpeggios
        for bar in range(self.bars):
            # Choose chord
            degree_idx = rng.choice(7, p=[0.25, 0.12, 0.10, 0.18, 0.20, 0.12, 0.03])
            if bar == 0:
                degree_idx = 0  # Start on I
            
            root_semitone, quality, _ = diatonic[degree_idx]
            
            # === Rule 8: Extended chords for arpeggios (30% chance) ===
            if rng.random() < 0.30 and quality == "major":
                quality = "add9"
            elif rng.random() < 0.20 and quality == "major":
                quality = "maj7"
            elif rng.random() < 0.20 and quality == "minor":
                quality = "min7"
            
            # Voice leading
            inversion = choose_best_inversion(prev_chord_semitones, root_semitone, quality)
            chord_notes = get_chord_notes_semitones(root_semitone, quality, inversion)
            
            # === Rule 11: Choose arpeggio pattern ===
            pattern = rng.choice(self.patterns)
            
            # Generate notes for this bar
            bar_notes = self._arpeggiate_chord(chord_notes, pattern, current_octave, rng)
            
            # Convert to grid and add
            for semitone, octave in bar_notes:
                col_in_octave, is_black = SEMITONE_TO_GRID[semitone % 12]
                column = octave * 7 + col_in_octave
                if 0 <= column < self.pitch_range:
                    notes.append((column, is_black))
            
            prev_chord_semitones = chord_notes
        
        return notes
    
    def _arpeggiate_chord(self, chord_notes: List[int], pattern: ArpeggioPattern,
                          octave: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """
        Arpeggiate a chord according to a pattern.
        
        Returns:
            List of (semitone, octave) tuples
        """
        notes_needed = self.beats_per_bar * self.notes_per_beat
        result = []
        n = len(chord_notes)
        
        if pattern == ArpeggioPattern.ASCENDING:
            # Simple ascending, repeating through octaves
            for i in range(notes_needed):
                note_idx = i % n
                oct_adjust = i // n
                result.append((chord_notes[note_idx], octave + oct_adjust))
                
        elif pattern == ArpeggioPattern.DESCENDING:
            # Descending from top
            reversed_notes = chord_notes[::-1]
            for i in range(notes_needed):
                note_idx = i % n
                oct_adjust = i // n
                result.append((reversed_notes[note_idx], octave + 1 - oct_adjust))
                
        elif pattern == ArpeggioPattern.BROKEN:
            # Broken pattern: 0-2-1-2-0-2-1-2... (skip pattern)
            if n >= 3:
                broken_order = [0, 2, 1, 2]
            else:
                broken_order = [0, 1, 0, 1]
            for i in range(notes_needed):
                note_idx = broken_order[i % len(broken_order)]
                if note_idx >= n:
                    note_idx = note_idx % n
                oct_adjust = i // (len(broken_order) * 2)
                result.append((chord_notes[note_idx], octave + oct_adjust))
                
        elif pattern == ArpeggioPattern.ALBERTI:
            # Alberti bass: 0-2-1-2 (low-high-mid-high)
            if n >= 3:
                alberti_order = [0, 2, 1, 2]
            else:
                alberti_order = [0, 1, 0, 1]
            for i in range(notes_needed):
                note_idx = alberti_order[i % len(alberti_order)]
                if note_idx >= n:
                    note_idx = note_idx % n
                result.append((chord_notes[note_idx], octave))
        
        return result
