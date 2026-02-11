"""
MEI file score generator for piano fingering training.

Uses music21 to parse MEI files and converts them to the (column, is_black)
grid format used by the PianoEnv. Also provides utilities for writing
fingering annotations back to MEI files.
"""

import os
import numpy as np
import music21
from typing import List, Tuple, Optional
from pathlib import Path


# Pitch name to white key column within an octave
# Grid layout: C=0, D=1, E=2, F=3, G=4, A=5, B=6
STEP_TO_COL = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6,
}

# Which white keys have a black key (sharp) above them
# C#, D#, F#, G#, A# exist; E# and B# don't (they're white keys)
KEYS_WITH_SHARPS = {"C", "D", "F", "G", "A"}


def pitch_to_grid(pitch: music21.pitch.Pitch) -> Tuple[int, int]:
    """
    Convert a music21 Pitch to grid (column, is_black).
    
    The grid has 7 columns per octave (white keys: C D E F G A B).
    Black keys share the column of the white key below them with is_black=1.
    
    Args:
        pitch: A music21 Pitch object
        
    Returns:
        (column, is_black) tuple for the grid
    """
    step = pitch.step  # "C", "D", etc.
    octave = pitch.octave
    accidental = pitch.accidental
    
    col_in_octave = STEP_TO_COL[step]
    column = octave * 7 + col_in_octave
    
    is_black = 0
    if accidental is not None:
        alter = accidental.alter  # +1 for sharp, -1 for flat
        if alter == 1:
            # Sharp: this note is a black key above the natural
            is_black = 1
        elif alter == -1:
            # Flat: this note is a black key above the note below
            # e.g., Db is black key on column C (col_in_octave for C)
            # We need to map to the column below: Db -> C column (is_black=1)
            # Fb -> E (but that's actually a white key... edge case)
            # Gb -> F# column = F column with is_black
            prev_step_map = {"D": "C", "E": "D", "G": "F", "A": "G", "B": "A"}
            if step in prev_step_map:
                prev_step = prev_step_map[step]
                col_in_octave = STEP_TO_COL[prev_step]
                column = octave * 7 + col_in_octave
                is_black = 1
            elif step == "C":
                # Cb = B of the previous octave (white key)
                column = (octave - 1) * 7 + STEP_TO_COL["B"]
                is_black = 0
            elif step == "F":
                # Fb = E (white key)
                column = octave * 7 + STEP_TO_COL["E"]
                is_black = 0
        elif alter == 2:
            # Double sharp: move up one white key
            next_step_map = {"C": "D", "D": "E", "F": "G", "G": "A", "A": "B"}
            if step in next_step_map:
                next_step = next_step_map[step]
                column = octave * 7 + STEP_TO_COL[next_step]
                is_black = 0
            elif step == "E":
                # E## = F# 
                column = octave * 7 + STEP_TO_COL["F"]
                is_black = 1
            elif step == "B":
                # B## = C# of next octave
                column = (octave + 1) * 7 + STEP_TO_COL["C"]
                is_black = 1
        elif alter == -2:
            # Double flat: move down one white key
            prev2_map = {"D": "C", "E": "D", "G": "F", "A": "G", "B": "A"}
            if step in prev2_map:
                column = octave * 7 + STEP_TO_COL[prev2_map[step]]
                is_black = 0
            elif step == "C":
                column = (octave - 1) * 7 + STEP_TO_COL["B"]
                is_black = 1  # Cbb = Bb
            elif step == "F":
                column = octave * 7 + STEP_TO_COL["D"]
                is_black = 1  # Fbb = Eb = D#
    
    return (column, is_black)


def parse_mei_file(
    path: str, 
    staff: int = 1
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Parse an MEI file and extract notes for a specific staff.
    
    Args:
        path: Path to the MEI file
        staff: Which staff to extract (1=treble/right hand, 2=bass/left hand)
        
    Returns:
        Tuple of:
        - List of (column, is_black) tuples for each note
        - List of note xml:id strings (for linking fingerings back)
    """
    score = music21.converter.parse(path)
    
    # staff index is 0-based in music21 parts
    part_idx = staff - 1
    if part_idx >= len(score.parts):
        raise ValueError(
            f"Staff {staff} not found in {path}. "
            f"File has {len(score.parts)} parts."
        )
    
    part = score.parts[part_idx]
    
    note_positions = []
    note_ids = []
    
    for element in part.flatten().notes:
        if isinstance(element, music21.note.Note):
            pos = pitch_to_grid(element.pitch)
            note_positions.append(pos)
            note_ids.append(element.id)
        elif isinstance(element, music21.chord.Chord):
            # For chords, create a tuple of tuples (matches env chord format)
            chord_positions = []
            chord_ids = []
            for p in element.pitches:
                chord_positions.append(pitch_to_grid(p))
            # Chord ID is the chord element's ID
            note_positions.append(tuple(chord_positions))
            note_ids.append(element.id)
    
    return note_positions, note_ids


class MEIScoreGenerator:
    """
    Score generator that loads MEI files and yields their notes.
    
    Compatible with the PianoEnv score generator interface.
    Loads MEI files from a path (single file or directory) and
    returns their notes sequentially via generate().
    
    Args:
        mei_path: Path to a single .mei file or a directory of .mei files
        staff: Which staff to extract (1=treble, 2=bass)
        loop: If True, loop through files when exhausted; if False, 
              replay the last file's notes
    """
    
    def __init__(
        self, 
        mei_path: str, 
        staff: int = 1, 
        loop: bool = True,
        pitch_range: int = 52,
    ):
        self.staff = staff
        self.loop = loop
        self.pitch_range = pitch_range
        
        # Discover MEI files
        mei_path = Path(mei_path)
        if mei_path.is_file():
            self.mei_files = [str(mei_path)]
        elif mei_path.is_dir():
            self.mei_files = sorted(
                str(f) for f in mei_path.glob("*.mei")
            )
        else:
            raise FileNotFoundError(f"MEI path not found: {mei_path}")
        
        if not self.mei_files:
            raise FileNotFoundError(
                f"No .mei files found in {mei_path}"
            )
        
        # Parse and cache all files
        self._scores: List[List[Tuple[int, int]]] = []
        self._note_ids: List[List[str]] = []
        
        for fp in self.mei_files:
            positions, ids = parse_mei_file(fp, staff=self.staff)
            self._scores.append(positions)
            self._note_ids.append(ids)
        
        self._current_idx = 0
    
    @property
    def num_files(self) -> int:
        return len(self.mei_files)
    
    @property
    def current_file(self) -> str:
        return self.mei_files[self._current_idx]
    
    @property
    def current_note_ids(self) -> List[str]:
        """Note IDs for the current file (useful for annotation)."""
        return self._note_ids[self._current_idx]
    
    def generate(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """
        Return the notes from the current MEI file, then advance.
        
        The rng parameter is accepted for interface compatibility but
        is not used (MEI files are deterministic).
        
        Returns:
            List of (column, is_black) tuples representing the score
        """
        score = self._scores[self._current_idx]
        
        # Advance to next file
        if self.loop:
            self._current_idx = (self._current_idx + 1) % len(self._scores)
        else:
            # Stay on last file when exhausted
            self._current_idx = min(
                self._current_idx + 1, len(self._scores) - 1
            )
        
        return score
    
    def reset_index(self):
        """Reset file index to the beginning."""
        self._current_idx = 0
    
    def get_score_for_file(self, file_idx: int) -> List[Tuple[int, int]]:
        """Get the parsed score for a specific file index."""
        return self._scores[file_idx]
    
    def get_note_ids_for_file(self, file_idx: int) -> List[str]:
        """Get note IDs for a specific file index."""
        return self._note_ids[file_idx]
