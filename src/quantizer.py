"""
Bridge gap between raw transcription in continuous time to discrete values for sheet music.
Tempo is estimated from note onsets using onset periodicity.
V1 uses a default or user-provided tempo.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import numpy as np

from .note_events import NoteEvent, TranscriptionResult, midi_to_note_name


class NoteValue(Enum):
    """Standard musical note durations as fractions of a whole note."""
    WHOLE = 1.0
    HALF = 0.5
    QUARTER = 0.25
    EIGHTH = 0.125
    SIXTEENTH = 0.0625
    THIRTY_SECOND = 0.03125

    # Dotted notes 1.5x duration
    DOTTED_HALF = 0.75
    DOTTED_QUARTER = 0.375
    DOTTED_EIGHTH = 0.1875

    # Triplets 2/3 of the regular value
    TRIPLET_QUARTER = 0.25 * (2/3) 
    TRIPLET_EIGHTH = 0.125 * (2/3)  


# Standard durations in beats for quantization matching
STANDARD_DURATIONS_BEATS = [
    (4.0, "whole"),
    (3.0, "dotted_half"),
    (2.0, "half"),
    (1.5, "dotted_quarter"),
    (1.0, "quarter"),
    (0.75, "dotted_eighth"),
    (0.5, "eighth"),
    (0.375, "dotted_sixteenth"),
    (0.25, "sixteenth"),
    (0.125, "thirty_second"),
]


@dataclass
class QuantizedNote:
    """
    A note with musical timing in beats instead of raw seconds.
    Velocity can range from 0 to 127. 
    """
    pitch: int
    measure: int
    beat: float
    duration_beats: float
    velocity: int

    onset_sec: float
    duration_sec: float

    @property
    def pitch_name(self) -> str:
        return midi_to_note_name(self.pitch)

    @property
    def duration_name(self) -> str:
        """Get approximate note value name."""
        for dur, name in STANDARD_DURATIONS_BEATS:
            if abs(self.duration_beats - dur) < 0.1:
                return name
        return f"{self.duration_beats:.2f}_beats"

    def __repr__(self) -> str:
        return (
            f"QuantizedNote({self.pitch_name}, "
            f"m{self.measure} b{self.beat:.2f}, "
            f"dur={self.duration_name})"
        )