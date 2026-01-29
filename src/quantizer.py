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


@dataclass
class QuantizationConfig:
    """Configuration for the quantization process."""
    tempo_bpm: float = 120.0
    time_signature: tuple = (4, 4) # beats per measure, beat note value quarter-note 
    grid_resolution: int = 16

    @property
    def beat_duration_sec(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def measure_duration_sec(self) -> float:
        beats_per_measure = self.time_signature[0]
        return beats_per_measure * self.beat_duration_sec

    @property
    def grid_duration_sec(self) -> float:
        subdivisions_per_beat = self.grid_resolution / 4
        return self.beat_duration_sec / subdivisions_per_beat


@dataclass
class QuantizedScore:
    """Complete quantized score ready for notation encoding."""
    notes: List[QuantizedNote]
    config: QuantizationConfig
    duration_sec: float
    num_measures: int
    original_num_notes: int = 0

    def get_notes_in_measure(self, measure: int) -> List[QuantizedNote]:
        return [n for n in self.notes if n.measure == measure]

    def __repr__(self) -> str:
        return (
            f"QuantizedScore({len(self.notes)} notes, "
            f"{self.num_measures} measures @ {self.config.tempo_bpm} BPM)"
        )


class Quantizer:
    """
    Converts raw note events to quantized musical notation.
    Define time grid based on tempo and resolution, snap each not eonset to nearest grid point, each note duration to nearest standard value, assign measure and beat numbers. 
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def quantize(self, transcription: TranscriptionResult) -> QuantizedScore:
        quantized_notes = []

        for note in transcription.notes:
            q_note = self._quantize_note(note)
            quantized_notes.append(q_note)
