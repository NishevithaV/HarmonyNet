"""
This module defines the core data structures used throughout the pipeline for musical note representation.
Formula used for MIDI pitch: MIDI_pitch = 12 * (octave + 1) + semitone_offset.
"""

from dataclasses import dataclass, field
from typing import List

# Semitone offsets 
SEMITONE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Alternative names using flats (for proper enharmonic spelling later)
SEMITONE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

PIANO_MIN_MIDI = 21   # A0
PIANO_MAX_MIDI = 108  # C8


def midi_to_note_name(midi_pitch: int, use_flats: bool = False) -> str:
    """
    Convert MIDI pitch number to note name string.
    """
    if not 0 <= midi_pitch <= 127:
        raise ValueError(f"MIDI pitch must be 0-127, got {midi_pitch}")

    # Decompose MIDI pitch into octave and semitone
    octave = (midi_pitch // 12) - 1  # -1 because MIDI octave 0 starts at C-1
    semitone = midi_pitch % 12

    names = SEMITONE_NAMES_FLAT if use_flats else SEMITONE_NAMES
    return f"{names[semitone]}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """
    Convert note name string to MIDI pitch number.
    """
    note_name = note_name.strip()

    base = note_name[0].upper()

    # Extract accidental and octave
    rest = note_name[1:]
    accidental = 0

    if rest.startswith('#') or rest.startswith('♯'):
        accidental = 1
        rest = rest[1:]
    elif rest.startswith('b') or rest.startswith('♭'):
        accidental = -1
        rest = rest[1:]

    octave = int(rest)

    # Base semitone offsets 
    base_semitones = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    if base not in base_semitones:
        raise ValueError(f"Invalid note name: {note_name}")

    semitone = base_semitones[base] + accidental
    midi_pitch = 12 * (octave + 1) + semitone

    return midi_pitch


@dataclass
class NoteEvent:
    """
    Represents a single musical note with timing and dynamics.
    Core unit of musical information used in pipeline. Each NoteEvent
    captures when a note starts, how long it lasts, its pitch, and velocity.
    """
    pitch: int
    onset_sec: float
    duration_sec: float
    velocity: int = 64  # Default to mezzo-forte (64/127)

    def __post_init__(self):
        """Validate note event data."""
        if not PIANO_MIN_MIDI <= self.pitch <= PIANO_MAX_MIDI:
            raise ValueError(
                f"Pitch {self.pitch} outside piano range ({PIANO_MIN_MIDI}-{PIANO_MAX_MIDI})"
            )
        if self.onset_sec < 0:
            raise ValueError(f"Onset time cannot be negative: {self.onset_sec}")
        if self.duration_sec <= 0:
            raise ValueError(f"Duration must be positive: {self.duration_sec}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Velocity must be 0-127: {self.velocity}")

    @property
    def pitch_name(self) -> str:
        """Get note name"""
        return midi_to_note_name(self.pitch)

    @property
    def offset_sec(self) -> float:
        """Get note end time"""
        return self.onset_sec + self.duration_sec

    @property
    def frequency_hz(self) -> float:
        """
        Get fundamental frequency in Hz.
        Formula used: f = 440 * 2^((midi - 69) / 12) based on A4 = 440 Hz (MIDI 69) as the reference pitch.
        """
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))

    def __repr__(self) -> str:
        return (
            f"NoteEvent({self.pitch_name}, "
            f"onset={self.onset_sec:.3f}s, "
            f"dur={self.duration_sec:.3f}s, "
            f"vel={self.velocity})"
        )


@dataclass
class TranscriptionResult:
    """
    Container for all note events plus metadata about the transcription process.
    Passed from the inference stage to the quantizer.
    """
    notes: List[NoteEvent]
    duration_sec: float  
    sample_rate: int     # Original audio sample rate

    # Inference metadata 
    onset_threshold: float = 0.5  
    frame_threshold: float = 0.3 
    min_note_length_sec: float = 0.05 

    def __post_init__(self):
        self.notes = sorted(self.notes, key=lambda n: (n.onset_sec, n.pitch))

    @property
    def num_notes(self) -> int:
        return len(self.notes)

    @property
    def pitch_range(self) -> tuple:
        if not self.notes:
            return (None, None)
        pitches = [n.pitch for n in self.notes]
        return (midi_to_note_name(min(pitches)), midi_to_note_name(max(pitches)))

    def get_notes_in_range(self, start_sec: float, end_sec: float) -> List[NoteEvent]:
        """Get all notes that start within a time range."""
        return [n for n in self.notes if start_sec <= n.onset_sec < end_sec]

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult({self.num_notes} notes, "
            f"{self.duration_sec:.1f}s, "
            f"range={self.pitch_range})"
        )
