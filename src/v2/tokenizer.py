"""
MIDI-like event tokenizer for piano transcription

Converts between two representations: MIDI notes (pitch, start, end, velocity) & Token sequences (flat list of integers)

TOKEN VOCABULARY DESIGN:
  PAD: Padding for batching variable-length seqs
  BOS: Beginning of sequence marker
  EOS: End of sequence marker
  NOTE_ON_21..108: One per piano key (A0 to C8)
  NOTE_OFF_21..108: One per piano key
  TIME_SHIFT_1..100: 10ms increments (10ms to 1000ms)
  SET_VELOCITY_1..32: Velocity bins (4 MIDI units per bin)

Note:
- Velocity is only emitted when it changes from the previous note
- Time shifts are relative (each one advances from current position)
- Chord notes share the same time position (no TIME_SHIFT between)
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path
import pretty_midi

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NUM_SPECIAL = 3

# Piano range
PIANO_MIN = 21   # A0
PIANO_MAX = 108  # C8
NUM_PIANO_KEYS = PIANO_MAX - PIANO_MIN + 1  # 88

# Note ON tokens: IDs 3..90 (maps to MIDI pitch 21..108)
NOTE_ON_OFFSET = NUM_SPECIAL  # 3
# To get token ID: NOTE_ON_OFFSET + (pitch - PIANO_MIN)
# To get pitch from token: (token_id - NOTE_ON_OFFSET) + PIANO_MIN

# Note OFF tokens: IDs 91..178
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_PIANO_KEYS  # 91

# Time shift tokens: IDs 179..278
# Each token represents (token_value * 10) milliseconds
# token_value ranges 1..100, so 10ms to 1000ms
TIME_SHIFT_OFFSET = NOTE_OFF_OFFSET + NUM_PIANO_KEYS  # 179
NUM_TIME_SHIFTS = 100
TIME_SHIFT_MS = 10  # Each unit = 10ms
MAX_TIME_SHIFT_MS = NUM_TIME_SHIFTS * TIME_SHIFT_MS  # 1000ms

# Velocity tokens: IDs 279..310
# Quantize velocity (0-127) into 32 bins
VELOCITY_OFFSET = TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS  # 279
NUM_VELOCITY_BINS = 32
VELOCITY_BIN_SIZE = 4  # 128 / 32 = 4 MIDI units per bin

# Total vocabulary size
VOCAB_SIZE = VELOCITY_OFFSET + NUM_VELOCITY_BINS  # 311


# token to id conversion helpers 
def note_on_token(pitch: int) -> int:
    """Convert MIDI pitch to NOTE_ON token ID."""
    assert PIANO_MIN <= pitch <= PIANO_MAX, f"Pitch {pitch} out of range"
    return NOTE_ON_OFFSET + (pitch - PIANO_MIN)


def note_off_token(pitch: int) -> int:
    """Convert MIDI pitch to NOTE_OFF token ID."""
    assert PIANO_MIN <= pitch <= PIANO_MAX, f"Pitch {pitch} out of range"
    return NOTE_OFF_OFFSET + (pitch - PIANO_MIN)


def time_shift_token(units: int) -> int:
    """
    Convert time shift units (1-100) to token ID.
    Each unit = 10ms. So units=50 means 500ms.
    """
    assert 1 <= units <= NUM_TIME_SHIFTS, f"Time shift {units} out of range"
    return TIME_SHIFT_OFFSET + (units - 1)


def velocity_token(velocity: int) -> int:
    """
    Convert MIDI velocity (0-127) to velocity bin token ID.
    Bin = ceil(velocity / 4), clamped to [1, 32].
    """
    bin_num = max(1, min(NUM_VELOCITY_BINS, (velocity + VELOCITY_BIN_SIZE - 1) // VELOCITY_BIN_SIZE))
    return VELOCITY_OFFSET + (bin_num - 1)


def token_to_str(token_id: int) -> str:
    """
    Convert token ID to human-readable string.
    Useful for debugging and inspecting model output.
    """
    if token_id == PAD_TOKEN:
        return "PAD"
    elif token_id == BOS_TOKEN:
        return "BOS"
    elif token_id == EOS_TOKEN:
        return "EOS"
    elif NOTE_ON_OFFSET <= token_id < NOTE_OFF_OFFSET:
        pitch = (token_id - NOTE_ON_OFFSET) + PIANO_MIN
        return f"NOTE_ON_{pitch}({_midi_name(pitch)})"
    elif NOTE_OFF_OFFSET <= token_id < TIME_SHIFT_OFFSET:
        pitch = (token_id - NOTE_OFF_OFFSET) + PIANO_MIN
        return f"NOTE_OFF_{pitch}({_midi_name(pitch)})"
    elif TIME_SHIFT_OFFSET <= token_id < VELOCITY_OFFSET:
        units = (token_id - TIME_SHIFT_OFFSET) + 1
        ms = units * TIME_SHIFT_MS
        return f"TIME_SHIFT_{ms}ms"
    elif VELOCITY_OFFSET <= token_id < VOCAB_SIZE:
        bin_num = (token_id - VELOCITY_OFFSET) + 1
        return f"SET_VELOCITY_{bin_num}"
    else:
        return f"UNKNOWN_{token_id}"


def _midi_name(pitch: int) -> str:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{names[pitch % 12]}{(pitch // 12) - 1}"


# Encoder: MIDI Notes → token sequence
@dataclass
class MidiNote:
    """Simplified note representation for tokenization."""
    pitch: int
    start: float   # seconds
    end: float      # seconds
    velocity: int   # 0-127


def encode_notes(notes: List[MidiNote]) -> List[int]:
    """
    Encode a list of MIDI notes into a token sequence and
    retrns a list of token IDs 

    Algorithm:
    1. Create an event list from all note-ons and note-offs
    2. Sort events by time (note-ons before note-offs at same time)
    3. Walk through events, emitting tokens:
       - TIME_SHIFT tokens to advance the clock
       - SET_VELOCITY tokens when velocity changes
       - NOTE_ON / NOTE_OFF tokens for note events
    """
    if not notes:
        return [BOS_TOKEN, EOS_TOKEN]

    # Step 1: Build event list
    # Sorting by (time, type) puts note_offs before note_ons at same time
    events = []
    for n in notes:
        if not (PIANO_MIN <= n.pitch <= PIANO_MAX):
            continue
        events.append((n.start, 1, n.pitch, n.velocity))  # note on
        events.append((n.end, 0, n.pitch, 0))            # note off

    # Sort: by time, then note_offs before note_ons at same time
    events.sort(key=lambda e: (e[0], e[1]))

    # Step 2: Walk through events and emit tokens
    tokens = [BOS_TOKEN]
    current_time = 0.0
    current_velocity_bin = -1  # Force first velocity emission

    for event_time, event_type, pitch, vel in events:
        # Emit TIME_SHIFT tokens if time has advanced
        time_delta_ms = int(round((event_time - current_time) * 1000))

        if time_delta_ms > 0:
            shift_tokens = _encode_time_shift(time_delta_ms)
            tokens.extend(shift_tokens)
            # Note: advance by the QUANTIZED amount, not the true event time. Decoder advances by this quantized amount.
            quantized_ms = sum(
                ((t - TIME_SHIFT_OFFSET) + 1) * TIME_SHIFT_MS
                for t in shift_tokens
            )
            current_time += quantized_ms / 1000.0

        if event_type == 1:  # note on
            # Emit velocity if changed
            vel_bin = max(1, min(NUM_VELOCITY_BINS, (vel + VELOCITY_BIN_SIZE - 1) // VELOCITY_BIN_SIZE))
            if vel_bin != current_velocity_bin:
                tokens.append(velocity_token(vel))
                current_velocity_bin = vel_bin

            tokens.append(note_on_token(pitch))

        else:  # note off
            tokens.append(note_off_token(pitch))

    tokens.append(EOS_TOKEN)
    return tokens


def _encode_time_shift(delta_ms: int) -> List[int]:
    """
    Encode a time delta (diff in time between two events) as one 
    or more TIME_SHIFT tokens.
    For deltas > 1000ms, we emit multiple tokens.
    """
    tokens = []
    remaining = delta_ms

    while remaining > 0:
        # Units of 10ms, capped at 100 (= 1000ms)
        units = min(NUM_TIME_SHIFTS, max(1, remaining // TIME_SHIFT_MS))
        tokens.append(time_shift_token(units))
        remaining -= units * TIME_SHIFT_MS

    return tokens


# Decoder: Token Sequence → MIDI Notes

def decode_tokens(tokens: List[int]) -> List[MidiNote]:
    """
    Decode a token sequence back into MIDI notes.
    This is the inverse of encode_notes(). The model's output goes through this function to 
    produce actual musical notes.
    Returns a list of MidiNote objects.
    """
    notes = []
    current_time = 0.0
    current_velocity = 64  # Default mezzo-forte
    active_notes = {}  # pitch to (start_time, velocity) object 

    for token in tokens:
        if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            continue

        elif NOTE_ON_OFFSET <= token < NOTE_OFF_OFFSET:
            # NOTE_ON
            pitch = (token - NOTE_ON_OFFSET) + PIANO_MIN

            # If this pitch is already active, end it first
            if pitch in active_notes:
                start, vel = active_notes.pop(pitch)
                if current_time > start:
                    notes.append(MidiNote(pitch, start, current_time, vel))

            active_notes[pitch] = (current_time, current_velocity)

        elif NOTE_OFF_OFFSET <= token < TIME_SHIFT_OFFSET:
            # NOTE_OFF
            pitch = (token - NOTE_OFF_OFFSET) + PIANO_MIN

            if pitch in active_notes:
                start, vel = active_notes.pop(pitch)
                if current_time > start:
                    notes.append(MidiNote(pitch, start, current_time, vel))

        elif TIME_SHIFT_OFFSET <= token < VELOCITY_OFFSET:
            # TIME_SHIFT
            units = (token - TIME_SHIFT_OFFSET) + 1
            current_time += units * TIME_SHIFT_MS / 1000.0

        elif VELOCITY_OFFSET <= token < VOCAB_SIZE:
            # SET_VELOCITY
            bin_num = (token - VELOCITY_OFFSET) + 1
            # Convert bin back to MIDI velocity (center of bin)
            current_velocity = min(127, bin_num * VELOCITY_BIN_SIZE - VELOCITY_BIN_SIZE // 2)

    # Close any notes still active at end of sequence
    for pitch, (start, vel) in active_notes.items():
        if current_time > start:
            notes.append(MidiNote(pitch, start, current_time, vel))

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# MIDI FILE to TOKENS
def encode_midi_file(midi_path: str | Path) -> List[int]:
    """
    Read a MIDI file and encode it to tokens.
    Convenience function for the full pipeline.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            all_notes.append(MidiNote(
                pitch=n.pitch,
                start=n.start,
                end=n.end,
                velocity=n.velocity,
            ))

    return encode_notes(all_notes)


def decode_to_midi_file(tokens: List[int], output_path: str | Path, tempo: float = 120.0):
    """
    Decode tokens and write to a MIDI file.
    Useful for listening to model output.
    """
    notes = decode_tokens(tokens)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for n in notes:
        midi_note = pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start,
            end=n.end,
        )
        piano.notes.append(midi_note)

    pm.instruments.append(piano)
    pm.write(str(output_path))


# Validation and debugging
def tokens_to_readable(tokens: List[int]) -> str:
    """
    Convert token sequence to human-readable string.
    Adds arrow separators for clarity. Useful for debugging.
    """
    return ' → '.join(token_to_str(t) for t in tokens)


def validate_roundtrip(midi_path: str | Path, tolerance_ms: float = 20.0) -> dict:
    """
    Test encode to decode roundtrip accuracy on a MIDI file.
    Encodes a MIDI file to tokens, decodes back to notes, and compares
    against the original and reports accuracy metrics.

    This is critical for verifying the tokenizer doesn't lose information.
    If roundtrip accuracy is low, the model can never do better than the
    tokenizer as it would be trained on lossy labels. 
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    original_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if PIANO_MIN <= n.pitch <= PIANO_MAX:
                original_notes.append(MidiNote(n.pitch, n.start, n.end, n.velocity))

    # Encode then decode
    tokens = encode_notes(original_notes)
    recovered_notes = decode_tokens(tokens)

    # For each original note, find a recovered note with
    # matching pitch and onset within tolerance
    tolerance_sec = tolerance_ms / 1000.0
    matched = 0
    unmatched_original = 0
    used_recovered = set()

    for orig in original_notes:
        found = False
        for i, rec in enumerate(recovered_notes):
            if i in used_recovered:
                continue
            if (rec.pitch == orig.pitch and
                    abs(rec.start - orig.start) < tolerance_sec):
                matched += 1
                used_recovered.add(i)
                found = True
                break
        if not found:
            unmatched_original += 1

    extra_recovered = len(recovered_notes) - len(used_recovered)

    precision = matched / len(recovered_notes) if recovered_notes else 0
    recall = matched / len(original_notes) if original_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'original_notes': len(original_notes),
        'recovered_notes': len(recovered_notes),
        'matched': matched,
        'missed': unmatched_original,
        'extra': extra_recovered,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_tokens': len(tokens),
        'tokens_per_note': len(tokens) / len(original_notes) if original_notes else 0,
    }


if __name__ == '__main__':
    """Quick test: encode a MAESTRO file, inspect tokens, validate roundtrip."""
    import csv

    MAESTRO_DIR = Path(__file__).parent.parent.parent / "data" / "maestro" / "maestro-v3.0.0"
    CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Pick first training file
    row = rows[0]
    midi_path = MAESTRO_DIR / row['midi_filename']

    print(f"File: {row['canonical_composer']} - {row['canonical_title']}")
    print(f"Path: {midi_path}")

    # Encode
    tokens = encode_midi_file(midi_path)
    print(f"\nEncoded: {len(tokens)} tokens")
    print(f"Vocab used: {len(set(tokens))} unique token IDs out of {VOCAB_SIZE}")

    # Show first 30 tokens
    print(f"\nFirst 30 tokens:")
    for t in tokens[:30]:
        print(f"  [{t:3d}] {token_to_str(t)}")

    # Roundtrip validation
    print(f"\nRoundtrip validation...")
    results = validate_roundtrip(midi_path)
    print(f"  Original notes: {results['original_notes']}")
    print(f"  Recovered notes: {results['recovered_notes']}")
    print(f"  Matched: {results['matched']}")
    print(f"  Missed: {results['missed']}")
    print(f"  Extra: {results['extra']}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Tokens per note: {results['tokens_per_note']:.1f}")
