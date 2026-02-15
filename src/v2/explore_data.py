"""
This script is to perform EDA (Exploratory Data Analysis) and understand the MAESTRO dataset for informed tokenizer design. 

Run: python -m src.v2.explore_data
"""

import csv
from pathlib import Path
from collections import Counter
import numpy as np
import pretty_midi

MAESTRO_DIR = Path(__file__).parent.parent.parent / "data" / "maestro" / "maestro-v3.0.0"
CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"


def load_metadata() -> list[dict]:
    """
    Load the MAESTRO CSV metadata.

    Each row contains:
    - canonical_composer: Composer name
    - canonical_title: Piece title
    - split: 'train', 'validation', or 'test'
    - year: Competition year (2004-2018)
    - midi_filename: Relative path to MIDI file
    - audio_filename: Relative path to WAV file
    - duration: Duration in seconds
    """
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        return list(reader)


def dataset_overview(metadata: list[dict]):
    """
    Print high-level statistics about the dataset.

    Training data available, split distribution and balance, and range of piece durations for seq length considerations.
    """
    print("=" * 60)
    print("MAESTRO v3.0.0 DATASET OVERVIEW")
    print("=" * 60)

    # Split distribution
    splits = Counter(row['split'] for row in metadata)
    total = len(metadata)
    print(f"\nTotal performances: {total}")
    for split, count in sorted(splits.items()):
        duration = sum(float(r['duration']) for r in metadata if r['split'] == split)
        hours = duration / 3600
        print(f"  {split:>12s}: {count:>4d} performances ({hours:.1f} hours)")

    # Duration distribution
    durations = [float(row['duration']) for row in metadata]
    print(f"\nDuration stats (seconds):")
    print(f"  Min: {min(durations):.0f}s ({min(durations)/60:.1f} min)")
    print(f"  Max: {max(durations):.0f}s ({max(durations)/60:.1f} min)")
    print(f"  Mean: {np.mean(durations):.0f}s ({np.mean(durations)/60:.1f} min)")
    print(f"  Median: {np.median(durations):.0f}s ({np.median(durations)/60:.1f} min)")

    # Composer distribution
    composers = Counter(row['canonical_composer'] for row in metadata)
    print(f"\nUnique composers: {len(composers)}")
    print("  Top 10:")
    for composer, count in composers.most_common(10):
        print(f"    {composer}: {count} performances")


def analyze_midi_file(midi_path: Path) -> dict:
    """
    Extract statistical properties from a single MIDI file.

    pretty_midi parses a MIDI file into a list of Instrument objects
    Each instrument has notes as Note objects with start, end, pitch, velocity

    For MAESTRO piano recordings, there's typically 1 instrument. Returns dict 
    with statistics for this file.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Collect all notes across all instruments
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)

    if not all_notes:
        return None

    # Sort by onset time
    all_notes.sort(key=lambda n: n.start)

    pitches = [n.pitch for n in all_notes]
    velocities = [n.velocity for n in all_notes]
    durations = [n.end - n.start for n in all_notes]

    # Inter-onset intervals (time between consecutive note starts) for note density and tempo 
    onsets = [n.start for n in all_notes]
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]

    return {
        'num_notes': len(all_notes),
        'pitches': pitches,
        'velocities': velocities,
        'durations': durations,
        'iois': iois,
        'total_duration': all_notes[-1].end,
    }


def analyze_dataset_sample(metadata: list[dict], max_files: int = 50):
    """
    Analyze a representative sample of MIDI files from the training set.
    """
    print("\n" + "=" * 60)
    print(f"MIDI ANALYSIS (sampling {max_files} training files)")
    print("=" * 60)

    train_rows = [r for r in metadata if r['split'] == 'train']

    # Take evenly spaced sample for diversity
    step = max(1, len(train_rows) // max_files)
    sample = train_rows[::step][:max_files]

    all_pitches = []
    all_velocities = []
    all_durations = []
    all_iois = []
    all_note_counts = []

    for i, row in enumerate(sample):
        midi_path = MAESTRO_DIR / row['midi_filename']
        if not midi_path.exists():
            continue

        stats = analyze_midi_file(midi_path)
        if stats is None:
            continue

        all_pitches.extend(stats['pitches'])
        all_velocities.extend(stats['velocities'])
        all_durations.extend(stats['durations'])
        all_iois.extend(stats['iois'])
        all_note_counts.append(stats['num_notes'])

        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i+1}/{len(sample)} files...")

    print(f"\n  Total notes analyzed: {len(all_pitches):,}")

    # pitch analysis
    print("\n*** PITCH DISTRIBUTION of piano range in performances ***")
    pitches = np.array(all_pitches)
    print(f"  Range: {pitches.min()} ({_midi_name(pitches.min())}) "
          f"to {pitches.max()} ({_midi_name(pitches.max())})")
    print(f"  Mean: {pitches.mean():.0f} ({_midi_name(int(pitches.mean()))})")

    # Percentiles show where most notes actually fall
    p5 = int(np.percentile(pitches, 5))
    p95 = int(np.percentile(pitches, 95))
    print(f"  5th-95th percentile: {p5} ({_midi_name(p5)}) "
          f"to {p95} ({_midi_name(p95)})")
    print(f"  → 90% of notes fall in a {p95-p5} semitone range")

    # velocity analysis
    print("\n*** VELOCITY DISTRIBUTION ***")
    print("  (This tells us how many velocity bins we need in our tokenizer)")
    vels = np.array(all_velocities)
    print(f"  Range: {vels.min()} to {vels.max()}")
    print(f"  Mean: {vels.mean():.0f}, Std: {vels.std():.0f}")

    # check if velocity is roughly uniform or clustered
    vel_bins = [0, 32, 64, 96, 128]
    vel_labels = ['pp (0-31)', 'p-mp (32-63)', 'mf-f (64-95)', 'ff (96-127)']
    vel_hist, _ = np.histogram(vels, bins=vel_bins)
    print(f"  Distribution across dynamics:")
    for label, count in zip(vel_labels, vel_hist):
        pct = count / len(vels) * 100
        bar = '#' * int(pct / 2)
        print(f"    {label:>15s}: {pct:5.1f}% {bar}")

    # duration analysis
    print("\n*** NOTE DURATION DISTRIBUTION ***")
    print("  (This tells us the time resolution our tokenizer needs)")
    durs = np.array(all_durations)
    print(f"  Range: {durs.min():.4f}s to {durs.max():.2f}s")
    print(f"  Mean: {durs.mean():.3f}s, Median: {np.median(durs):.3f}s")

    # duration buckets in seconds 
    dur_thresholds = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float('inf')]
    dur_labels = ['<50ms', '50-100ms', '100-250ms', '250-500ms',
                  '0.5-1s', '1-2s', '>2s']
    dur_hist, _ = np.histogram(durs, bins=dur_thresholds)
    print(f"  Duration buckets:")
    for label, count in zip(dur_labels, dur_hist):
        pct = count / len(durs) * 100
        bar = '#' * int(pct / 2)
        print(f"    {label:>12s}: {pct:5.1f}% {bar}")

    # inter-onset intervals (IOIs) analysis
    print("\n*** INTER-ONSET INTERVALS ***")
    print("  (Time between consecutive note starts: affects TIME_SHIFT granularity)")
    iois = np.array(all_iois)
    # Filter out zero IOIs (simultaneous notes in chords)
    nonzero_iois = iois[iois > 0.001]
    print(f"  Simultaneous notes (chords): {(iois <= 0.001).sum() / len(iois) * 100:.1f}%")
    print(f"  Non-zero IOI median: {np.median(nonzero_iois):.4f}s")
    print(f"  Non-zero IOI mean: {np.mean(nonzero_iois):.4f}s")

    p5_ioi = np.percentile(nonzero_iois, 5)
    p95_ioi = np.percentile(nonzero_iois, 95)
    print(f"  5th-95th percentile: {p5_ioi:.4f}s to {p95_ioi:.4f}s")

    # notes per piece
    print("\n*** NOTES PER PIECE ***")
    print("  (This determines output sequence length for the model)")
    nc = np.array(all_note_counts)
    print(f"  Range: {nc.min()} to {nc.max()}")
    print(f"  Mean: {nc.mean():.0f}, Median: {np.median(nc):.0f}")

    # summar for tokenizer design 
    print("\n" + "=" * 60)
    print("IMPLICATIONS FOR TOKENIZER DESIGN")
    print("=" * 60)
    print(f"""
""")


def _midi_name(midi_pitch: int) -> str:
    """Convert MIDI pitch to note name."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_pitch // 12) - 1
    note = names[midi_pitch % 12]
    return f"{note}{octave}"


def inspect_single_midi(metadata: list[dict]):
    """
    Look at the raw content of one MIDI file in detail to show what the model will have to learn to predict. 
    """
    print("\n" + "=" * 60)
    print("Single MIDI File Inspection")
    print("=" * 60)

    # Well-known piece for demonstration
    row = metadata[0]  # First training file
    midi_path = MAESTRO_DIR / row['midi_filename']

    print(f"\n  File: {row['midi_filename']}")
    print(f"  Composer: {row['canonical_composer']}")
    print(f"  Title: {row['canonical_title']}")
    print(f"  Duration: {float(row['duration']):.0f}s")

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    print(f"\n  Instruments: {len(pm.instruments)}")
    for i, inst in enumerate(pm.instruments):
        print(f"    [{i}] program={inst.program}, "
              f"is_drum={inst.is_drum}, "
              f"notes={len(inst.notes)}")

    # Show first 20 notes used as ground truth by model 
    notes = pm.instruments[0].notes
    notes.sort(key=lambda n: n.start)

    print(f"\n  First 20 notes (what the model must learn to output):")
    print(f"  {'Start':>8s} {'End':>8s} {'Dur':>7s} {'Pitch':>5s} {'Note':>5s} {'Vel':>4s}")
    print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*5} {'-'*4}")
    for n in notes[:20]:
        dur = n.end - n.start
        name = _midi_name(n.pitch)
        print(f"  {n.start:8.3f} {n.end:8.3f} {dur:7.3f} {n.pitch:5d} {name:>5s} {n.velocity:4d}")

    # Show what chords look like 
    print(f"\n  Chord detection (notes starting within 30ms of each other):")
    chord_threshold = 0.030  # 30ms
    i = 0
    chord_count = 0
    # If groups of 2+ notes start within 30ms, we consider them a chord. Show first 5 chords in the piece. 
    # Needed for tokenizater to learn to predict multiple simultaneous notes.
    while i < len(notes) and chord_count < 5:
        chord = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j].start - notes[i].start < chord_threshold:
            chord.append(notes[j])
            j += 1

        if len(chord) > 1:
            chord_count += 1
            names = [_midi_name(n.pitch) for n in chord]
            print(f"    t={chord[0].start:.3f}s: {', '.join(names)} "
                  f"({len(chord)} notes)")

        i = j


if __name__ == '__main__':
    metadata = load_metadata()
    dataset_overview(metadata)
    inspect_single_midi(metadata)
    analyze_dataset_sample(metadata, max_files=50)