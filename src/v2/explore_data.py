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