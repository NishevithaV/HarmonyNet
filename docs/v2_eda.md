MAESTRO DATASET: 
Each row contains:
- canonical_composer: Composer name
- canonical_title: Piece title
- split: 'train', 'validation', or 'test'
- year: Competition year (2004-2018)
- midi_filename: Relative path to MIDI file
- audio_filename: Relative path to WAV file
- duration: Duration in seconds

1. PITCH VOCABULARY
    - Full range: 88 keys (MIDI 21-108)
    - Effective range (90%): {p95-p5} keys (MIDI {p5}-{p95})
    - Use all 88 keys. The vocabulary cost is tiny and edge cases matter.

2. VELOCITY BINS
    - Raw range: 0-127 (128 values)
    - Quantize to 32 bins (4 velocity units per bin).
    - Human perception can't distinguish finer than ~4 MIDI velocity units.
    - This cuts vocabulary size without losing musical information.

3. TIME RESOLUTION
    - Smallest meaningful IOI: {p5_ioi*1000:.1f}ms
    - 10ms time resolution covers 95% of cases.
    - We'll use TIME_SHIFT tokens in 10ms increments up to some max.

4. SEQUENCE LENGTH
    - Mean notes per piece: {nc.mean():.0f}
    - With timing tokens, sequence length ≈ 3-4x note count
    - Mean sequence: ~{int(nc.mean() * 3.5):,} tokens
    - We'll need to chunk long pieces into segments for training.