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

Key Findings from running `explore_data`: 
  1. Pitch: Full range 21-108 is used, but 90% of notes are in 40-87 (E2 to D#6)
  2. Velocity: Range 1-126, mostly in 32-95 (piano to forte), roughly bell-shaped distribution
  3. Duration: Median 89ms, 80% of notes are under 250ms. Long notes (>2s) are rare.
  4. IOI: 4.5% simultaneous (chords), median non-zero IOI is 43.8ms, 95th percentile is 349ms
  5. Notes per piece: Average around 5885 notes per piece, ranging from 1147 to 20559.
<br/>

**Tokenizer Design Takeaways:**
These patterns suggest that the full pitch range needs to be preserved despite the skew, and 32 velocity bins should work well. 
A 10ms time resolution captures the timing nuances, and sequences will need to be chunked since each piece generates roughly 20,000 tokens. 

- Velocity: 128 raw velcoity values is wasteful. Most notes cluster in the 32-95 range and can be quantized without losing information
- Pitch: 90% of notes fall in a 47-semitone range and extremes are rare 
- Duration: Median is 89ms. Long sustained notes over 2s are rare but still need to be considered 
- IOI: 4.5% of notes are chords so simulatenous within 1ms which needs to be handled 
- Seq Length: Must be chunked into segments for training so transformer attention is O(n^2)