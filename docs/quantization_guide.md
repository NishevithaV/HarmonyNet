#### Music is organized hierarchically:

Measure (bar)
    └── Beat (quarter note in 4/4)
          └── Subdivision (8th, 16th, triplet) beat divided into smaller notes 
                └── Tick (finest resolution)

For 4/4 time at 120 BPM:
- 1 measure = 4 beats = 2.0 seconds
- 1 beat = 0.5 seconds
- 1 eighth note = 0.25 seconds
- 1 sixteenth note = 0.125 seconds

**Example:**
  At 120 BPM (beat = 0.5s):
    Raw onset: 1.237s
    Grid points: 1.125s (16th), 1.250s (16th), 1.375s (16th)
    Nearest: 1.250s
    Quantized: beat 2.5 of measure 1 (the "and" of beat 2)

Note durations are also quantized. 0.487s will be 0.5s. 
