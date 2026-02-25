
**WHY CHUNKING IS NECESSARY:**
<br/>
Full pieces are 1-40 minutes long. That's 2,500-100,000+ spectrogram frames and 5,000-70,000+ tokens. No Transformer can process that in one pass:
- Self-attention is O(n²) in sequence length
- GPU memory is finite

So each piece is chunked into fixed-length segments:
- Spectrogram: [1, 256, num_frames] where num_frames = segment duration / frame_duration
- Tokens: corresponding token sequence for notes in that time window

**SEGMENT ALIGNMENT:**
<br/>
The spectrogram and token sequence must be aligned, the tokens describe
exactly the notes that occur during the spectrogram's time window.

For a 10-second segment starting at t=30s:
- Spectrogram: frames corresponding to audio from 30.0s to 40.0s
- Tokens: encode notes with onset in [30.0s, 40.0s), shifted so t=30s becomes t=0

This shift is important: the model learns to transcribe audio starting
from time 0, regardless of where in the piece the segment came from.

**DATASET SPLITS:**
<br/>
MAESTRO provides pre-defined train/validation/test splits at the piece level.<br/>
Chunk within each split. A piece in the test split will never appear in training, even partially.

**LAZY vs EAGER LOADING:**
<br/>
Precompute and cache spectrograms to disk (eager) because:
- Computing spectrograms on-the-fly is slow (I/O + STFT per batch)
- Spectrograms are deterministic (same audio → same spectrogram)
- Cache once, train many epochs without recomputation
<br/>
Token sequences are small enough to hold in memory (eager).