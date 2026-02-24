**WHAT IS A SPECTROGRAM?**

Raw audio is a 1D signal of amplitude over time. A neural network could process this directly, but it would need to learn frequency decomposition from scratch.

A spectrogram pre-computes the frequency content, giving the model a 2D representation: frequency × time. Each pixel is how much energy is at this frequency at this time.
<br/>

**HOW IT'S COMPUTED (STFT → Mel → Log):**

Step 1: Short-Time Fourier Transform (STFT)
- Slide a window (e.g., 2048 samples) across the audio
- At each position, compute FFT (Fast Fourier Transform) to get frequency content from amplitude over time
- Result: linear-frequency spectrogram [freq_bins × time_frames]

The window size controls the time-frequency tradeoff:
- Larger window → better frequency resolution, worse time resolution
- Smaller window → better time resolution, worse frequency resolution
- 2048 samples at 22050Hz = ~93ms window. Good balance for music.

Step 2: Mel filterbank
- Human pitch perception is approximately logarithmic
- Low frequencies (200-400 Hz) sound "one octave apart"
- High frequencies (4000-8000 Hz) also sound "one octave apart"
- The mel scale compresses high frequencies and expands low ones
- We apply triangular filters to map linear freq bins to mel bins
- Result: mel spectrogram [n_mels × time_frames]

Why mel instead of CQT?
- torchaudio has fast GPU-accelerated mel spectrogram
- Most modern audio ML uses mel (Whisper, AudioLM, MusicGen)
- CQT is more music-theory-aligned, but mel works well in practice
- Using mel means skills transfer to speech/audio ML broadly

Step 3: Log compression
- Energy values span huge range (quiet = 0.001, loud = 1000)
- Take log to compress dynamic range: log(mel + epsilon)
- Neural networks work better with inputs in a bounded range
- Result: log-mel spectrogram [n_mels × time_frames]

**Waveform tensor**
- A full MAESTRO piece chunked into 10-sec segments are already each a tensor