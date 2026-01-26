"""
ML inference pipeline using basic-pitch. Model outputs multi-task predictions. 

Current implementation inference pipeline steps: 

1. Pre-processing raw audio into model input: 
   Raw audio → Harmonic CQT spectrogram → Tensor batches

2. Forward Pass model input to raw predictions:
   CNN processes spectrogram → 3 probability matrices (onset, note, contour)

3. Post-processing raw Predictions to structured output:
   Probability matrices → Thresholding → Note event extraction → NoteEvents

basic-pitch architecture and assumptions made for general music transcription: 

Input representation: Harmonic CQT (long-frequency spacing). Harmonic CQT stacks multiple CQTs at harmonic intervals (f, 2f, 3f, 4f, 5f) to distinguish harmonics from fundamental frequency of middle C.  
Shape: [time_frames, 264 frequency_bins, num_harmonics]

- Onset: P(note starts at this frame) for each of 88 piano keys
- Note: P(note is active at this frame) for each of 88 piano keys
- Contour: Fine-grained pitch estimate (not as relevant for pianos but still considered in this model)

- onset_threshold: If P(onset) > threshold, mark as note start
- frame_threshold: If P(note) > threshold, note is active

Default values (0.5, 0.3) are tuned for general piano transcription.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

from .audio_loader import AudioData, load_audio, MODEL_SAMPLE_RATE
from .note_events import NoteEvent, TranscriptionResult, PIANO_MIN_MIDI, PIANO_MAX_MIDI


DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MIN_NOTE_LENGTH = 0.05  


class PianoTranscriber:
    """
    Wrapper around basic-pitch for piano transcription. Handles model loading, audio preprocessing, inference, and post-processing predictions into NoteEvents. 
    """

    def __init__(
        self,
        onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
        frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
        min_note_length: float = DEFAULT_MIN_NOTE_LENGTH,
    ):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_length = min_note_length

        # Lazy model loading 
        self._model: Optional[Model] = None
