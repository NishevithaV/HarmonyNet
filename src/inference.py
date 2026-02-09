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

NOTE on basic-pitch 0.3.0 API:
- predict() takes audio_path (file path) instead of numpy arrays
- minimum_note_length is in milliseconds (default 127.7ms)
- ICASSP_2022_MODEL_PATH points to TF saved model; we use .onnx suffix
  for the ONNX backend which works on Python 3.12 + macOS
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

# Compatibility changes: scipy.signal.gaussian was moved to scipy.signal.windows.gaussian
# in scipy 1.14+. basic-pitch 0.3.0 references the old location.
import scipy.signal
if not hasattr(scipy.signal, 'gaussian'):
    from scipy.signal.windows import gaussian
    scipy.signal.gaussian = gaussian

from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

from .audio_loader import load_audio
from .note_events import NoteEvent, TranscriptionResult, PIANO_MIN_MIDI, PIANO_MAX_MIDI


DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MIN_NOTE_LENGTH = 0.05  # seconds (converted to ms for basic-pitch)

# Resolve ONNX model path from ICASSP_2022_MODEL_PATH
# ICASSP_2022_MODEL_PATH points to .../nmp (TF saved model)
# We need .../nmp.onnx for the ONNX runtime backend
ONNX_MODEL_PATH = Path(str(ICASSP_2022_MODEL_PATH) + '.onnx')


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

    @property
    def model(self) -> Model:
        """
        Lazy-load the model on first access.
        Uses ONNX backend for Python 3.12 compatibility on macOS.
        """
        if self._model is None:
            self._model = Model(ONNX_MODEL_PATH)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to note events. This is the main entry point and runs the full pipeline.
        basic-pitch 0.3.0 handles audio loading internally, so we pass the file path directly.
        We still load audio separately to get metadata (duration, sample rate).
        """
        audio_path = Path(audio_path)

        # Step 1: Load audio for metadata (duration, sample rate)
        audio_data = load_audio(audio_path)

        # Step 2: Run inference (basic-pitch loads audio internally from file path)
        notes = self._run_inference(audio_path)

        # Step 3: Package results
        return TranscriptionResult(
            notes=notes,
            duration_sec=audio_data.duration_sec,
            sample_rate=audio_data.original_sample_rate,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            min_note_length_sec=self.min_note_length,
        )

    def _run_inference(self, audio_path: Path) -> List[NoteEvent]:
        """
        Run the inference pipeline.
        basic_pitch.inference.predict() takes a file path, computes Harmonic CQT
        spectrogram, batches it, runs CNN forward pass on each batch, stitches
        predictions, applies thresholding and note extraction.
        Returns note_events as (start, end, pitch, velocity, pitch_bends) tuples.
        """
        # Convert min_note_length from seconds to milliseconds for basic-pitch API
        min_note_ms = self.min_note_length * 1000

        _, _, note_events = predict(
            audio_path=audio_path,
            model_or_model_path=self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=min_note_ms,
            melodia_trick=True,
            minimum_frequency=None,
            maximum_frequency=None,
        )

        # Convert to NoteEvent format
        # basic-pitch 0.3.0 note_events: (start_sec, end_sec, pitch, velocity, pitch_bends)
        notes = []
        for event in note_events:
            start_sec, end_sec, pitch, velocity = event[0], event[1], event[2], event[3]

            # Filter to piano range
            if not (PIANO_MIN_MIDI <= pitch <= PIANO_MAX_MIDI):
                continue

            # Velocity from basic-pitch is 0-1, scale to 0-127
            velocity_midi = int(np.clip(velocity * 127, 0, 127))

            duration = end_sec - start_sec
            if duration < self.min_note_length:
                continue

            notes.append(NoteEvent(
                pitch=int(pitch),
                onset_sec=float(start_sec),
                duration_sec=float(duration),
                velocity=velocity_midi,
            ))

        return notes

    def get_raw_predictions(
        self,
        audio_path: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get raw model output matrices for debugging and visualization.
        """
        audio_path = Path(audio_path)
        min_note_ms = self.min_note_length * 1000

        model_output, _, _ = predict(
            audio_path=audio_path,
            model_or_model_path=self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=min_note_ms,
        )

        return (
            model_output['onset'],
            model_output['note'],
            model_output['contour'],
        )


def transcribe_audio(
    audio_path: str | Path,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    min_note_length: float = DEFAULT_MIN_NOTE_LENGTH,
) -> TranscriptionResult:
    """
    Convenience function for one-shot transcription.
    (Note: For batch processing, create a PianoTranscriber once and reuse to avoid reloading model.)
    """
    transcriber = PianoTranscriber(
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_length=min_note_length,
    )
    return transcriber.transcribe(audio_path)