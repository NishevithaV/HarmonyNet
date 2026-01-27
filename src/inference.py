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
        

    @property
    def model(self) -> Model:
        """
        Lazy-load the model on first access.
        TensorFlow Lite model wrapped by basic-pitch.
        """
        if self._model is None:
            # points to the pre-trained weights
            self._model = Model(ICASSP_2022_MODEL_PATH)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to note events. This is the main entry point and runs the full pipeline:
        """
        # Step 1: Load audio
        audio_data = load_audio(audio_path)

        # Step 2: Run inference
        notes = self._run_inference(audio_data)

        # Step 3: Package results
        return TranscriptionResult(
            notes=notes,
            duration_sec=audio_data.duration_sec,
            sample_rate=audio_data.original_sample_rate,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            min_note_length_sec=self.min_note_length,
        )

    def transcribe_from_array(
        self,
        samples: np.ndarray,
        sample_rate: int = MODEL_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """
        Transcribe from a numpy array. Useful for testing and streaming.
        """
        if sample_rate != MODEL_SAMPLE_RATE: # since model is trained on audio at 22,050 Hz
            raise ValueError(
                f"Sample rate must be {MODEL_SAMPLE_RATE}, got {sample_rate}. "
                "Resample before calling this method."
            )

        notes = self._run_inference_on_array(samples)
        duration_sec = len(samples) / sample_rate

        return TranscriptionResult(
            notes=notes,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            min_note_length_sec=self.min_note_length,
        )

    def _run_inference(self, audio_data: AudioData) -> List[NoteEvent]:
        """
        Run the inference pipeline on loaded audio.
        basic_pitch.inference.predict() computes Harmonic CQT spectrogram, bacthes it, runs CNN forward pass on each batch, stitches predictions, applies thresholding and note extraction.
        note_events in the format: 
        (onset_time, end_time, pitch, velocity, confidence) tuples.
        """
        return self._run_inference_on_array(audio_data.samples)

    def _run_inference_on_array(self, samples: np.ndarray) -> List[NoteEvent]:
        """
        Calls basic-pitch's predict() function which returns a model_output. 
        Model_output is raw probability matrices (onset, note, contour)
        Converts note_events to NoteEvent format.
        """
        # predict() handles CQT, batching, inference, postprocessing
        model_output, midi_data, note_events = predict(
            audio_path_or_array=samples,
            model_or_model_path=self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.min_note_length,
            # These parameters control pitch bend detection (not needed for piano)
            melodia_trick=True,  # Helps with note tracking
            minimum_frequency=None,  # Use default (piano range)
            maximum_frequency=None,
        )

        # Convert to NoteEvent format
        notes = []
        for start_sec, end_sec, pitch, velocity, confidence in note_events:
            # Filter to piano range
            if not (PIANO_MIN_MIDI <= pitch <= PIANO_MAX_MIDI):
                continue

            # Velocity from basic-pitch 0-1 scale to 0-127
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
        audio_data = load_audio(audio_path)

        model_output, _, _ = predict(
            audio_path_or_array=audio_data.samples,
            model_or_model_path=self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.min_note_length,
        )

        # model_output is a dict with keys as below
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
