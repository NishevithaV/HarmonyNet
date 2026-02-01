from .note_events import NoteEvent, TranscriptionResult
from .audio_loader import load_audio, AudioData
from .inference import PianoTranscriber, transcribe_audio
from .quantizer import (
    Quantizer,
    QuantizedNote,
    QuantizedScore,
    QuantizationConfig,
    quantize_transcription,
)
from .encoder import MusicXMLEncoder, encode_to_musicxml
from .renderer import PDFRenderer, render_to_pdf, is_musescore_available

__version__ = '0.1.0'

__all__ = [
    # Note representation
    'NoteEvent',
    'TranscriptionResult',
    # Audio loading
    'AudioData',
    'load_audio',
    # Inference
    'PianoTranscriber',
    'transcribe_audio',
    # Quantization
    'Quantizer',
    'QuantizedNote',
    'QuantizedScore',
    'QuantizationConfig',
    'quantize_transcription',
    # Encoding
    'MusicXMLEncoder',
    'encode_to_musicxml',
    # Rendering
    'PDFRenderer',
    'render_to_pdf',
    'is_musescore_available',
]
