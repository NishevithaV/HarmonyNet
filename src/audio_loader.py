"""
Handles the first pipeline stage for getting audio from disk into a format for the ML model.
Sampling rate quality is at 44, 100 Hz but we downsample to 22,050 Hz for model input by the Nyquist theorem (sample_rate/2 > highest frequency).
Only mono audio is needed. mono = (left + right) / 2 for stereo files.
Scale amplitude to [-1, 1] for consistent model behavior across recordings.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import librosa

MODEL_SAMPLE_RATE = 22050
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

@dataclass
class AudioData:
    """Container for loaded audio data."""
    samples: np.ndarray          
    sample_rate: int             
    original_sample_rate: int    # original file's sample rate
    duration_sec: float
    file_path: Path

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"AudioData({self.file_path.name}, "
            f"{self.duration_sec:.2f}s @ {self.sample_rate}Hz, "
            f"{self.num_samples:,} samples)"
        )


def load_audio(
    file_path: str | Path,
    target_sr: int = MODEL_SAMPLE_RATE,
    normalize: bool = True,
) -> AudioData:
    """
    Load an audio file and prepare it for inference.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}")

    original_sr = librosa.get_samplerate(file_path)

    samples, sr = librosa.load(file_path, sr=target_sr, mono=True)

    if normalize:
        samples = normalize_audio(samples)

    duration_sec = len(samples) / sr

    return AudioData(
        samples=samples,
        sample_rate=sr,
        original_sample_rate=original_sr,
        duration_sec=duration_sec,
        file_path=file_path,
    )


def normalize_audio(samples: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """
    Peak normalization to ensure all audio uses full [-1, 1] range.
    """
    peak = np.max(np.abs(samples))
    if peak < 1e-8:  # Avoid division by zero for silent audio
        return samples
    return samples * (target_peak / peak) 


def get_audio_info(file_path: str | Path) -> dict:
    """Get metadata without fully loading the file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sr = librosa.get_samplerate(file_path)
    duration = librosa.get_duration(path=file_path)

    # Check channel count
    y, _ = librosa.load(file_path, sr=None, mono=False, duration=0.1)
    num_channels = 1 if y.ndim == 1 else y.shape[0]

    return {
        'sample_rate': sr,
        'duration_sec': duration,
        'num_channels': num_channels,
        'format': file_path.suffix.lower(),
    }