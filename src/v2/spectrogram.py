"""
Convert audio to mel spectrogram tensors for model input
Note: a mel scale compresses high frequencies and expands low ones, 
matching how we perceive pitch.

PARAMETERS chosen based on EDA:
- sample_rate: 22050 Hz (same as basic-pitch, covers piano range)
- n_fft: 2048 (window size, ~93ms, good freq resolution for piano)
- hop_length: 512 (step between windows, ~23ms, good time resolution)
- n_mels: 256 (number of mel frequency bins)
  Why 256? Piano spans ~7 octaves. 256 bins gives ~36 bins per octave,
  enough to distinguish adjacent semitones even in the upper range.
- fmin: 20 Hz (just below A0 = 27.5 Hz)
- fmax: 8000 Hz (well above C8 = 4186 Hz, captures harmonics)

OUTPUT SHAPE:
For a 10-second audio clip at 22050 Hz:
- Raw samples: 220,500
- After STFT with hop_length=512: 220500 / 512 ≈ 431 time frames
- Shape: [1, 256, 431] (channels, mel_bins, time_frames)

This is what the model's encoder will process.
"""

import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass


# Spectrogram configuration
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
F_MIN = 20.0
F_MAX = 8000.0

# Time resolution: HOP_LENGTH / SAMPLE_RATE = 512/22050 ≈ 0.0232 seconds per frame
FRAME_DURATION_SEC = HOP_LENGTH / SAMPLE_RATE  # ~23.2ms


@dataclass
class SpectrogramConfig:
    """All spectrogram parameters in one place."""
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS
    f_min: float = F_MIN
    f_max: float = F_MAX

    @property
    def frame_duration_sec(self) -> float:
        return self.hop_length / self.sample_rate

    def num_frames(self, duration_sec: float) -> int:
        """How many spectrogram frames for a given audio duration."""
        num_samples = int(duration_sec * self.sample_rate)
        return num_samples // self.hop_length + 1


class MelSpectrogramExtractor:
    """
    Extracts log-mel spectrograms from audio files or waveforms.

    This class pre-computes the mel filterbank (a matrix of triangular filters)
    once at initialization, then reuses it for all audio. The filterbank is
    determined entirely by the config parameters and doesn't change.
    """

    def __init__(self, config: SpectrogramConfig = None):
        self.config = config or SpectrogramConfig()

        # torchaudio.transforms.MelSpectrogram combines STFT + mel filterbank
        # in one operation. It's GPU-acceleratable.
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,  # Power spectrogram (energy, not amplitude)
        )

    def from_file(self, audio_path: str | Path) -> torch.Tensor:
        """
        Load audio file and compute log-mel spectrogram.
        Returns log-mel spectrogram tensor, shape [1, n_mels, time_frames]
        """
        audio_path = Path(audio_path)

        # waveform shape: [channels, num_samples]
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return self._compute_log_mel(waveform)

    def from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram from a waveform tensor.
        Returns log-mel spectrogram tensor, shape [1, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return self._compute_log_mel(waveform)

    def _compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Core computation: waveform → mel spectrogram → log compression.
        Returns log-mel spectrogram tensor, shape [1, n_mels, time_frames]
        """
        # Compute mel spectrogram (STFT + mel filterbank)
        mel_spec = self._transform(waveform)

        # Log compression to reduce dynamic range
        # 1e-9 is a common choice (~-180 dB, well below audible range)
        log_mel = torch.log(mel_spec + 1e-9)

        return log_mel

    # for spectrogram to tokenizer alignment
    def seconds_to_frames(self, seconds: float) -> int:
        """Convert time in seconds to spectrogram frame index."""
        return int(seconds / self.config.frame_duration_sec)

    # for tokenizer alignment in seconds 
    def frames_to_seconds(self, frames: int) -> float:
        """Convert spectrogram frame index to time in seconds."""
        return frames * self.config.frame_duration_sec


def extract_spectrogram(audio_path: str | Path) -> torch.Tensor:
    """Convenience function for one-shot extraction."""
    extractor = MelSpectrogramExtractor()
    return extractor.from_file(audio_path)
