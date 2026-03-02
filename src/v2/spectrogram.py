"""
Convert audio to mel spectrogram tensors for model input
Note: a mel scale compresses high frequencies and expands low ones,
matching how we perceive pitch.

TWO CONFIGS:
- SpectrogramConfig: our original 256-mel config (used by V1 pipeline)
- WhisperSpectrogramConfig: matches Whisper's exact preprocessing

WHY TWO CONFIGS?
Whisper was pre-trained with specific parameters (16kHz, 80 mels, hop=160,
3000 frames for 30s). Using different parameters means the encoder receives
inputs it has never seen during pre-training, which degrades SFT quality.

For V2 SFT training, we use WhisperSpectrogramConfig to match Whisper's
expected input format exactly. Whisper pads/truncates all input to 3000
frames regardless of segment length.
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

FRAME_DURATION_SEC = HOP_LENGTH / SAMPLE_RATE  # ~23.2ms

# Whisper's exact parameters (must match what the encoder was pre-trained on)
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MELS = 80
WHISPER_NUM_FRAMES = 3000  # 30 seconds × 100 frames/sec


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


class WhisperSpectrogramExtractor:
    """
    Produces log-mel spectrograms in Whisper's exact expected format.

    Whisper requires:
    - sample_rate: 16000 Hz
    - n_mels: 80
    - hop_length: 160 (→ 100 frames per second)
    - n_fft: 400
    - length: exactly 3000 frames (pad/truncate to 30 seconds)

    This extractor enforces all of these constraints so the Whisper
    encoder receives input in the same format it was pre-trained on.
    The output shape is always [1, 80, 3000] regardless of segment length.
    """

    def __init__(self):
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=WHISPER_SAMPLE_RATE,
            n_fft=WHISPER_N_FFT,
            hop_length=WHISPER_HOP_LENGTH,
            n_mels=WHISPER_N_MELS,
            power=2.0,
        )

    def from_waveform(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Compute Whisper-format log-mel from a waveform.

        Args:
            waveform: [1, num_samples] or [num_samples]
            orig_sr: Original sample rate of waveform

        Returns:
            [1, 80, 3000] log-mel spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample to 16000 Hz
        if orig_sr != WHISPER_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_sr, WHISPER_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute mel spectrogram
        mel = self._transform(waveform)  # [1, 80, T]
        log_mel = torch.log(mel + 1e-9)

        # Pad or truncate to exactly WHISPER_NUM_FRAMES (3000)
        T = log_mel.shape[2]
        if T < WHISPER_NUM_FRAMES:
            log_mel = torch.nn.functional.pad(log_mel, (0, WHISPER_NUM_FRAMES - T))
        else:
            log_mel = log_mel[:, :, :WHISPER_NUM_FRAMES]

        return log_mel  # [1, 80, 3000]


def extract_spectrogram(audio_path: str | Path) -> torch.Tensor:
    """Convenience function for one-shot extraction (V1 pipeline)."""
    extractor = MelSpectrogramExtractor()
    return extractor.from_file(audio_path)
