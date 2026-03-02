"""
PyTorch Dataset for piano transcription training
This module produces (spectrogram_chunk, token_sequence) pairs from MAESTRO.
"""

import csv
from pathlib import Path
from typing import List, Optional
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from .tokenizer import (
    encode_notes, MidiNote, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    PIANO_MIN, PIANO_MAX, VOCAB_SIZE,
)
from .spectrogram import WhisperSpectrogramExtractor, SpectrogramConfig, WHISPER_SAMPLE_RATE

import pretty_midi


# Default segment duration in seconds
DEFAULT_SEGMENT_SEC = 10.0

# From EDA: ~5.9 tokens per note, ~10s segment at dense passages ≈ 200 notes ≈ 1200 tokens
# Use 2048 as a safe upper bound with padding
MAX_TOKEN_LENGTH = 2048

MAESTRO_DIR = Path(__file__).parent.parent.parent / "data" / "maestro" / "maestro-v3.0.0"
CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"


def load_maestro_metadata(split: str = 'train') -> List[dict]:
    """Load MAESTRO metadata CSV, filtered to a specific split."""
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row['split'] == split]


class PianoTranscriptionDataset(Dataset):
    """
    PyTorch Dataset that produces (spectrogram, tokens, token_length) tuples.
    Each item is a fixed-duration segment of a MAESTRO performance.
    DataLoader wraps this for batching, shuffling, and parallel loading
    """

    def __init__(
        self,
        split: str = 'train',
        segment_sec: float = DEFAULT_SEGMENT_SEC,
        max_token_length: int = MAX_TOKEN_LENGTH,
        spec_config: SpectrogramConfig = None,
        audio_dir: Optional[Path] = None,
    ):
        
        self.split = split
        self.segment_sec = segment_sec
        self.max_token_length = max_token_length
        self.spec_config = spec_config or SpectrogramConfig()
        self.audio_dir = audio_dir or MAESTRO_DIR

        self.extractor = WhisperSpectrogramExtractor()
        self.metadata = load_maestro_metadata(split)

        # Build segment index: list of (piece_idx, start_sec) tuples
        # This is the mapping from dataset index → specific segment
        self.segments = self._build_segments()

        # Cache MIDI data in memory (MIDI files are small, ~50KB each)
        self._midi_cache = {}

    def _build_segments(self) -> List[tuple]:
        """
        Compute all (piece_idx, start_sec) pairs.

        We skip the last segment if it would be shorter than half
        the segment duration to avoid very short segments.
        """
        segments = []
        for i, row in enumerate(self.metadata):
            duration = float(row['duration'])
            start = 0.0
            while start + self.segment_sec * 0.5 < duration:
                segments.append((i, start))
                start += self.segment_sec

        return segments

    # returns total number of segments across all pieces
    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example.
        """
        piece_idx, start_sec = self.segments[idx]
        row = self.metadata[piece_idx]
        end_sec = start_sec + self.segment_sec

        # Get spectrogram for this segment
        spectrogram = self._get_spectrogram_segment(row, start_sec, end_sec)

        # Get token sequence for notes in this segment
        tokens, token_length = self._get_token_segment(row, start_sec, end_sec)

        piece_name = f"{row['canonical_composer']} - {row['canonical_title']}"

        return {
            'spectrogram': spectrogram,
            'tokens': tokens,
            'token_length': token_length,
            'piece_name': piece_name,
        }

    def _get_spectrogram_segment(
        self, row: dict, start_sec: float, end_sec: float
    ) -> torch.Tensor:
        """
        Extract spectrogram for a time segment.

        Steps:
        1. Query native sample rate (torchaudio.info — no decode cost)
        2. Load just the segment at native sample rate
        3. Pass to WhisperSpectrogramExtractor, which handles:
           - resampling to 16kHz
           - mono conversion
           - log-mel computation
           - pad/truncate to exactly [1, 80, 3000]

        frame_offset and num_frames must use the file's native sample
        rate, not Whisper's 16kHz, so we query it first.
        """
        audio_path = self.audio_dir / row['audio_filename']

        # Get native sample rate without decoding any audio frames
        info = torchaudio.info(str(audio_path))
        native_sr = info.sample_rate

        # Compute frame indices in native sample rate
        start_sample = int(start_sec * native_sr)
        num_samples = int(self.segment_sec * native_sr)

        # Load just this segment (efficient partial decode)
        waveform, sr = torchaudio.load(
            str(audio_path),
            frame_offset=start_sample,
            num_frames=num_samples,
        )

        # WhisperSpectrogramExtractor handles resampling to 16kHz,
        # mono conversion, and padding/truncation to [1, 80, 3000]
        return self.extractor.from_waveform(waveform, sr)

    def _get_token_segment(
        self, row: dict, start_sec: float, end_sec: float
    ) -> tuple:
        """
        Get token sequence for notes within a time segment.

        Steps:
        1. Load MIDI file (cached)
        2. Filter notes with onset in [start_sec, end_sec)
        3. Shift note times so start_sec becomes 0 (relative timing)
        4. Encode to tokens
        5. Pad or truncate to max_token_length
        """
        midi_path = MAESTRO_DIR / row['midi_filename']

        # Load MIDI (with caching)
        notes = self._load_midi_notes(midi_path)

        # Filter to segment and shift timing
        segment_notes = []
        for n in notes:
            if start_sec <= n.start < end_sec:
                # Shift so segment starts at t=0
                shifted_end = min(n.end, end_sec) - start_sec
                segment_notes.append(MidiNote(
                    pitch=n.pitch,
                    start=n.start - start_sec,
                    end=shifted_end,
                    velocity=n.velocity,
                ))

        # Encode to tokens
        tokens = encode_notes(segment_notes)

        # Record actual length (including BOS/EOS)
        token_length = len(tokens)

        # Pad or truncate
        if len(tokens) > self.max_token_length:
            # Truncate but keep EOS at the end
            tokens = tokens[:self.max_token_length - 1] + [EOS_TOKEN]
            token_length = self.max_token_length
        elif len(tokens) < self.max_token_length:
            # Pad with PAD_TOKEN
            tokens = tokens + [PAD_TOKEN] * (self.max_token_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), token_length

    def _load_midi_notes(self, midi_path: Path) -> List[MidiNote]:
        """Load and cache MIDI notes from file."""
        key = str(midi_path)
        if key not in self._midi_cache:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            notes = []
            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                for n in inst.notes:
                    if PIANO_MIN <= n.pitch <= PIANO_MAX:
                        notes.append(MidiNote(n.pitch, n.start, n.end, n.velocity))
            notes.sort(key=lambda n: (n.start, n.pitch))
            self._midi_cache[key] = notes

        return self._midi_cache[key]


def create_dataloaders(
    batch_size: int = 8,
    segment_sec: float = DEFAULT_SEGMENT_SEC,
    num_workers: int = 0,
    audio_dir: Optional[Path] = None,
) -> dict:
    """
    Create train/validation/test DataLoaders.

    Args:
        batch_size: Number of segments per batch
        segment_sec: Segment duration
        num_workers: Parallel data loading workers (0 = main process)
        audio_dir: Path to MAESTRO audio directory
    """
    loaders = {}

    for split in ['train', 'validation', 'test']:
        dataset = PianoTranscriptionDataset(
            split=split,
            segment_sec=segment_sec,
            audio_dir=audio_dir,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Shuffle only training data
            num_workers=num_workers,
            # Custom collate not needed: all tensors are already padded to same size
        )

    return loaders


if __name__ == '__main__':
    """Test dataset construction (MIDI-only, no audio needed)."""
    print("Building segment index...")

    for split in ['train', 'validation', 'test']:
        metadata = load_maestro_metadata(split)
        total_duration = sum(float(r['duration']) for r in metadata)

        # Count segments
        segments = []
        for i, row in enumerate(metadata):
            dur = float(row['duration'])
            start = 0.0
            while start + DEFAULT_SEGMENT_SEC * 0.5 < dur:
                segments.append((i, start))
                start += DEFAULT_SEGMENT_SEC

        print(f"\n  {split}:")
        print(f"    Pieces: {len(metadata)}")
        print(f"    Total audio: {total_duration/3600:.1f} hours")
        print(f"    Segments ({DEFAULT_SEGMENT_SEC}s each): {len(segments)}")
        print(f"    Batches (batch_size=8): {len(segments) // 8}")

    # Test token generation for a few segments (no audio needed in this scope)
    print("\n\nTesting token generation (MIDI only)...")
    ds = PianoTranscriptionDataset.__new__(PianoTranscriptionDataset)
    ds.segment_sec = DEFAULT_SEGMENT_SEC
    ds.max_token_length = MAX_TOKEN_LENGTH
    ds.spec_config = SpectrogramConfig()
    ds._midi_cache = {}
    ds.metadata = load_maestro_metadata('train')
    ds.segments = []
    for i, row in enumerate(ds.metadata[:3]):  # First 3 pieces
        ds.segments.append((i, 0.0))  # First segment only

    for seg_idx in range(min(3, len(ds.segments))):
        piece_idx, start_sec = ds.segments[seg_idx]
        row = ds.metadata[piece_idx]
        tokens, length = ds._get_token_segment(row, start_sec, start_sec + DEFAULT_SEGMENT_SEC)

        print(f"\n  Piece: {row['canonical_composer']} - {row['canonical_title']}")
        print(f"  Segment: {start_sec:.0f}s - {start_sec + DEFAULT_SEGMENT_SEC:.0f}s")
        print(f"  Tokens: {length} (padded to {len(tokens)})")
        print(f"  Token utilization: {length / MAX_TOKEN_LENGTH * 100:.1f}%")
