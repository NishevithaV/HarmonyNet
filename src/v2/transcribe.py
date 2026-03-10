"""
V2 inference: full audio file → TranscriptionResult

This is the bridge between the trained SFT model and the V1 downstream
pipeline (quantizer → MusicXML encoder → PDF renderer).

The V1 pipeline expects a TranscriptionResult containing NoteEvents.
The V2 model produces MidiNote objects from token sequences.
This module does the chunking, stitching, and type conversion.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from ..note_events import NoteEvent, TranscriptionResult
from .model import PianoTranscriptionModel, build_model
from .tokenizer import decode_tokens, MidiNote
from .spectrogram import WhisperSpectrogramExtractor, WHISPER_SAMPLE_RATE


DEFAULT_CHECKPOINT = Path(__file__).parent.parent.parent / "models" / "v2" / "best_model.pt"
SEGMENT_SEC = 10.0
MIN_NOTE_DURATION = 0.03  # drop notes shorter than 30ms (model artefacts)


def load_checkpoint(checkpoint_path: Path, whisper_size: str = "tiny") -> PianoTranscriptionModel:
    """Load a trained checkpoint and return the model in eval mode."""
    device = _get_device()
    model = build_model(whisper_size)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def transcribe_audio(
    audio_path: Path,
    checkpoint_path: Optional[Path] = None,
    whisper_size: str = "tiny",
    max_gen_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> TranscriptionResult:
    """
    This is the V2 drop-in replacement for src.inference.PianoTranscriber.transcribe().
    """
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Train the model first: python -m src.v2.train"
        )

    device = _get_device()
    extractor = WhisperSpectrogramExtractor()
    model = load_checkpoint(checkpoint_path, whisper_size)

    # Load full audio at native sample rate
    info = sf.info(str(audio_path))
    native_sr = info.samplerate
    total_frames = info.frames
    duration_sec = total_frames / native_sr

    all_notes: list[NoteEvent] = []
    chunk_sec = SEGMENT_SEC
    start_sec = 0.0

    print(f"  V2 transcription: {duration_sec:.1f}s audio, "
          f"{int(duration_sec / chunk_sec) + 1} chunks")

    while start_sec < duration_sec:
        end_sec = min(start_sec + chunk_sec, duration_sec)

        # Load chunk at native sample rate
        start_sample = int(start_sec * native_sr)
        stop_sample  = int(end_sec   * native_sr)
        audio_np, sr = sf.read(
            str(audio_path),
            start=start_sample,
            stop=stop_sample,
            dtype='float32',
            always_2d=True,
        )
        waveform = torch.from_numpy(audio_np.T)  # [channels, frames]

        # Spectrogram: [1, 1, 80, 3000]
        spec = extractor.from_waveform(waveform, sr).unsqueeze(0).to(device)

        # Generate token sequence autoregressively
        with torch.no_grad():
            token_ids = model.generate(
                spec,
                max_length=max_gen_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode tokens → notes (times are relative to this chunk's start)
        chunk_notes: list[MidiNote] = decode_tokens(token_ids)

        # Convert MidiNote → NoteEvent, offsetting times to absolute position
        for note in chunk_notes:
            abs_onset = note.start + start_sec
            abs_end   = note.end   + start_sec

            # Clamp to audio duration
            abs_onset = min(abs_onset, duration_sec)
            abs_end   = min(abs_end,   duration_sec)
            dur = abs_end - abs_onset

            if dur < MIN_NOTE_DURATION:
                continue

            try:
                all_notes.append(NoteEvent(
                    pitch=note.pitch,
                    onset_sec=abs_onset,
                    duration_sec=dur,
                    velocity=note.velocity,
                ))
            except ValueError:
                # NoteEvent validates pitch range — skip if out of range
                continue

        start_sec += chunk_sec

    # Deduplicate: if two notes at same pitch overlap in the same chunk boundary,
    # keep the one with earlier onset.
    all_notes = _deduplicate(all_notes)

    print(f"  Found {len(all_notes)} notes")

    return TranscriptionResult(
        notes=all_notes,
        duration_sec=duration_sec,
        sample_rate=WHISPER_SAMPLE_RATE,  # report at 16kHz (Whisper's rate)
    )


def _deduplicate(notes: list[NoteEvent]) -> list[NoteEvent]:
    """
    Remove duplicate notes at the same pitch with onset within 20ms of each other.
    Can happen at chunk boundaries (last note of chunk N ≈ first note of chunk N+1).
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.pitch, n.onset_sec))
    kept = [notes[0]]
    for n in notes[1:]:
        prev = kept[-1]
        if n.pitch == prev.pitch and abs(n.onset_sec - prev.onset_sec) < 0.02:
            continue  # duplicate across chunk boundary
        kept.append(n)
    return sorted(kept, key=lambda n: (n.onset_sec, n.pitch))


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")