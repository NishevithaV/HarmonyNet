"""
Evaluation for the V2 piano transcription model.

Measures note-level Precision / Recall / F1 against MIDI ground truth.

Note matching uses onset tolerance: a predicted note is a true positive
if it matches a ground-truth note on the same pitch AND its onset is
within `onset_tolerance_sec` (default 50ms, the mir_eval standard).

"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import argparse

import soundfile as sf
import torch
import pretty_midi

from .model import build_model, PianoTranscriptionModel
from .tokenizer import decode_tokens, MidiNote
from .dataset import load_maestro_metadata, MAESTRO_DIR
from .spectrogram import WhisperSpectrogramExtractor


CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "models" / "v2"


# ---------------------------------------------------------------------------
# Note matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_ref: int
    n_tp: int


def match_notes(
    pred: List[MidiNote],
    ref: List[MidiNote],
    onset_tolerance: float = 0.05,   # 50ms, standard in mir_eval
) -> MatchResult:
    """
    Count true positives by greedily matching predicted notes to ground truth.

    A predicted note matches a reference note if:
      - Same pitch (always required)
      - |pred.start - ref.start| <= onset_tolerance

    Each reference note can only be matched once (greedy, sorted by onset
    proximity).  This is the standard AMT (automatic music transcription)
    evaluation protocol used in papers like Hawthorne et al. 2018.

    Why 50ms? Human timing jitter in MIDI recordings is ~10-20ms; 50ms is
    generous enough to handle quantisation differences while still being
    musically meaningful (a 16th note at 120 BPM is ~125ms).
    """
    # Group reference notes by pitch for fast lookup
    ref_by_pitch: dict[int, list[MidiNote]] = {}
    for n in ref:
        ref_by_pitch.setdefault(n.pitch, []).append(n)

    # Track which reference notes have been consumed
    matched_ref: set[int] = set()
    tp = 0

    for p in pred:
        candidates = ref_by_pitch.get(p.pitch, [])
        best_idx = None
        best_dt = float('inf')
        for i, r in enumerate(candidates):
            ref_id = id(r)
            if ref_id in matched_ref:
                continue
            dt = abs(p.start - r.start)
            if dt <= onset_tolerance and dt < best_dt:
                best_dt = dt
                best_idx = i
                best_ref_id = ref_id
        if best_idx is not None:
            tp += 1
            matched_ref.add(best_ref_id)

    n_pred = len(pred)
    n_ref = len(ref)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / n_ref  if n_ref  > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return MatchResult(precision, recall, f1, n_pred, n_ref, tp)


# Ground-truth extraction from MIDI

def load_ground_truth(midi_path: Path, start_sec: float, end_sec: float) -> List[MidiNote]:
    """
    Load notes from a MIDI file within [start_sec, end_sec).

    Times are kept relative to start_sec so they align with model output
    (which always starts at t=0 for each segment).
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in pm.instruments:
        for n in instrument.notes:
            # Note must START within the window
            if start_sec <= n.start < end_sec:
                notes.append(MidiNote(
                    pitch=n.pitch,
                    start=n.start - start_sec,  # relative to segment start
                    end=min(n.end, end_sec) - start_sec,
                    velocity=n.velocity,
                ))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# Spectrogram extraction for a single audio segment

def load_spectrogram(
    audio_path: Path,
    start_sec: float,
    segment_sec: float,
    extractor: WhisperSpectrogramExtractor,
) -> torch.Tensor:
    """Load one segment from a WAV file and return a [1, 1, 80, 3000] tensor."""
    info = sf.info(str(audio_path))
    native_sr = info.samplerate
    start_sample = int(start_sec * native_sr)
    stop_sample  = start_sample + int(segment_sec * native_sr)

    audio_np, sr = sf.read(
        str(audio_path),
        start=start_sample,
        stop=stop_sample,
        dtype='float32',
        always_2d=True,
    )
    waveform = torch.from_numpy(audio_np.T)           # [channels, frames]
    spec = extractor.from_waveform(waveform, sr)      # [1, 80, 3000]
    return spec.unsqueeze(0)                          # [1, 1, 80, 3000]


# Evaluator

@dataclass
class EvalConfig:
    checkpoint:        Path
    split:             str   = 'validation'
    max_segments:      int   = 20          # cap to keep eval fast
    segment_sec:       float = 10.0
    onset_tolerance:   float = 0.05        # 50ms
    max_gen_tokens:    int   = 512
    temperature:       float = 1.0
    top_p:             float = 0.9
    whisper_size:      str   = 'tiny'


class Evaluator:
    def __init__(self, config: EvalConfig, audio_dir: Optional[Path] = None):
        self.config = config
        self.audio_dir = audio_dir or MAESTRO_DIR
        self.device = self._get_device()
        self.extractor = WhisperSpectrogramExtractor()
        self.model = self._load_model()

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> PianoTranscriptionModel:
        print(f"Loading model from {self.config.checkpoint}")
        model = build_model(self.config.whisper_size)
        ckpt = torch.load(self.config.checkpoint, map_location=self.device, weights_only=True)

        # Checkpoint stores: epoch, phase, global_step, val_loss, model_state, config
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        print(f"  Device: {self.device}")
        return model

    def _get_segments(self) -> list[dict]:
        """Return a list of segment dicts that have audio on disk."""
        metadata = load_maestro_metadata(self.config.split)
        segments = []
        for row in metadata:
            audio_path = self.audio_dir / row['audio_filename']
            if not audio_path.exists():
                continue
            midi_path = self.audio_dir / row['midi_filename']
            duration = float(row['duration'])
            start = 0.0
            while start + self.config.segment_sec * 0.5 < duration:
                segments.append({
                    'audio_path': audio_path,
                    'midi_path':  midi_path,
                    'start_sec':  start,
                    'end_sec':    start + self.config.segment_sec,
                    'piece':      f"{row['canonical_composer']} - {row['canonical_title']}",
                })
                start += self.config.segment_sec
                if len(segments) >= self.config.max_segments:
                    return segments
        return segments

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run model on up to max_segments validation segments and compute
        aggregate note-level Precision / Recall / F1.
        """
        segments = self._get_segments()
        if not segments:
            raise RuntimeError(
                f"No audio found for split='{self.config.split}' in {self.audio_dir}. "
                "Download audio files first."
            )
        print(f"\nEvaluating on {len(segments)} segments (split={self.config.split})")
        print(f"  Onset tolerance: {self.config.onset_tolerance*1000:.0f}ms\n")

        all_results: List[MatchResult] = []

        for i, seg in enumerate(segments):
            # 1. Load spectrogram
            spec = load_spectrogram(
                seg['audio_path'],
                seg['start_sec'],
                self.config.segment_sec,
                self.extractor,
            ).to(self.device)

            # 2. Generate token sequence autoregressively
            token_ids = self.model.generate(
                spec,
                max_length=self.config.max_gen_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            # 3. Decode tokens → predicted notes
            pred_notes = decode_tokens(token_ids)

            # 4. Load ground-truth notes from MIDI
            ref_notes = load_ground_truth(
                seg['midi_path'],
                seg['start_sec'],
                seg['end_sec'],
            )

            # 5. Match
            result = match_notes(pred_notes, ref_notes, self.config.onset_tolerance)
            all_results.append(result)

            print(
                f"  [{i+1:>3}/{len(segments)}] "
                f"P={result.precision:.3f}  R={result.recall:.3f}  F1={result.f1:.3f}  "
                f"pred={result.n_pred}  ref={result.n_ref}  tp={result.n_tp}"
                f"  | {seg['piece'][:50]}"
            )

        # Aggregate: micro-average (sum TP/pred/ref, then compute ratios)
        total_tp   = sum(r.n_tp   for r in all_results)
        total_pred = sum(r.n_pred for r in all_results)
        total_ref  = sum(r.n_ref  for r in all_results)

        micro_p  = total_tp / total_pred if total_pred > 0 else 0.0
        micro_r  = total_tp / total_ref  if total_ref  > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        # Macro-average (mean over segments)
        macro_f1 = sum(r.f1 for r in all_results) / len(all_results)

        print(f"\n{'='*60}")
        print(f"Micro  P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
        print(f"Macro  F1={macro_f1:.3f}  (mean over {len(all_results)} segments)")
        print(f"Total  pred={total_pred}  ref={total_ref}  tp={total_tp}")
        print(f"{'='*60}")

        return {
            'micro_precision': micro_p,
            'micro_recall':    micro_r,
            'micro_f1':        micro_f1,
            'macro_f1':        macro_f1,
            'n_segments':      len(all_results),
            'total_pred':      total_pred,
            'total_ref':       total_ref,
            'total_tp':        total_tp,
        }


# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 piano transcription model")
    parser.add_argument('--checkpoint', type=Path,
                        default=CHECKPOINT_DIR / 'best_model.pt',
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--split', default='validation',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--max-segments', type=int, default=20)
    parser.add_argument('--onset-tolerance', type=float, default=0.05,
                        help='Onset matching tolerance in seconds (default 0.05)')
    parser.add_argument('--max-gen-tokens', type=int, default=512,
                        help='Max tokens to generate per segment (default 512). '
                             'Reduce for faster eval; note: no KV cache so cost is O(n²).')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--whisper-size', default='tiny', choices=['tiny', 'base'])
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint=args.checkpoint,
        split=args.split,
        max_segments=args.max_segments,
        onset_tolerance=args.onset_tolerance,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        whisper_size=args.whisper_size,
    )
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
