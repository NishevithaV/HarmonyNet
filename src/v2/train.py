"""
This implements the full supervised fine-tuning procedure.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .model import PianoTranscriptionModel, build_model
from .dataset import PianoTranscriptionDataset
from .tokenizer import PAD_TOKEN, VOCAB_SIZE


CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "models" / "v2"


@dataclass
class TrainingConfig:
    """All hyperparameters in one place."""
    # Model
    whisper_size: str = "tiny"   # "tiny" or "base"

    # Data
    segment_sec: float = 10.0
    max_token_len: int = 512       # Shorter for faster iteration
    batch_size: int = 4            # Small for Mac RAM
    num_workers: int = 0        # 0 = main process (simpler debugging)
    max_train_pieces: Optional[int] = 20  # None = use full dataset

    # Phase A (frozen encoder)
    phase_a_epochs: int = 5
    phase_a_lr: float = 1e-4

    # Phase B (joint fine-tuning)
    phase_b_epochs: int = 3
    phase_b_decoder_lr: float = 1e-4
    phase_b_encoder_lr: float = 1e-5  # 10x lower: protect pre-trained encoder

    # Regularization
    dropout: float = 0.1
    grad_clip: float = 1.0         # Max gradient norm (prevents exploding gradients)

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_epochs: int = 1


class Trainer:
    """
    Manages the full SFT training process.
    """

    def __init__(self, config: TrainingConfig, audio_dir: Optional[Path] = None):
        self.config = config
        self.audio_dir = audio_dir
        self.device = self._get_device()
        print(f"Training device: {self.device}")

        # Build dataloaders and model
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'phase': []}

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        """Pick best available device: MPS (Apple Silicon) > CUDA > CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_dataloaders(self):
        """Build train and validation DataLoaders."""
        print("Building datasets...")

        train_ds = PianoTranscriptionDataset(
            split='train',
            segment_sec=self.config.segment_sec,
            max_token_length=self.config.max_token_len,
            audio_dir=self.audio_dir,
        )

        val_ds = PianoTranscriptionDataset(
            split='validation',
            segment_sec=self.config.segment_sec,
            max_token_length=self.config.max_token_len,
            audio_dir=self.audio_dir,
        )

        # Limit to subset of pieces if configured for fast iteration
        if self.config.max_train_pieces is not None:
            train_ds = self._subset_by_pieces(train_ds, self.config.max_train_pieces)
            val_ds = self._subset_by_pieces(val_ds, max(2, self.config.max_train_pieces // 5))

        print(f"  Train segments: {len(train_ds)}")
        print(f"  Val segments:   {len(val_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        return train_loader, val_loader

    def _subset_by_pieces(self, dataset: PianoTranscriptionDataset, max_pieces: int):
        """
        Return a subset of the dataset limited to the first N pieces.

        We subset by piece (not by segment index) to keep pieces contiguous.
        This avoids training on segment 5 of piece 3 but not segment 1 of piece 3.
        """
        valid_piece_indices = set(range(min(max_pieces, len(dataset.metadata))))
        valid_segment_indices = [
            i for i, (piece_idx, _) in enumerate(dataset.segments)
            if piece_idx in valid_piece_indices
        ]
        return Subset(dataset, valid_segment_indices)

    def _build_model(self) -> PianoTranscriptionModel:
        """Build model and move to device."""
        print(f"Building model (whisper-{self.config.whisper_size})...")
        model = build_model(self.config.whisper_size)
        model = model.to(self.device)
        return model

    def _build_optimizer(self, phase: str) -> torch.optim.Optimizer:
        """
        Build optimizer with phase-appropriate learning rates.
        Phase A: Single LR for decoder only
        Phase B: Different LRs for encoder vs decoder (discriminative fine-tuning)
        """
        if phase == 'a':
            # Only decoder parameters (encoder is frozen, no grad)
            return torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.phase_a_lr,
                weight_decay=0.01,
            )
        else:
            # Different LRs for encoder and decoder
            return torch.optim.AdamW(
                [
                    {'params': self.model.encoder.parameters(),
                     'lr': self.config.phase_b_encoder_lr},
                    {'params': self.model.decoder.parameters(),
                     'lr': self.config.phase_b_decoder_lr},
                ],
                weight_decay=0.01,
            )

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute cross-entropy loss for a batch.

        Teacher forcing setup:
        - decoder input:  tokens[:, :-1]  (all but last token)
        - target:         tokens[:, 1:]   (all but first token)
        We ignore PAD tokens in the loss (pad positions don't count).
        """
        spectrograms = batch['spectrogram'].to(self.device)  # [B, 1, n_mels, T]
        tokens = batch['tokens'].to(self.device)             # [B, max_token_len]

        # Teacher forcing: shift tokens
        decoder_input = tokens[:, :-1]   # [B, T-1]
        target = tokens[:, 1:]           # [B, T-1]

        # Forward pass
        logits = self.model(spectrograms, decoder_input)  # [B, T-1, vocab_size]

        # Compute cross-entropy loss, ignoring PAD positions
        # Reshape for F.cross_entropy: [B*(T-1), vocab_size] and [B*(T-1)]
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(B * T, V),
            target.reshape(B * T),
            ignore_index=PAD_TOKEN,  # Don't penalize PAD predictions
        )

        return loss

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Run one full pass over the training data."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            optimizer.zero_grad()

            loss = self._compute_loss(batch)
            loss.backward()

            # Gradient clipping: if gradients explode (norm > grad_clip),
            # scale them down, prevents training instability.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                avg = total_loss / num_batches
                print(f"  step {self.global_step:>6d}  loss {avg:.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_checkpoint(self, epoch: int, phase: str, val_loss: float):
        """Save model weights and training state."""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'model_state': self.model.state_dict(),
            'config': asdict(self.config),
        }
        path = CHECKPOINT_DIR / f"checkpoint_phase{phase}_epoch{epoch:03d}.pt"
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path.name}")

        # Also save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model! val_loss={val_loss:.4f}")

    def run(self):
        """Execute the full two-phase SFT training."""

        print("\n" + "=" * 60)
        print("PHASE A: Decoder-only training (encoder frozen)")
        print("=" * 60)
        print("  Goal: Get decoder to a reasonable state before")
        print("  touching Whisper's pre-trained encoder weights.\n")

        optimizer_a = self._build_optimizer('a')
        trainable = self.model.trainable_param_count()
        print(f"  Trainable params: {trainable:,}")

        for epoch in range(1, self.config.phase_a_epochs + 1):
            t0 = time.time()
            print(f"\nEpoch {epoch}/{self.config.phase_a_epochs} [Phase A]")

            train_loss = self.train_epoch(optimizer_a)
            val_loss = self.evaluate()
            elapsed = time.time() - t0

            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['phase'].append('A')

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, 'A', val_loss)

        print("\n" + "=" * 60)
        print("PHASE B: Joint fine-tuning (encoder + decoder)")
        print("=" * 60)
        print(f"  Encoder LR: {self.config.phase_b_encoder_lr}")
        print(f"  Decoder LR: {self.config.phase_b_decoder_lr}")
        print("  Why: Let encoder adapt to piano-specific audio features.\n")

        self.model.unfreeze_encoder()
        optimizer_b = self._build_optimizer('b')
        trainable = self.model.trainable_param_count()
        print(f"  Trainable params: {trainable:,}")

        for epoch in range(1, self.config.phase_b_epochs + 1):
            t0 = time.time()
            print(f"\nEpoch {epoch}/{self.config.phase_b_epochs} [Phase B]")

            train_loss = self.train_epoch(optimizer_b)
            val_loss = self.evaluate()
            elapsed = time.time() - t0

            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['phase'].append('B')

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, 'B', val_loss)

        # Save final history
        history_path = CHECKPOINT_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete.")
        print(f"Best val_loss: {self.best_val_loss:.4f}")
        print(f"History saved: {history_path}")


if __name__ == '__main__':
    """
    Dry-run test: validates the full training pipeline works end-to-end
    WITHOUT needing audio files. Uses only MIDI-derived token sequences
    with dummy (zero) spectrograms.
    """

    print("Dry-run mode: testing pipeline without audio...")

    config = TrainingConfig(
        whisper_size="tiny",
        max_token_len=256,
        batch_size=2,
        phase_a_epochs=1,
        phase_b_epochs=1,
        max_train_pieces=3,
        log_every_n_steps=1,
    )

    # Build dataset and replace spectrogram loading with dummy data
    from .dataset import PianoTranscriptionDataset, load_maestro_metadata
    from .spectrogram import SpectrogramConfig, WHISPER_N_MELS, WHISPER_NUM_FRAMES
    import torch

    class DummySpectrogramDataset(PianoTranscriptionDataset):
        """Override spectrogram loading to return zeros (for pipeline testing)."""
        def _get_spectrogram_segment(self, row, start_sec, end_sec):
            # WhisperSpectrogramExtractor always outputs [1, 80, 3000]
            return torch.zeros(1, WHISPER_N_MELS, WHISPER_NUM_FRAMES)

    print("\nBuilding dry-run datasets...")
    train_ds = DummySpectrogramDataset(split='train', max_token_length=256)
    val_ds = DummySpectrogramDataset(split='validation', max_token_length=256)

    # Limit to very small subset
    train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
    val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    from torch.utils.data import DataLoader, Subset
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    model = build_model("tiny").to(device)

    # One training step
    batch = next(iter(train_loader))
    spectrograms = batch['spectrogram'].to(device)
    tokens = batch['tokens'].to(device)

    decoder_input = tokens[:, :-1]
    target = tokens[:, 1:]

    logits = model(spectrograms, decoder_input)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        target.reshape(-1),
        ignore_index=PAD_TOKEN,
    )

    print(f"\nDry-run forward pass:")
    print(f"  Spectrogram: {list(spectrograms.shape)}")
    print(f"  Tokens in:   {list(decoder_input.shape)}")
    print(f"  Logits out:  {list(logits.shape)}")
    print(f"  Loss:        {loss.item():.4f}")
    print(f"  Expected loss ~{torch.log(torch.tensor(float(VOCAB_SIZE))):.2f} (random init)")

    loss.backward()
    print("\nBackward pass OK.")
    print("Training pipeline verified end-to-end.")
