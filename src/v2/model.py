"""
Encoder-decoder Transformer for piano transcription
"""

import math
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig

from .tokenizer import VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


# Whisper tiny config: d_model=384, encoder_layers=4, decoder_layers=4, heads=6
WHISPER_TINY = "openai/whisper-tiny"

# Whisper base: d_model=512, encoder_layers=6, decoder_layers=6, heads=8
WHISPER_BASE = "openai/whisper-base"


class TranscriptionDecoder(nn.Module):
    """
    Transformer decoder that maps encoder output → MIDI token sequence.

    This is a standard causal Transformer decoder:
    - Self-attention: each token attends to all previous tokens (causal mask)
    - Cross-attention: each token attends to all encoder frames
    - Feed-forward: position-wise transformation
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,       # Must match encoder d_model
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,         # Feed-forward hidden dim (typically 4x d_model)
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = PAD_TOKEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Token embedding: maps token ID → d_model vector
        # This is a lookup table: vocab_size rows, d_model columns
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

    
        # Unlike Whisper which uses sinusoidal, this uses learned embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, # model dimension 
            nhead=n_heads, # num of attention heads 
            dim_feedforward=d_ff, # tokens pass through 2-layer MLP 
            dropout=dropout, # zeros out activations for regularization
            batch_first=True, # [batch, seq, d_model] instead of [seq, batch, d_model]
            norm_first=True, # Pre-norm where LayerNorm is applied before attention and feed-forward 
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection: d_model → vocab logits
        # CrossEntropyLoss applies softmax internally
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Scale embeddings keep embedding magnitudes comparable to positional encodings
        self.embed_scale = math.sqrt(d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,   # [B, T_dec] token IDs
        encoder_out: torch.Tensor,   # [B, T_enc, d_model] encoder hidden states
        encoder_padding_mask: torch.Tensor = None,  # [B, T_enc] True where padded
    ) -> torch.Tensor:
        """
        Forward pass. Inputs token sequence, hidden encoder states to cross-attend to, and optional padding mask for encoder.
        """
        B, T = tokens.shape
        device = tokens.device

        # Positional indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        # token_embedding gives semantic meaning, pos_embedding gives position
        x = self.token_embedding(tokens) * self.embed_scale  # [B, T, d_model]
        x = x + self.pos_embedding(positions)                 # [B, T, d_model]
        x = self.dropout(x)

        # Causal mask: prevents attending to future tokens
        # Shape: [T, T], True means "ignore this position"
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding mask for decoder input (ignore PAD tokens)
        tgt_key_padding_mask = (tokens == self.pad_token_id)  # [B, T]

        # Transformer decoder: self-attn + cross-attn + feed-forward
        # x attends to itself (causal) and to encoder_out (cross-attention)
        x = self.transformer(
            tgt=x,
            memory=encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )  # [B, T, d_model]

        # Project to vocabulary
        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits


class PianoTranscriptionModel(nn.Module):
    """
    Full encoder-decoder model for piano transcription.
    Combines whisper encoder with custom decoder. 
    """

    def __init__(
        self,
        whisper_model_name: str = WHISPER_BASE,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        max_token_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load pre-trained Whisper encoder
        print(f"Loading Whisper encoder: {whisper_model_name}")
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper.encoder

        # Get encoder's d_model
        d_model = self.encoder.config.d_model
        print(f"Encoder d_model: {d_model}")

        # Build our decoder to match encoder's d_model
        self.decoder = TranscriptionDecoder(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            d_ff=decoder_ff_dim,
            max_seq_len=max_token_len,
            dropout=dropout,
        )

        # Start with encoder frozen (Phase A of SFT)
        self.freeze_encoder()

    def freeze_encoder(self):
        """
        Freeze encoder weights - won't update during Phase A.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False  # parameters unchanged 
        print("Encoder frozen")

    def unfreeze_encoder(self):
        """
        Unfreeze encoder for Phase B joint fine-tuning.
        Called after Phase A has converged.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen for joint fine-tuning")

    # numel gives total params in tensor p 
    def encoder_param_count(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters()) 

    def decoder_param_count(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        spectrogram: torch.Tensor,   # [B, 1, n_mels, T_audio]
        tokens: torch.Tensor,        # [B, T_dec] decoder input (teacher forcing)
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Teacher forcing is when during training, correct previous tokens are fed as decoder input, regardless of what the model would have predicted. This makes training stable and fast.
        """

        spec = spectrogram.squeeze(1)  # [B, n_mels, T_audio]

        # Encoder forward pass
        # Whisper encoder returns BaseModelOutput with .last_hidden_state
        encoder_out = self.encoder(input_features=spec).last_hidden_state
        # Shape: [B, T_enc, d_model]

        # Decoder forward pass with cross-attention to encoder
        logits = self.decoder(tokens, encoder_out)
        # Shape: [B, T_dec, vocab_size]

        return logits

    # disable gradient computation for inference 
    @torch.no_grad()
    def generate(
        self,
        spectrogram: torch.Tensor,   # [1, 1, n_mels, T_audio] (single example)
        max_length: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> list[int]:
        """
        Autoregressive generation for inference.

        At each step:
        1. Feed current token sequence to decoder
        2. Get logits for next token
        3. Sample from distribution (or take argmax for greedy)
        4. Append predicted token
        5. Repeat until EOS or max_length
        """
        self.eval() # disables dropout so all neurons are active for deterministic output
        device = next(self.parameters()).device

        # Encode audio once
        spec = spectrogram.squeeze(1) # runs spectrogram through the encoder to get hidden states for cross-attention
        encoder_out = self.encoder(input_features=spec).last_hidden_state

        # Start with BOS token
        tokens = torch.tensor([[BOS_TOKEN]], device=device)

        for _ in range(max_length - 1):
            # Decode current sequence
            logits = self.decoder(tokens, encoder_out)

            # Get logits for the LAST position (next token prediction)
            next_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            next_logits = next_logits / temperature

            # Sample next token
            next_token = _sample_token(next_logits, top_p=top_p)

            # Append to sequence
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)

            # Stop at EOS
            if next_token == EOS_TOKEN:
                break

        return tokens[0].tolist()


def _sample_token(logits: torch.Tensor, top_p: float = 0.9) -> int:
    """
    Nucleus (top-p) sampling.

    Instead of always picking the highest-probability token (greedy),
    we sample from the smallest set of tokens whose cumulative probability exceeds top_p. This produces more diverse, natural-sounding output.

    top_p=1.0 → sample from full distribution
    top_p=0.0 → equivalent to greedy (always pick top-1)
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative prob above top_p
    # Shift right by 1 to include the token that crosses the threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0.0

    # Renormalize and sample
    sorted_probs /= sorted_probs.sum()
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[sampled_idx].item()


def build_model(whisper_size: str = "base") -> PianoTranscriptionModel:
    """
    Build model with reasonable defaults for Mac M-series training.
    """
    model_name = f"openai/whisper-{whisper_size}"
    return PianoTranscriptionModel(
        whisper_model_name=model_name,
        decoder_layers=4,
        decoder_heads=8 if whisper_size == "base" else 6,
        decoder_ff_dim=2048,
        max_token_len=2048,
        dropout=0.1,
    )


if __name__ == '__main__':
    """Verify model builds and forward pass works."""
    print("Building model...")
    model = build_model("tiny")  # Use tiny for quick test

    total = sum(p.numel() for p in model.parameters())
    trainable = model.trainable_param_count()

    print(f"\nModel summary:")
    print(f"  Encoder params:  {model.encoder_param_count():>10,}  (frozen)")
    print(f"  Decoder params:  {model.decoder_param_count():>10,}  (trainable)")
    print(f"  Total params:    {total:>10,}")
    print(f"  Trainable now:   {trainable:>10,}  ({trainable/total*100:.1f}%)")

    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    B = 2  # batch size
    n_mels = 80   # Whisper tiny uses 80 mel bins
    T_audio = 3000  # ~69 seconds at Whisper's 43 frames/sec
    T_dec = 128

    # Whisper expects specific mel shape [B, n_mels, T]
    spec = torch.randn(B, 1, n_mels, T_audio)
    tokens = torch.randint(0, VOCAB_SIZE, (B, T_dec))
    tokens[:, 0] = BOS_TOKEN  # First token is always BOS

    logits = model(spec, tokens)
    print(f"  Input spec shape:   {list(spec.shape)}")
    print(f"  Input tokens shape: {list(tokens.shape)}")
    print(f"  Output logits shape:{list(logits.shape)}")
    print(f"  Expected:           [{B}, {T_dec}, {VOCAB_SIZE}]")
    assert logits.shape == (B, T_dec, VOCAB_SIZE), "Shape mismatch!"
    print("\nForward pass OK.")
