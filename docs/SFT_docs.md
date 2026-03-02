### Implemented in `v2/train.py` 

**Training Objective:**

Given (spectrogram, token_sequence) pairs, train the model to predict
each token given all previous tokens and the full audio encoding.

`Loss = CrossEntropy(predicted_logits, true_next_tokens)`

Concretely, for a sequence [BOS, t1, t2, t3, EOS]:
  Input to decoder:  [BOS, t1, t2, t3]  (all but last)
  Target for loss:   [t1,  t2, t3, EOS] (all but first)

The model predicts t1 from BOS+audio, t2 from [BOS,t1]+audio, etc.
We average the cross-entropy loss over all non-PAD token positions.

**Two-Phase SFT:**

Phase A - Frozen encoder:
  - Only decoder weights update
  - Higher LR (1e-4), faster convergence
  - Run until validation loss plateaus

Phase B - Joint fine-tuning:
  - Both encoder + decoder update
  - Encoder LR = 1e-5 (10x lower)
  - Decoder LR = 1e-4 (unchanged)
  - This is the actual SFT step: adapting Whisper to piano
  - Run until validation loss plateaus again

**Why two learning rates?**

Using the same LR for pre-trained and randomly-initialized weights causes
"catastrophic forgetting" where the pre-trained encoder adapts too fast and
loses the general audio understanding that made it useful.

Lower LR for the encoder = gentler nudges to its pre-trained knowledge.
Higher LR for the decoder = faster convergence for randomly-init weights.

This technique is called "discriminative fine-tuning" (Howard & Ruder, 2018,
same paper that introduced ULMFiT, the predecessor to modern LLM fine-tuning).