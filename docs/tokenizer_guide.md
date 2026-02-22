**State machine: (`decode_tokens`):**
- Maintain current_time (advances with TIME_SHIFT tokens)
- Maintain current_velocity (set by SET_VELOCITY tokens)
- Maintain active_notes dict: pitch → (start_time, velocity)
- On NOTE_ON: add to active_notes
- On NOTE_OFF: remove from active_notes, emit MidiNote