EDA (Exploratory Data Analysis) on MAESTRO dataset 

Questions to understand about the MAESTRO dataset before implementing the tokenizer: 

1. How many performances? How long are they?
2. What's the distribution of notes per piece?
3. What pitch range is actually used? (Theory says 21-108, but practice may differ)
4. What velocity range? (Are there lots of quiet notes? Loud notes?)
5. What are typical note durations? (Mostly 16ths? Lots of held notes?)
6. What's the time gap between consecutive notes? (Dense chords? Sparse melodies?)

How this affects tokenization design: 
- If 95% of notes are in the range 36-96, we can use a smaller pitch vocab
- If most durations cluster around certian values, quantization of notes become simpler
- If velocity has a bimodal distribution (where the data clusters around two peaks rather than being spread evenly into the 128 possibl MIDI values), fewer velocity bins can we used