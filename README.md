# HarmonyNet

HarmonyNet is an end-to-end AI pipeline that converts **solo piano audio (MP3/WAV)** into **readable sheet music**, with planned support for musical annotations fingerings, dynamics, phrasing in v2. 

The project is intentionally scoped to **solo piano** pieces
(e.g. *Fur Elise*, *Gymnopedie No. 1*) to reduce ambiguity and improve
transcription quality.

## Version 1 - Baseline System (Complete)

Version 1 builds a working end-to-end pipeline for solo piano transcription:
raw audio in, sheet music out.

### Pipeline

![HarmonyNet pipeline](assets/pipeline_v1.png)

**Audio** &rarr; **ML Inference** &rarr; **Quantization** &rarr; **MusicXML Encoding** &rarr; **PDF Rendering**

 Audio Loading (`src/audio_loader.py`):  Load MP3/WAV/FLAC, resample to 22050 Hz mono, peak-normalize 
ML Inference (`src/inference.py`): basic-pitch CNN (ONNX backend) &rarr; onset, note, contour probability matrices &rarr; note events 
Quantization (`src/quantizer.py`): Snap continuous time to musical grid (16th-note resolution), assign measures and beats 
Encoding (`src/encoder.py`): Convert quantized notes to MusicXML via music21 
Rendering (`src/renderer.py`): MusicXML &rarr; PDF via MuseScore CLI 
CLI (`src/cli.py`): Click-based interface tying it all together 

### What V1 does well
- Accurate pitch detection across the full 88-key piano range (MIDI 21-108)
- Correct onset timing and note durations
- Works on real recordings (tested with Fur Elise, Gymnopedie No. 1)
- Configurable tempo, time signature, and detection thresholds

### Known V1 limitations
- Single treble clef (no bass clef / grand staff splitting)
- Slightly poor rest detection
- Dense notation can appear crowded
- Accuracy improves when tempo is specified explicitly by user

### Tested on
- C major scale (synthetic, 8 notes) - perfect transcription
- Fur Elise (3 min recording, 1747 notes) - correct opening melody, full piece captured
- Gymnopedie No. 1 (3 min recording, 841 notes) - sparse texture transcribed cleanly

## Setup and Requirements

**Python 3.12+** with virtual environment:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### MuseScore (optional, for PDF rendering)

Install MuseScore 4 from: https://musescore.org/en/download

Default locations:
macOS:  `/Applications/MuseScore 4.app/Contents/MacOS/mscore` 
Linux:  `/usr/bin/mscore` or `/usr/local/bin/mscore4` 
Windows: `C:\Program Files\MuseScore 4\bin\MuseScore4.exe`

If MuseScore is not installed, the pipeline still produces MusicXML output that can be opened in any notation software.

## Usage

**Generate sheet music PDF:**
```bash
python -m src.cli transcribe input.mp3 -o output.pdf
```

**With custom tempo and time signature:**
```bash
python -m src.cli transcribe data/inputs/Gymnopedie.mp3 -o data/outputs/gymnopedie.pdf --tempo 70 --time-sig 3/4
```

**MusicXML only (no PDF):**
```bash
python -m src.cli transcribe input.mp3 --no-pdf
```

**Keep intermediate MusicXML alongside PDF:**
```bash
python -m src.cli transcribe input.mp3 -o output.pdf --keep-musicxml
```

**Check dependencies:**
```bash
python -m src.cli check
```

### CLI Options

 `--tempo`:  120, Tempo in BPM 
 `--time-sig`: 4/4, Time signature 
 `--onset-threshold`: 0.5, Onset detection sensitivity (0-1) 
 `--frame-threshold`: 0.3, Note frame sensitivity (0-1) 
 `--title`: filename, Score title 
 `--no-pdf`: false, Output MusicXML only 
 `--keep-musicxml`: false, Keep MusicXML alongside PDF 

## Project Structure

```
HarmonyNet/
  src/
    note_events.py    # Core data structures (NoteEvent, TranscriptionResult)
    audio_loader.py   # Audio loading and preprocessing
    inference.py      # ML inference (basic-pitch + ONNX)
    quantizer.py      # Musical grid quantization
    encoder.py        # MusicXML generation (music21)
    renderer.py       # PDF rendering (MuseScore CLI)
    cli.py            # Command-line interface
  data/
    inputs/           # Audio files
    outputs/          # Generated sheet music
  docs/               # Technical notes
  assets/             # Pipeline diagrams
  tests/
```

## Technical Notes

- **ONNX Runtime** is used instead of TensorFlow for Python 3.12 compatibility. See `docs/inference_guide.md`.
- **basic-pitch** (ICASSP 2022 model) provides the CNN that produces onset, note, and contour predictions from Harmonic CQT spectrograms.
- **music21** handles MusicXML encoding. **MuseScore** handles PDF rendering.
- A scipy compatibility shim patches `scipy.signal.gaussian` for scipy 1.14+ (see `src/inference.py`).

## Version 2 - Upcoming 
