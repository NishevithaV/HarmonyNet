# HarmonyNet

HarmonyNet is an end-to-end AI pipeline that converts **solo piano audio (MP3/WAV)** into **readable sheet music**, with planned support for musical annotations fingerings, dynamics, phrasing in v2. 

The project is intentionally scoped to **solo piano** pieces
(e.g. *Für Elise*, *River Flows in You*) to reduce ambiguity and improve
transcription quality.

## Version 1 - Baseline System

Version 1 focuses on building a reliable, modular pipeline for solo piano
transcription. The goal is to convert raw piano audio into readable sheet music
using a reproducible and interpretable system.

This version prioritizes: 
- correctness
- modularity
- clear separation of concerns

and establishes a strong foundation for higher-level musical reasoning and
annotation in **Version 2**.

**Note:**
Standard piano sheet music uses both treble and bass clefs. In Version 1,
HarmonyNet represents the score using a single part, with clef management
handled internally. Multi-part and more expressive notation will be explored
in later versions.

## Setup and Requirements

### Virtual Environment

Activate virtual environment: 

```bash
source venv/bin/activate
```

### MuseScore (Required for rendering)

HarmonyNet uses MuseScore, a free and open-source notation program, to render
MusicXML scores into human-readable sheet music formats.

Install MuseScore 4 from: `https://musescore.org/en/download` <br/>
Ensure that `mscore` is available in your system `PATH`, or configure the path
manually.

**Default Locations:** 

macOS: `/Applications/MuseScore 4.app/Contents/MacOS/mscore` <br/>
Linux: `/usr/bin/mscore or /usr/local/bin/mscore4` <br/>
Windows: `C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe` <br/>

**MuseScore CLI usage:**
```bash
    mscore input.musicxml -o output.pdf
```

The -o flag determines output format based on extension: <br/>
-    .pdf  → PDF <br/>
-    .png  → PNG image <br/>
-    .svg  → SVG vector <br/>
-    .mid  → MIDI <br/>
-    .mp3  → Audio (if soundfont configured) <br/>

### Running Version 1 

**Generate Sheet Music PDF:**
```bash
python -m src.cli transcribe input.mp3 -o output.pdf
```

**Generate MusicXML only (for Editing):**
```bash
python -m src.cli transcribe input.mp3 --tempo 100 --no-pdf
```
This produces a MusicXML file that can be opened and edited directly in
MuseScore.

### High-Level Pipeline

![HarmonyNet pipeline](assets/pipeline_1.png)

## Version 2 - Upcoming 
