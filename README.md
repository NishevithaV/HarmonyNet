# HarmonyNet

An end-to-end AI pipeline that converts **solo piano audio (MP3/WAV) into readable sheet music, with future support for musical annotations
(fingerings, dynamics, phrasing).

This project is intentionally scoped to solo piano pieces
(e.g. *Für Elise*, *River Flows in You*) to reduce ambiguity and improve
transcription quality.

## Version 1 (Baseline System)

Version 1 focuses on building a reliable end-to-end pipeline for solo piano transcription.
The goal is to convert raw piano audio into readable sheet music using a modular, reproducible system. This version prioritizes correctness, modularity, and interpretability, establishing a strong foundation for higher-level musical reasoning and annotation in version 2. 

Note: For piano sheet music, there is typically both the treble and bass cleff. However, for V1 only a single Part is used and clef management is handled internally. 

To activate virtual environment: `source venv/bin/activate`

## Setup and Requirements for Usage 

MuseScore is a free, open-source notation program used in HarmoneyNet that can be driven from the command line. 

1. Install MuseScore 4 at : https://musescore.org/en/download
2. Ensure 'mscore' is in PATH, or configure the path below

macOS: /Applications/MuseScore 4.app/Contents/MacOS/mscore
Linux: /usr/bin/mscore or /usr/local/bin/mscore4
Windows: C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe

MuseScore CLI usage:
    mscore input.musicxml -o output.pdf

The -o flag determines output format based on extension:
    .pdf  → PDF
    .png  → PNG image
    .svg  → SVG vector
    .mid  → MIDI
    .mp3  → Audio (if soundfont configured)

## High-Level Pipeline
