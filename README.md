# HarmonyNet

An end-to-end AI pipeline that converts **solo piano audio (MP3/WAV) into readable sheet music, with future support for musical annotations
(fingerings, dynamics, phrasing).

This project is intentionally scoped to solo piano pieces
(e.g. *Für Elise*, *River Flows in You*) to reduce ambiguity and improve
transcription quality.

## Version 1 (Baseline System)

Version 1 focuses on building a reliable end-to-end pipeline for solo piano transcription.
The goal is to convert raw piano audio into readable sheet music using a modular, reproducible system. This version prioritizes correctness, modularity, and interpretability, establishing a strong foundation for higher-level musical reasoning and annotation in version 2. 

To activate virtual environment: `source venv/bin/activate`

## High-Level Pipeline
