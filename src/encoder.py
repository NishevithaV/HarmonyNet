"""
Convert quantized notes to MusicXML using music21.
"""

from pathlib import Path
from typing import Optional

from music21 import stream, note, meter, tempo, metadata, clef, instrument, layout

from .quantizer import QuantizedScore, QuantizedNote

# Notes >= MIDI 60 (middle C) → treble clef; below → bass clef
TREBLE_BASS_SPLIT = 60

# Break onto a new system every N measures to prevent horizontal crowding
MEASURES_PER_SYSTEM = 4


class MusicXMLEncoder:
    """
    Converts a QuantizedScore to a music21 Score object and exports to MusicXML.
    Outputs a grand staff (treble + bass clef) with system breaks every 4 measures.
    """

    def __init__(self, title: str = "Transcription", composer: str = "HarmonyNet"):
        self.title = title
        self.composer = composer

    def encode(self, score: QuantizedScore) -> stream.Score:
        """Convert QuantizedScore to a grand-staff music21 Score."""
        m21_score = stream.Score()
        m21_score.metadata = metadata.Metadata()
        m21_score.metadata.title = self.title
        m21_score.metadata.composer = self.composer

        treble_part = stream.Part()
        bass_part = stream.Part()
        treble_part.insert(0, instrument.Piano())
        bass_part.insert(0, instrument.Piano())

        ts_num, ts_denom = score.config.time_signature
        time_sig_str = f'{ts_num}/{ts_denom}'
        treble_part.insert(0, tempo.MetronomeMark(
            number=score.config.tempo_bpm,
            referent=note.Note(type='quarter'),
        ))
        treble_part.insert(0, meter.TimeSignature(time_sig_str))
        bass_part.insert(0, meter.TimeSignature(time_sig_str))

        for measure_num in range(1, score.num_measures + 1):
            all_notes = score.get_notes_in_measure(measure_num)
            treble_notes = [n for n in all_notes if n.pitch >= TREBLE_BASS_SPLIT]
            bass_notes   = [n for n in all_notes if n.pitch <  TREBLE_BASS_SPLIT]

            treble_m = self._create_measure(measure_num, treble_notes, score.config.time_signature, is_treble=True)
            bass_m   = self._create_measure(measure_num, bass_notes,   score.config.time_signature, is_treble=False)

            # System break: new line every MEASURES_PER_SYSTEM measures
            if measure_num > 1 and (measure_num - 1) % MEASURES_PER_SYSTEM == 0:
                treble_m.insert(0, layout.SystemLayout(isNew=True))

            treble_part.append(treble_m)
            bass_part.append(bass_m)

        # Brace the two parts as a piano grand staff
        piano_staff = layout.StaffGroup(
            [treble_part, bass_part],
            name='Piano', abbreviation='Pno.',
            symbol='brace',
        )
        m21_score.insert(0, piano_staff)
        m21_score.append(treble_part)
        m21_score.append(bass_part)
        return m21_score

    def _create_measure(
        self,
        measure_num: int,
        notes: list[QuantizedNote],
        time_signature: tuple,
        is_treble: bool = True,
    ) -> stream.Measure:
        """Create a music21 Measure from quantized notes."""
        m = stream.Measure(number=measure_num)

        if measure_num == 1:
            m.insert(0, clef.TrebleClef() if is_treble else clef.BassClef())

        for qnote in notes:
            m21_note = self._create_note(qnote, time_signature)
            offset = qnote.beat - 1.0
            m.insert(offset, m21_note)

        return m

    def _create_note(
        self,
        qnote: QuantizedNote,
        time_signature: tuple
    ) -> note.Note:
        """Convert a QuantizedNote to a music21 Note."""
        m21_note = note.Note()
        m21_note.pitch.midi = qnote.pitch
        m21_note.quarterLength = qnote.duration_beats
        m21_note.volume.velocity = qnote.velocity
        m21_note.addLyric(qnote.pitch_name)
        return m21_note

    def to_musicxml(
        self,
        score: QuantizedScore,
        output_path: str | Path
    ) -> Path:
        """
        Encode and write to MusicXML file. Returns path to written file.
        """
        output_path = Path(output_path)
        m21_score = self.encode(score)
        m21_score.write('musicxml', fp=str(output_path))
        return output_path

    def to_midi(
        self,
        score: QuantizedScore,
        output_path: str | Path
    ) -> Path:
        """
        Encode and write to MIDI file (useful for playback verification).
        """
        output_path = Path(output_path)
        m21_score = self.encode(score)
        m21_score.write('midi', fp=str(output_path))
        return output_path


def encode_to_musicxml(
    score: QuantizedScore,
    output_path: str | Path,
    title: str = "Transcription",
    composer: str = "HarmonyNet",
) -> Path:
    """Convenience function for one-shot encoding."""
    encoder = MusicXMLEncoder(title=title, composer=composer)
    return encoder.to_musicxml(score, output_path)
