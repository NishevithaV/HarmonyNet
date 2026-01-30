"""
Convert quantized notes to MusicXML using music21.
"""

from pathlib import Path
from typing import Optional

from music21 import stream, note, meter, tempo, metadata, clef, instrument

from .quantizer import QuantizedScore, QuantizedNote


class MusicXMLEncoder:
    """
    Converts a QuantizedScore to a music21 Score object and exports to MusicXML.
    """

    def __init__(self, title: str = "Transcription", composer: str = "HarmonyNet"):
        # optionally later extend to ask user for piano metadata 
        self.title = title
        self.composer = composer

    def encode(self, score: QuantizedScore) -> stream.Score:
        """
        Convert QuantizedScore to music21 Score object.

        Returns a music21 Score that can be:
        - Written to MusicXML: score.write('musicxml', 'output.musicxml')
        - Written to MIDI: score.write('midi', 'output.mid')
        - Displayed in notation: score.show()
        """
        m21_score = stream.Score()

        # Add metadata
        m21_score.metadata = metadata.Metadata()
        m21_score.metadata.title = self.title
        m21_score.metadata.composer = self.composer

        piano_part = stream.Part()
        piano_part.insert(0, instrument.Piano())

        # Add tempo marking
        tempo_mark = tempo.MetronomeMark(
            number=score.config.tempo_bpm,
            referent=note.Note(type='quarter')
        )
        piano_part.insert(0, tempo_mark)

        # Add time signature
        ts_num, ts_denom = score.config.time_signature
        time_sig = meter.TimeSignature(f'{ts_num}/{ts_denom}')
        piano_part.insert(0, time_sig)

        # Group notes by measure
        for measure_num in range(1, score.num_measures + 1):
            m21_measure = self._create_measure(
                measure_num,
                score.get_notes_in_measure(measure_num),
                score.config.time_signature
            )
            piano_part.append(m21_measure)

        m21_score.append(piano_part)
        return m21_score

    def _create_measure(
        self,
        measure_num: int,
        notes: list[QuantizedNote],
        time_signature: tuple
    ) -> stream.Measure:
        """Create a music21 Measure from quantized notes."""
        m = stream.Measure(number=measure_num)

        # Add clef to first measure
        if measure_num == 1:
            m.insert(0, clef.TrebleClef())

        for qnote in notes:
            m21_note = self._create_note(qnote, time_signature)

            # Calculate offset within measure where beat is 1-indexed
            offset = qnote.beat - 1.0
            m.insert(offset, m21_note)

        return m

    def _create_note(
        self,
        qnote: QuantizedNote,
        time_signature: tuple
    ) -> note.Note:
        """Convert a QuantizedNote to a music21 Note."""
        # Create note from MIDI pitch
        m21_note = note.Note()
        m21_note.pitch.midi = qnote.pitch

        # Set duration (quarterLength is in quarter-note units)
        m21_note.quarterLength = qnote.duration_beats

        # Set velocity (music21 uses 0-127)
        m21_note.volume.velocity = qnote.velocity

        # Add note name as lyric for labeling
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
