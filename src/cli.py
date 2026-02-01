"""
Command-line interface for HarmonyNet

Usage Commands:
    python -m src.cli transcribe input.mp3 -o output.pdf
    python -m src.cli transcribe input.mp3 --tempo 100 --no-pdf
"""

import click
from pathlib import Path

from .inference import PianoTranscriber
from .quantizer import quantize_transcription
from .encoder import MusicXMLEncoder
from .renderer import render_to_pdf, is_musescore_available


@click.group()
@click.version_option(version='0.1.0', prog_name='harmonynet')
def cli():
    pass


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output path (default: input name + .pdf)')
@click.option('--tempo', type=float, default=120.0, help='Tempo in BPM (default: 120)')
@click.option('--time-sig', type=str, default='4/4', help='Time signature (default: 4/4)')
@click.option('--title', type=str, default=None, help='Score title')
@click.option('--onset-threshold', type=float, default=0.5, help='Onset detection threshold (0-1)')
@click.option('--frame-threshold', type=float, default=0.3, help='Note frame threshold (0-1)')
@click.option('--no-pdf', is_flag=True, help='Skip PDF rendering, output MusicXML only')
@click.option('--keep-musicxml', is_flag=True, help='Keep intermediate MusicXML file')
def transcribe(
    audio_path: str,
    output: str,
    tempo: float,
    time_sig: str,
    title: str,
    onset_threshold: float,
    frame_threshold: float,
    no_pdf: bool,
    keep_musicxml: bool,
):
    """
    Transcribe piano audio to sheet music.
    Currently acceptable audio formats: MP3, WAV, or FLAC file
    """
    audio_path = Path(audio_path)

    # Parse time signature
    ts_parts = time_sig.split('/')
    time_signature = (int(ts_parts[0]), int(ts_parts[1]))

    # Determine output paths
    if output:
        output_path = Path(output)
    else:
        output_path = audio_path.with_suffix('.pdf')

    musicxml_path = output_path.with_suffix('.musicxml')

    # Use filename as title if not provided
    if title is None:
        title = audio_path.stem

    click.echo(f"Transcribing: {audio_path.name}")

    # Step 1: Transcribe audio to notes
    click.echo("  [1/4] Running ML inference...")
    transcriber = PianoTranscriber(
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    result = transcriber.transcribe(audio_path)
    click.echo(f"        Found {result.num_notes} notes")

    # Step 2: Quantize to musical grid
    click.echo("  [2/4] Quantizing to musical grid...")
    score = quantize_transcription(
        result,
        tempo_bpm=tempo,
        time_signature=time_signature,
    )
    click.echo(f"        {score.num_measures} measures @ {tempo} BPM")

    # Step 3: Encode to MusicXML
    click.echo("  [3/4] Encoding to MusicXML...")
    encoder = MusicXMLEncoder(title=title, composer="HarmonyNet")
    encoder.to_musicxml(score, musicxml_path)
    click.echo(f"        Wrote: {musicxml_path}")

    # Step 4: Render to PDF
    if no_pdf:
        click.echo("  [4/4] Skipping PDF render (--no-pdf)")
        final_output = musicxml_path
    else:
        click.echo("  [4/4] Rendering PDF...")
        if not is_musescore_available():
            click.echo("        MuseScore not found. Skipping PDF render.")
            click.echo("        To download PDF version, install MuseScore from: https://musescore.org/en/download")
            final_output = musicxml_path
        else:
            pdf_path = render_to_pdf(musicxml_path, output_path)
            click.echo(f"        Wrote: {pdf_path}")
            final_output = pdf_path

            # Clean up MusicXML if not keeping
            if not keep_musicxml:
                musicxml_path.unlink()

    click.echo(f"\nDone! Output: {final_output}")


@cli.command()
def check():
    """Check if all dependencies are available."""
    click.echo("Checking dependencies...\n")

    # Check basic-pitch
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        click.echo(f"  basic-pitch: OK")
    except ImportError:
        click.echo(f"  basic-pitch: MISSING (pip install basic-pitch)")

    # Check music21
    try:
        import music21
        click.echo(f"  music21: OK")
    except ImportError:
        click.echo(f"  music21: MISSING (pip install music21)")

    # Check librosa
    try:
        import librosa
        click.echo(f"  librosa: OK")
    except ImportError:
        click.echo(f"  librosa: MISSING (pip install librosa)")

    # Check MuseScore
    if is_musescore_available():
        click.echo(f"  MuseScore: OK")
    else:
        click.echo(f"  MuseScore: NOT FOUND (optional, for PDF rendering)")
        click.echo(f"             Install from: https://musescore.org/en/download")


def main():
    cli()


if __name__ == '__main__':
    main()