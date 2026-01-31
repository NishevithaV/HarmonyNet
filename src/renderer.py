"""
Render MusicXML to PDF using MuseScore CLI
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import platform


def find_musescore() -> Optional[Path]:
    """
    Attempt to find MuseScore installation.
    Returns path to executable or None if not found.
    """
    # Check if mscore is in PATH
    mscore_path = shutil.which('mscore')
    if mscore_path:
        return Path(mscore_path)

    mscore4_path = shutil.which('mscore4')
    if mscore4_path:
        return Path(mscore4_path)

    # Platform-specific default locations
    system = platform.system()

    if system == 'Darwin': # for macOS
        candidates = [
            Path('/Applications/MuseScore 4.app/Contents/MacOS/mscore'),
            Path('/Applications/MuseScore 3.app/Contents/MacOS/mscore'),
        ]
    elif system == 'Windows':
        candidates = [
            Path(r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'),
            Path(r'C:\Program Files\MuseScore 3\bin\MuseScore3.exe'),
        ]
    else:  # Linux
        candidates = [
            Path('/usr/bin/mscore'),
            Path('/usr/local/bin/mscore4'),
            Path('/usr/bin/musescore4'),
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


class PDFRenderer:
    """
    Renders MusicXML files to PDF using MuseScore.
    """

    def __init__(self, musescore_path: Optional[str | Path] = None):
        """
        Initialize renderer.
        """
        if musescore_path:
            self.musescore_path = Path(musescore_path)
        else:
            self.musescore_path = find_musescore()

        if self.musescore_path is None:
            raise RuntimeError(
                "MuseScore not found. Install from https://musescore.org/en/download "
                "or provide path via musescore_path parameter."
            )

        if not self.musescore_path.exists():
            raise FileNotFoundError(f"MuseScore not found at: {self.musescore_path}")

    def render_pdf(
        self,
        musicxml_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Render MusicXML file to PDF.
        Returns path to generated PDF file. 
        """
        musicxml_path = Path(musicxml_path)

        if not musicxml_path.exists():
            raise FileNotFoundError(f"MusicXML file not found: {musicxml_path}")

        if output_path is None:
            output_path = musicxml_path.with_suffix('.pdf')
        else:
            output_path = Path(output_path)

        # Run MuseScore CLI
        cmd = [
            str(self.musescore_path),
            str(musicxml_path),
            '-o', str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"MuseScore rendering failed:\n{result.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(f"PDF was not created at: {output_path}")

        return output_path

    def render_png(
        self,
        musicxml_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Render MusicXML to PNG image."""
        musicxml_path = Path(musicxml_path)

        if output_path is None:
            output_path = musicxml_path.with_suffix('.png')
        else:
            output_path = Path(output_path)

        cmd = [
            str(self.musescore_path),
            str(musicxml_path),
            '-o', str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"MuseScore rendering failed:\n{result.stderr}")

        return output_path


def render_to_pdf(
    musicxml_path: str | Path,
    output_path: Optional[str | Path] = None,
    musescore_path: Optional[str | Path] = None,
) -> Path:
    """Convenience function for one-shot PDF rendering."""
    renderer = PDFRenderer(musescore_path=musescore_path)
    return renderer.render_pdf(musicxml_path, output_path)


def is_musescore_available() -> bool:
    """Check if MuseScore is installed and accessible."""
    return find_musescore() is not None
