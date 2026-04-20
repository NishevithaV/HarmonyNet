import os
import sys
import uuid
import tempfile
from pathlib import Path

# Ensure project root is on path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.celery_app import celery
from api.storage import upload as r2_upload
from src.inference import PianoTranscriber
from src.quantizer import quantize_transcription
from src.encoder import MusicXMLEncoder
from src.renderer import render_to_pdf, is_musescore_available

RESULTS_DIR = Path(__file__).parent.parent / "data" / "job_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


@celery.task(bind=True) # allows Celery to pass task instance itself 
def transcribe_task(self, audio_bytes: bytes, filename: str, tempo: float, time_sig: str):
    """
    Background task: runs the full V1 pipeline on uploaded audio.
    Returns a dict with pdf_path and ai_analysis.
    """
    job_id = self.request.id
    ts_parts = time_sig.split("/")
    time_signature = (int(ts_parts[0]), int(ts_parts[1]))

    # 1. Write uploaded bytes to a temp file
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        # 2. Transcribe
        self.update_state(state="PROGRESS", meta={"step": "transcribing"})
        transcriber = PianoTranscriber()
        result = transcriber.transcribe(tmp_path)

        # 3. Quantize
        self.update_state(state="PROGRESS", meta={"step": "quantizing"})
        score = quantize_transcription(
            result,
            tempo_bpm=tempo,
            time_signature=time_signature,
        )

        # 4. Encode to MusicXML
        self.update_state(state="PROGRESS", meta={"step": "encoding"})
        musicxml_path = RESULTS_DIR / f"{job_id}.musicxml"
        encoder = MusicXMLEncoder(title=Path(filename).stem, composer="HarmonyNet")
        encoder.to_musicxml(score, musicxml_path)

        # 5. Render to PDF (keep MusicXML regardless for separate download)
        self.update_state(state="PROGRESS", meta={"step": "rendering"})
        pdf_path = RESULTS_DIR / f"{job_id}.pdf"
        has_pdf = False
        if is_musescore_available():
            render_to_pdf(musicxml_path, pdf_path)
            has_pdf = True

        # 6. Upload to R2
        self.update_state(state="PROGRESS", meta={"step": "uploading"})
        stem = Path(filename).stem
        r2_upload(str(musicxml_path), f"{job_id}.musicxml")
        if has_pdf:
            r2_upload(str(pdf_path), f"{job_id}.pdf")

        # 7. OpenAI analysis
        self.update_state(state="PROGRESS", meta={"step": "analysing"})
        ai_analysis = _get_ai_analysis(result, filename, tempo, time_sig)

        return {
            "has_pdf": has_pdf,
            "stem": stem,
            "num_notes": result.num_notes,
            "num_measures": score.num_measures,
            "ai_analysis": ai_analysis,
        }

    finally:
        tmp_path.unlink(missing_ok=True)


def _get_ai_analysis(result, filename: str, tempo: float, time_sig: str) -> str:
    """Call OpenAI to generate a short musical analysis of the transcription."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Build a compact note summary (first 40 notes to keep prompt small)
        sample_notes = result.notes[:40]
        note_list = ", ".join(
            f"{n.pitch_name}({n.duration_sec:.2f}s)" for n in sample_notes
        )

        prompt = (
            f"A piano piece called '{Path(filename).stem}' was transcribed. "
            f"Tempo: {tempo} BPM, time signature: {time_sig}. "
            f"Total notes detected: {result.num_notes}. "
            f"Opening notes: {note_list}. "
            "In 3-4 sentences: identify the likely piece if recognisable, "
            "describe the difficulty level, and give one specific practice tip for a beginner."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content

    except Exception:
        return None
