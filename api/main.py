import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from celery.result import AsyncResult

from api.celery_app import celery
from api.tasks import transcribe_task

app = FastAPI(title="HarmonyNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    tempo: float = Form(120.0),
    time_sig: str = Form("4/4"),
):
    """Accept an audio file and dispatch a background transcription job."""
    if not file.filename.endswith((".mp3", ".wav", ".flac")):
        raise HTTPException(status_code=400, detail="File must be MP3, WAV, or FLAC")

    audio_bytes = await file.read()

    task = transcribe_task.delay(
        audio_bytes=audio_bytes,
        filename=file.filename,
        tempo=tempo,
        time_sig=time_sig,
    )

    return {"job_id": task.id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll the status of a transcription job."""
    result = AsyncResult(job_id, app=celery)

    if result.state == "PENDING":
        return {"status": "pending"}
    if result.state == "PROGRESS":
        return {"status": "processing", "step": result.info.get("step")}
    if result.state == "SUCCESS":
        info = result.result
        return {
            "status": "done",
            "num_notes": info["num_notes"],
            "num_measures": info["num_measures"],
            "has_pdf": info["has_pdf"],
            "ai_analysis": info["ai_analysis"],
        }
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}

    return {"status": result.state.lower()}


@app.get("/result/{job_id}/pdf")
def get_pdf(job_id: str):
    """Download the PDF for a completed job."""
    result = AsyncResult(job_id, app=celery)
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Job not complete")
    if not result.result.get("has_pdf"):
        raise HTTPException(status_code=404, detail="PDF not available — MuseScore not installed")
    path = Path(result.result["pdf_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    stem = result.result.get("stem", job_id)
    return FileResponse(str(path), media_type="application/pdf", filename=f"{stem}.pdf")


@app.get("/result/{job_id}/musicxml")
def get_musicxml(job_id: str):
    """Download the MusicXML for a completed job."""
    result = AsyncResult(job_id, app=celery)
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Job not complete")
    path = Path(result.result["musicxml_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="MusicXML file not found")
    stem = result.result.get("stem", job_id)
    return FileResponse(str(path), media_type="application/xml", filename=f"{stem}.musicxml")
