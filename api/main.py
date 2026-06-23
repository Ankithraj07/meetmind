"""
api/main.py

FastAPI application — Meeting Assistant.
Run with: uvicorn api.main:app --reload

Architecture:
- Individual endpoints: fine-grained control, retry-able steps
- Pipeline endpoint POST /process: one call, full result — for the UI happy path
- Singletons: all core classes instantiated once, reused across requests
"""

import logging
import os
import tempfile
from datetime import date
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.extractor import Extractor, MeetingExtraction
from core.memory import MeetingMemory
from core.summarizer import Summarizer
from core.transcriber import Transcriber, TranscriptionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# APP SETUP
# ==============================================================================

app = FastAPI(
    title="Meeting Assistant API",
    description="Transcribe → Extract → Summarize → Search meetings.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# SINGLETONS
# Instantiated once on first request. Not at startup — avoids cold start penalty
# when the model/API isn't needed for a given request type.
# ==============================================================================

_transcriber: Optional[Transcriber] = None
_extractor: Optional[Extractor] = None
_summarizer: Optional[Summarizer] = None
_memory: Optional[MeetingMemory] = None


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def get_extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = Extractor()
    return _extractor


def get_summarizer() -> Summarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer


def get_memory() -> MeetingMemory:
    global _memory
    if _memory is None:
        _memory = MeetingMemory()
    return _memory


# ==============================================================================
# REQUEST / RESPONSE MODELS
# Explicit contracts — never return raw dicts from endpoints.
# ==============================================================================

class TranscribeTextRequest(BaseModel):
    text: str


class ExtractRequest(BaseModel):
    transcript: str


class StoreRequest(BaseModel):
    transcript: str
    extraction: dict           # MeetingExtraction serialized as dict from frontend
    meeting_date: Optional[str] = None
    project_name: Optional[str] = None


class ProcessRequest(BaseModel):
    """
    Pipeline endpoint request.
    Either text or an audio file upload — audio handled separately via multipart.
    This model covers the text-input pipeline path.
    """
    text: str
    meeting_date: Optional[str] = None
    project_name: Optional[str] = None


class TranscribeResponse(BaseModel):
    filename: Optional[str]
    transcript: str
    char_count: int


class StoreResponse(BaseModel):
    meeting_id: str
    status: str


class SummaryResponse(BaseModel):
    meeting_id: str
    email_summary: str


class ProcessResponse(BaseModel):
    """Full pipeline result — one object, everything the frontend needs."""
    meeting_id: str
    transcript: str
    extraction: dict
    email_summary: str
    meeting_date: str


class SearchResultResponse(BaseModel):
    query: str
    count: int
    results: list[dict]


class MeetingsListResponse(BaseModel):
    count: int
    meetings: list[dict]


# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """Basic liveness check. Returns 200 if the API is running."""
    return {"status": "ok", "version": "2.0.0"}


# ==============================================================================
# STEP 1 — TRANSCRIBE
# Two routes: audio file upload, or raw text passthrough.
# ==============================================================================

@app.post("/transcribe/audio", response_model=TranscribeResponse, tags=["Step 1 — Transcribe"])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload audio file → transcript text via Groq Whisper.
    Supported: .mp3 .wav .m4a .webm .mp4
    """
    suffix = os.path.splitext(file.filename or "audio.mp3")[-1].lower()

    # Whisper requires a real file path — write upload to temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = get_transcriber().transcribe(tmp_path)
    except TranscriptionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    return TranscribeResponse(
        filename=file.filename,
        transcript=transcript,
        char_count=len(transcript),
    )


@app.post("/transcribe/text", response_model=TranscribeResponse, tags=["Step 1 — Transcribe"])
def transcribe_text(body: TranscribeTextRequest):
    """
    Submit raw text → cleaned passthrough. No API call.
    Use this when the user pastes a transcript directly.
    """
    try:
        transcript = get_transcriber().transcribe_raw(body.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TranscribeResponse(
        filename=None,
        transcript=transcript,
        char_count=len(transcript),
    )


# ==============================================================================
# STEP 2 — EXTRACT
# ==============================================================================

@app.post("/extract", response_model=MeetingExtraction, tags=["Step 2 — Extract"])
def extract(body: ExtractRequest):
    """
    Transcript text → structured extraction via Groq LLM.
    Returns: action_items, decisions, open_questions, participants, meeting_topic.
    """
    if not body.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    try:
        return get_extractor().extract(body.transcript)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Extraction failed.")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# STEP 3 — STORE
# ==============================================================================

@app.post("/store", response_model=StoreResponse, tags=["Step 3 — Store"])
def store_meeting(body: StoreRequest):
    """
    Store transcript + extraction in ChromaDB for future search.
    Returns meeting_id — save this to retrieve the meeting later.
    """
    try:
        extraction = MeetingExtraction(**body.extraction)
        meeting_id = get_memory().store(
            transcript=body.transcript,
            extraction=extraction,
            meeting_date=body.meeting_date,
            project_name=body.project_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Store failed.")
        raise HTTPException(status_code=500, detail=str(e))

    return StoreResponse(meeting_id=meeting_id, status="stored")


# ==============================================================================
# STEP 4 — SUMMARY
# ==============================================================================

@app.get("/summary/{meeting_id}", response_model=SummaryResponse, tags=["Step 4 — Summary"])
def get_summary(meeting_id: str, meeting_date: Optional[str] = None):
    """
    Generate a formatted email summary for a stored meeting.
    Re-extracts from stored transcript then generates summary.
    """
    record = get_memory().get_by_id(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    # Pull raw transcript out of the stored enriched document
    raw_doc = record["document"]
    if "TRANSCRIPT:\n" in raw_doc:
        transcript = raw_doc.split("TRANSCRIPT:\n", 1)[1]
        # Strip anything appended after transcript (DECISIONS, ACTION ITEMS etc.)
        for section in ["\n\nDECISIONS:", "\n\nACTION ITEMS:", "\n\nOPEN QUESTIONS:", "\n\nPARTICIPANTS:"]:
            if section in transcript:
                transcript = transcript.split(section)[0]
    else:
        transcript = raw_doc

    try:
        extraction = get_extractor().extract(transcript)
        email = get_summarizer().generate(extraction, meeting_date=meeting_date)
    except Exception as e:
        logger.exception("Summary generation failed.")
        raise HTTPException(status_code=500, detail=str(e))

    return SummaryResponse(meeting_id=meeting_id, email_summary=email)


# ==============================================================================
# PIPELINE — POST /process
# Full pipeline in one call: text → extract → store → summarize → return all.
# This is what the Streamlit frontend calls for the happy path.
# ==============================================================================

@app.post("/process", response_model=ProcessResponse, tags=["Pipeline"])
def process_text(body: ProcessRequest):
    """
    Full pipeline — text input only.
    Transcribe (passthrough) → Extract → Store → Summarize.
    Returns everything the frontend needs in one response.

    For audio input, use POST /process/audio instead.
    """
    meeting_date = body.meeting_date or date.today().isoformat()

    # Step 1 — clean text
    try:
        transcript = get_transcriber().transcribe_raw(body.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2 — extract
    try:
        extraction = get_extractor().extract(transcript)
    except Exception as e:
        logger.exception("Pipeline extraction failed.")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Step 3 — store
    try:
        meeting_id = get_memory().store(
            transcript=transcript,
            extraction=extraction,
            meeting_date=meeting_date,
            project_name=body.project_name,
        )
    except Exception as e:
        logger.exception("Pipeline store failed.")
        raise HTTPException(status_code=500, detail=f"Store failed: {e}")

    # Step 4 — summarize
    try:
        email = get_summarizer().generate(extraction, meeting_date=meeting_date)
    except Exception as e:
        logger.exception("Pipeline summarization failed.")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    return ProcessResponse(
        meeting_id=meeting_id,
        transcript=transcript,
        extraction=extraction.model_dump(),
        email_summary=email,
        meeting_date=meeting_date,
    )


@app.post("/process/audio", response_model=ProcessResponse, tags=["Pipeline"])
async def process_audio(
    file: UploadFile = File(...),
    meeting_date: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """
    Full pipeline — audio file input.
    Transcribe (Groq Whisper) → Extract → Store → Summarize.
    """
    meeting_date = meeting_date or date.today().isoformat()

    # Step 1 — transcribe audio
    suffix = os.path.splitext(file.filename or "audio.mp3")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = get_transcriber().transcribe(tmp_path)
    except TranscriptionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    # Step 2 — extract
    try:
        extraction = get_extractor().extract(transcript)
    except Exception as e:
        logger.exception("Pipeline extraction failed.")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Step 3 — store
    try:
        meeting_id = get_memory().store(
            transcript=transcript,
            extraction=extraction,
            meeting_date=meeting_date,
            project_name=project_name,
        )
    except Exception as e:
        logger.exception("Pipeline store failed.")
        raise HTTPException(status_code=500, detail=f"Store failed: {e}")

    # Step 4 — summarize
    try:
        email = get_summarizer().generate(extraction, meeting_date=meeting_date)
    except Exception as e:
        logger.exception("Pipeline summarization failed.")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    return ProcessResponse(
        meeting_id=meeting_id,
        transcript=transcript,
        extraction=extraction.model_dump(),
        email_summary=email,
        meeting_date=meeting_date,
    )


# ==============================================================================
# SEARCH + HISTORY
# ==============================================================================

@app.get("/search", response_model=SearchResultResponse, tags=["Search"])
def search_meetings(q: str, project: Optional[str] = None, n: int = 5):
    """
    Semantic search across all stored meetings.
    Example: /search?q=what did we decide about authentication?
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = get_memory().search(query=q, n_results=n, project=project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResultResponse(query=q, count=len(results), results=results)


@app.get("/meetings", response_model=MeetingsListResponse, tags=["Search"])
def list_meetings(limit: int = 50):
    """List all stored meetings sorted by date descending."""
    try:
        meetings = get_memory().list_meetings(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return MeetingsListResponse(count=len(meetings), meetings=meetings)


@app.delete("/meetings/{meeting_id}", tags=["Search"])
def delete_meeting(meeting_id: str):
    """Delete a meeting from ChromaDB by ID."""
    deleted = get_memory().delete(meeting_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")
    return {"meeting_id": meeting_id, "status": "deleted"}