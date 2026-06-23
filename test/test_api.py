"""
tests/test_api.py

Tests for api/main.py
Run with: python -m pytest tests/test_api.py -v

Strategy:
- Uses FastAPI TestClient — no server needed, runs in-process.
- All core dependencies (Transcriber, Extractor, Summarizer, MeetingMemory)
  are mocked. API layer logic is what's being tested here, not core logic.
  Core logic already has its own test files.
- Each test group overrides only what it needs via dependency injection patches.
"""

import io
import os
import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.extractor import MeetingExtraction, ActionItem, Decision, OpenQuestion
from core.memory import MeetingMemory


# ==============================================================================
# SAMPLE DATA
# ==============================================================================

SAMPLE_TRANSCRIPT = "Alice: We decided to use PostgreSQL. Bob will finish the login API by Friday."

SAMPLE_EXTRACTION = MeetingExtraction(
    meeting_topic="Sprint Sync",
    participants=["Alice", "Bob"],
    action_items=[
        ActionItem(task="Finish login API", owner="Bob", deadline="Friday", priority="High"),
    ],
    decisions=[
        Decision(description="Use PostgreSQL instead of MySQL", rationale=None),
    ],
    open_questions=[],
)

SAMPLE_EMAIL = """Subject: Meeting Summary — Sprint Sync

Hi team,

Summary of Sprint Sync held on 2025-06-19.

KEY DECISIONS
- Use PostgreSQL instead of MySQL

ACTION ITEMS
Task | Owner | Deadline | Priority
Finish login API | Bob | Friday | High

Please action the above items."""

SAMPLE_MEETING_ID = "test-meeting-id-1234"


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_transcriber():
    with patch("api.main.get_transcriber") as mock:
        t = MagicMock()
        t.transcribe_raw.return_value = SAMPLE_TRANSCRIPT
        t.transcribe.return_value = SAMPLE_TRANSCRIPT
        mock.return_value = t
        yield t


@pytest.fixture
def mock_extractor():
    with patch("api.main.get_extractor") as mock:
        e = MagicMock()
        e.extract.return_value = SAMPLE_EXTRACTION
        mock.return_value = e
        yield e


@pytest.fixture
def mock_summarizer():
    with patch("api.main.get_summarizer") as mock:
        s = MagicMock()
        s.generate.return_value = SAMPLE_EMAIL
        mock.return_value = s
        yield s


@pytest.fixture
def mock_memory(tmp_path):
    with patch("api.main.get_memory") as mock:
        m = MagicMock()
        m.store.return_value = SAMPLE_MEETING_ID
        m.search.return_value = [
            {
                "meeting_id": SAMPLE_MEETING_ID,
                "date": "2025-06-19",
                "topic": "Sprint Sync",
                "participants": "Alice, Bob",
                "project": "backend",
                "action_count": 1,
                "decision_count": 1,
                "excerpt": SAMPLE_TRANSCRIPT[:500],
                "relevance_score": 0.92,
            }
        ]
        m.get_by_id.return_value = {
            "meeting_id": SAMPLE_MEETING_ID,
            "document": f"TOPIC: Sprint Sync\n\nTRANSCRIPT:\n{SAMPLE_TRANSCRIPT}\n\nDECISIONS:\n- Use PostgreSQL",
            "metadata": {
                "meeting_id": SAMPLE_MEETING_ID,
                "date": "2025-06-19",
                "topic": "Sprint Sync",
                "participants": "Alice, Bob",
                "project": "backend",
                "action_count": 1,
                "decision_count": 1,
                "question_count": 0,
            },
        }
        m.list_meetings.return_value = [
            {
                "meeting_id": SAMPLE_MEETING_ID,
                "date": "2025-06-19",
                "topic": "Sprint Sync",
                "participants": "Alice, Bob",
                "project": "backend",
                "action_count": 1,
                "decision_count": 1,
                "question_count": 0,
            }
        ]
        m.delete.return_value = True
        mock.return_value = m
        yield m


@pytest.fixture
def client(mock_transcriber, mock_extractor, mock_summarizer, mock_memory):
    """
    TestClient with all core dependencies mocked.
    Use this for most tests — fast, no API calls.
    """
    from api.main import app
    return TestClient(app)


# ==============================================================================
# TEST 1 — HEALTH
# ==============================================================================

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_returns_version(client):
    response = client.get("/health")
    assert "version" in response.json()


# ==============================================================================
# TEST 2 — POST /transcribe/text
# ==============================================================================

def test_transcribe_text_returns_200(client):
    response = client.post("/transcribe/text", json={"text": SAMPLE_TRANSCRIPT})
    assert response.status_code == 200


def test_transcribe_text_returns_transcript(client):
    response = client.post("/transcribe/text", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["transcript"] == SAMPLE_TRANSCRIPT


def test_transcribe_text_returns_char_count(client):
    response = client.post("/transcribe/text", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["char_count"] == len(SAMPLE_TRANSCRIPT)


def test_transcribe_text_filename_is_null(client):
    response = client.post("/transcribe/text", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["filename"] is None


def test_transcribe_text_empty_input_returns_400(client, mock_transcriber):
    mock_transcriber.transcribe_raw.side_effect = ValueError("Raw text input is empty.")
    response = client.post("/transcribe/text", json={"text": ""})
    assert response.status_code == 400


def test_transcribe_text_missing_field_returns_422(client):
    response = client.post("/transcribe/text", json={})
    assert response.status_code == 422


# ==============================================================================
# TEST 3 — POST /transcribe/audio
# ==============================================================================

def test_transcribe_audio_returns_200(client):
    audio_bytes = b"fake audio content"
    response = client.post(
        "/transcribe/audio",
        files={"file": ("meeting.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert response.status_code == 200


def test_transcribe_audio_returns_transcript(client):
    audio_bytes = b"fake audio content"
    response = client.post(
        "/transcribe/audio",
        files={"file": ("meeting.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert response.json()["transcript"] == SAMPLE_TRANSCRIPT


def test_transcribe_audio_returns_filename(client):
    audio_bytes = b"fake audio content"
    response = client.post(
        "/transcribe/audio",
        files={"file": ("meeting.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert response.json()["filename"] == "meeting.mp3"


def test_transcribe_audio_groq_failure_returns_422(client, mock_transcriber):
    from core.transcriber import TranscriptionError
    mock_transcriber.transcribe.side_effect = TranscriptionError("Groq Whisper failed")
    response = client.post(
        "/transcribe/audio",
        files={"file": ("meeting.mp3", io.BytesIO(b"bad audio"), "audio/mpeg")},
    )
    assert response.status_code == 422


# ==============================================================================
# TEST 4 — POST /extract
# ==============================================================================

def test_extract_returns_200(client):
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    assert response.status_code == 200


def test_extract_returns_action_items(client):
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    data = response.json()
    assert "action_items" in data
    assert isinstance(data["action_items"], list)


def test_extract_returns_decisions(client):
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    data = response.json()
    assert "decisions" in data
    assert isinstance(data["decisions"], list)


def test_extract_returns_open_questions(client):
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    data = response.json()
    assert "open_questions" in data


def test_extract_returns_participants(client):
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    data = response.json()
    assert "participants" in data


def test_extract_empty_transcript_returns_400(client):
    response = client.post("/extract", json={"transcript": "   "})
    assert response.status_code == 400


def test_extract_missing_field_returns_422(client):
    response = client.post("/extract", json={})
    assert response.status_code == 422


def test_extract_groq_failure_returns_500(client, mock_extractor):
    mock_extractor.extract.side_effect = Exception("Groq LLM error")
    response = client.post("/extract", json={"transcript": SAMPLE_TRANSCRIPT})
    assert response.status_code == 500


# ==============================================================================
# TEST 5 — POST /store
# ==============================================================================

def test_store_returns_200(client):
    response = client.post("/store", json={
        "transcript": SAMPLE_TRANSCRIPT,
        "extraction": SAMPLE_EXTRACTION.model_dump(),
    })
    assert response.status_code == 200


def test_store_returns_meeting_id(client):
    response = client.post("/store", json={
        "transcript": SAMPLE_TRANSCRIPT,
        "extraction": SAMPLE_EXTRACTION.model_dump(),
    })
    assert response.json()["meeting_id"] == SAMPLE_MEETING_ID


def test_store_returns_stored_status(client):
    response = client.post("/store", json={
        "transcript": SAMPLE_TRANSCRIPT,
        "extraction": SAMPLE_EXTRACTION.model_dump(),
    })
    assert response.json()["status"] == "stored"


def test_store_passes_project_name_to_memory(client, mock_memory):
    client.post("/store", json={
        "transcript": SAMPLE_TRANSCRIPT,
        "extraction": SAMPLE_EXTRACTION.model_dump(),
        "project_name": "backend",
    })
    call_kwargs = mock_memory.store.call_args[1]
    assert call_kwargs["project_name"] == "backend"


def test_store_empty_transcript_returns_400(client, mock_memory):
    mock_memory.store.side_effect = ValueError("Transcript is empty.")
    response = client.post("/store", json={
        "transcript": "",
        "extraction": SAMPLE_EXTRACTION.model_dump(),
    })
    assert response.status_code == 400


# ==============================================================================
# TEST 6 — GET /summary/{meeting_id}
# ==============================================================================

def test_summary_returns_200(client):
    response = client.get(f"/summary/{SAMPLE_MEETING_ID}")
    assert response.status_code == 200


def test_summary_returns_email(client):
    response = client.get(f"/summary/{SAMPLE_MEETING_ID}")
    assert response.json()["email_summary"] == SAMPLE_EMAIL


def test_summary_returns_meeting_id(client):
    response = client.get(f"/summary/{SAMPLE_MEETING_ID}")
    assert response.json()["meeting_id"] == SAMPLE_MEETING_ID


def test_summary_not_found_returns_404(client, mock_memory):
    mock_memory.get_by_id.return_value = None
    response = client.get("/summary/nonexistent-id")
    assert response.status_code == 404


def test_summary_with_meeting_date(client):
    response = client.get(f"/summary/{SAMPLE_MEETING_ID}?meeting_date=2025-06-19")
    assert response.status_code == 200


# ==============================================================================
# TEST 7 — POST /process (pipeline — text)
# ==============================================================================

def test_process_text_returns_200(client):
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.status_code == 200


def test_process_text_returns_meeting_id(client):
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["meeting_id"] == SAMPLE_MEETING_ID


def test_process_text_returns_transcript(client):
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["transcript"] == SAMPLE_TRANSCRIPT


def test_process_text_returns_extraction(client):
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    data = response.json()
    assert "extraction" in data
    assert "action_items" in data["extraction"]
    assert "decisions" in data["extraction"]


def test_process_text_returns_email_summary(client):
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["email_summary"] == SAMPLE_EMAIL


def test_process_text_returns_meeting_date(client):
    response = client.post("/process", json={
        "text": SAMPLE_TRANSCRIPT,
        "meeting_date": "2025-06-19",
    })
    assert response.json()["meeting_date"] == "2025-06-19"


def test_process_text_default_date_is_today(client):
    from datetime import date
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.json()["meeting_date"] == date.today().isoformat()


def test_process_text_empty_input_returns_400(client, mock_transcriber):
    mock_transcriber.transcribe_raw.side_effect = ValueError("Raw text input is empty.")
    response = client.post("/process", json={"text": ""})
    assert response.status_code == 400


def test_process_text_extraction_failure_returns_500(client, mock_extractor):
    mock_extractor.extract.side_effect = Exception("Groq down")
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.status_code == 500
    assert "Extraction failed" in response.json()["detail"]


def test_process_text_store_failure_returns_500(client, mock_memory):
    mock_memory.store.side_effect = Exception("ChromaDB full")
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.status_code == 500
    assert "Store failed" in response.json()["detail"]


def test_process_text_summary_failure_returns_500(client, mock_summarizer):
    mock_summarizer.generate.side_effect = Exception("Groq timeout")
    response = client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    assert response.status_code == 500
    assert "Summary failed" in response.json()["detail"]


def test_process_calls_all_four_steps(client, mock_transcriber, mock_extractor, mock_summarizer, mock_memory):
    """Pipeline must call all 4 steps exactly once."""
    client.post("/process", json={"text": SAMPLE_TRANSCRIPT})
    mock_transcriber.transcribe_raw.assert_called_once()
    mock_extractor.extract.assert_called_once()
    mock_memory.store.assert_called_once()
    mock_summarizer.generate.assert_called_once()


# ==============================================================================
# TEST 8 — POST /process/audio (pipeline — audio)
# ==============================================================================

def test_process_audio_returns_200(client):
    response = client.post(
        "/process/audio",
        files={"file": ("meeting.mp3", io.BytesIO(b"fake audio"), "audio/mpeg")},
    )
    assert response.status_code == 200


def test_process_audio_returns_all_fields(client):
    response = client.post(
        "/process/audio",
        files={"file": ("meeting.mp3", io.BytesIO(b"fake audio"), "audio/mpeg")},
    )
    data = response.json()
    assert "meeting_id" in data
    assert "transcript" in data
    assert "extraction" in data
    assert "email_summary" in data
    assert "meeting_date" in data


def test_process_audio_groq_failure_returns_422(client, mock_transcriber):
    from core.transcriber import TranscriptionError
    mock_transcriber.transcribe.side_effect = TranscriptionError("Whisper failed")
    response = client.post(
        "/process/audio",
        files={"file": ("meeting.mp3", io.BytesIO(b"bad audio"), "audio/mpeg")},
    )
    assert response.status_code == 422


# ==============================================================================
# TEST 9 — GET /search
# ==============================================================================

def test_search_returns_200(client):
    response = client.get("/search?q=PostgreSQL database")
    assert response.status_code == 200


def test_search_returns_results_list(client):
    response = client.get("/search?q=PostgreSQL database")
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_returns_count(client):
    response = client.get("/search?q=PostgreSQL database")
    assert "count" in response.json()


def test_search_returns_query_in_response(client):
    response = client.get("/search?q=PostgreSQL database")
    assert response.json()["query"] == "PostgreSQL database"


def test_search_empty_query_returns_400(client):
    response = client.get("/search?q=")
    assert response.status_code == 400


def test_search_missing_query_returns_422(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_search_with_project_filter(client, mock_memory):
    client.get("/search?q=database&project=backend&n=3")
    mock_memory.search.assert_called_once_with(
        query="database", n_results=3, project="backend"
    )


# ==============================================================================
# TEST 10 — GET /meetings
# ==============================================================================

def test_list_meetings_returns_200(client):
    response = client.get("/meetings")
    assert response.status_code == 200


def test_list_meetings_returns_count(client):
    response = client.get("/meetings")
    assert "count" in response.json()


def test_list_meetings_returns_meetings_list(client):
    response = client.get("/meetings")
    assert isinstance(response.json()["meetings"], list)


def test_list_meetings_count_matches_list_length(client):
    response = client.get("/meetings")
    data = response.json()
    assert data["count"] == len(data["meetings"])


def test_list_meetings_respects_limit(client, mock_memory):
    client.get("/meetings?limit=10")
    mock_memory.list_meetings.assert_called_once_with(limit=10)


# ==============================================================================
# TEST 11 — DELETE /meetings/{meeting_id}
# ==============================================================================

def test_delete_meeting_returns_200(client):
    response = client.delete(f"/meetings/{SAMPLE_MEETING_ID}")
    assert response.status_code == 200


def test_delete_meeting_returns_deleted_status(client):
    response = client.delete(f"/meetings/{SAMPLE_MEETING_ID}")
    assert response.json()["status"] == "deleted"


def test_delete_nonexistent_returns_404(client, mock_memory):
    mock_memory.delete.return_value = False
    response = client.delete("/meetings/nonexistent-id")
    assert response.status_code == 404


def test_delete_returns_meeting_id_in_response(client):
    response = client.delete(f"/meetings/{SAMPLE_MEETING_ID}")
    assert response.json()["meeting_id"] == SAMPLE_MEETING_ID