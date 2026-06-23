"""
tests/test_transcriber.py

Tests for core/transcriber.py
Run with: python -m pytest tests/test_transcriber.py -v

Two types of tests:
- Unit tests  : No API call. Tests logic, routing, error handling. Always fast.
- Live test   : Real Groq API call. Only runs if you have a real audio file + GROQ_API_KEY set.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.transcriber import Transcriber, TranscriptionError


# ==============================================================================
# SETUP — runs before every test
# ==============================================================================

@pytest.fixture
def transcriber():
    """Creates a Transcriber with a fake API key — enough for unit tests."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"}):
        with patch("core.transcriber.Groq"):  # don't actually connect to Groq
            return Transcriber()


@pytest.fixture
def txt_file(tmp_path):
    """Creates a real temporary .txt file with content."""
    f = tmp_path / "meeting.txt"
    f.write_text("Alice: Let's ship the feature by Friday.\nBob: Agreed.", encoding="utf-8")
    return f


@pytest.fixture
def empty_txt_file(tmp_path):
    """Creates a real temporary .txt file that is empty."""
    f = tmp_path / "empty.txt"
    f.write_text("   ", encoding="utf-8")
    return f


# ==============================================================================
# TEST 1 — Missing API key should raise immediately
# ==============================================================================

def test_raises_if_no_api_key():
    """Transcriber must fail at init if GROQ_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            Transcriber()


# ==============================================================================
# TEST 2 — transcribe_raw() — no API call involved
# ==============================================================================

def test_transcribe_raw_returns_clean_text(transcriber):
    raw = "  Hello this is a meeting transcript.  "
    result = transcriber.transcribe_raw(raw)
    assert result == "Hello this is a meeting transcript."


def test_transcribe_raw_raises_on_empty_string(transcriber):
    with pytest.raises(ValueError, match="empty"):
        transcriber.transcribe_raw("")


def test_transcribe_raw_raises_on_whitespace_only(transcriber):
    with pytest.raises(ValueError, match="empty"):
        transcriber.transcribe_raw("     ")


# ==============================================================================
# TEST 3 — .txt file routing — no API call
# ==============================================================================

def test_transcribe_txt_file_returns_content(transcriber, txt_file):
    result = transcriber.transcribe(str(txt_file))
    assert "Alice" in result
    assert "Friday" in result


def test_transcribe_txt_file_strips_whitespace(tmp_path, transcriber):
    f = tmp_path / "padded.txt"
    f.write_text("\n\n  Some content.  \n\n", encoding="utf-8")
    result = transcriber.transcribe(str(f))
    assert result == "Some content."


def test_transcribe_empty_txt_raises(transcriber, empty_txt_file):
    with pytest.raises(TranscriptionError, match="empty"):
        transcriber.transcribe(str(empty_txt_file))


# ==============================================================================
# TEST 4 — File not found
# ==============================================================================

def test_transcribe_missing_file_raises(transcriber):
    with pytest.raises(FileNotFoundError, match="File not found"):
        transcriber.transcribe("/nonexistent/path/audio.mp3")


# ==============================================================================
# TEST 5 — Unsupported file format
# ==============================================================================

def test_transcribe_unsupported_format_raises(transcriber, tmp_path):
    f = tmp_path / "notes.pdf"
    f.write_bytes(b"fake pdf content")
    with pytest.raises(TranscriptionError, match="Unsupported"):
        transcriber.transcribe(str(f))


# ==============================================================================
# TEST 6 — Audio routing (mocked — no real API call)
# ==============================================================================

@pytest.mark.parametrize("ext", [".mp3", ".wav", ".m4a", ".webm", ".mp4"])
def test_audio_formats_route_to_whisper(tmp_path, ext):
    """All supported audio extensions must route to _transcribe_audio."""
    f = tmp_path / f"audio{ext}"
    f.write_bytes(b"fake audio bytes")

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.transcriber.Groq") as MockGroq:
            # Mock the Groq client to return a fake transcript
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = "This is the meeting transcript."
            MockGroq.return_value = mock_client

            t = Transcriber()
            result = t.transcribe(str(f))

    assert result == "This is the meeting transcript."
    mock_client.audio.transcriptions.create.assert_called_once()


def test_whisper_called_with_correct_model(tmp_path):
    """Verify model is whisper-large-v3-turbo — not whisper-1 (OpenAI)."""
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake audio")

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = "Transcript text."
            MockGroq.return_value = mock_client

            t = Transcriber()
            t.transcribe(str(f))

    call_kwargs = mock_client.audio.transcriptions.create.call_args
    assert call_kwargs.kwargs["model"] == "whisper-large-v3-turbo"


def test_whisper_empty_response_raises(tmp_path):
    """If Groq returns blank string, must raise TranscriptionError — not silently pass."""
    f = tmp_path / "silent.mp3"
    f.write_bytes(b"fake silent audio")

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = "   "
            MockGroq.return_value = mock_client

            t = Transcriber()
            with pytest.raises(TranscriptionError, match="empty transcript"):
                t.transcribe(str(f))


def test_groq_api_failure_raises_transcription_error(tmp_path):
    """If Groq throws any exception, it must be wrapped in TranscriptionError."""
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake audio")

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.side_effect = Exception("Connection timeout")
            MockGroq.return_value = mock_client

            t = Transcriber()
            with pytest.raises(TranscriptionError, match="Groq Whisper failed"):
                t.transcribe(str(f))


# ==============================================================================
# TEST 7 — LIVE TEST (only runs if you set GROQ_API_KEY and provide a real file)
# ==============================================================================

LIVE_AUDIO_FILE = os.getenv("TEST_AUDIO_FILE")  # set this in your shell before running

@pytest.mark.skipif(
    not LIVE_AUDIO_FILE or not os.getenv("GROQ_API_KEY"),
    reason="Set TEST_AUDIO_FILE and GROQ_API_KEY env vars to run live test."
)
def test_live_transcription_with_real_audio():
    """
    Real API call to Groq. Only runs when you explicitly set:
        export GROQ_API_KEY=your_real_key
        export TEST_AUDIO_FILE=path/to/your/audio.mp3
    """
    t = Transcriber()
    result = t.transcribe(LIVE_AUDIO_FILE)

    print(f"\n--- LIVE TRANSCRIPT ---\n{result}\n-----------------------")

    assert isinstance(result, str)
    assert len(result) > 10, "Transcript is suspiciously short — check the audio file."