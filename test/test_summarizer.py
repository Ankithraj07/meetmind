"""
tests/test_summarizer.py

Tests for core/summarizer.py
Run with: python -m pytest tests/test_summarizer.py -v

Two types of tests:
- Unit tests : No API call. Mocks Groq. Tests logic, input handling, output validation.
- Live test  : Real Groq API call. Only runs when Groq_API_KEY is set.
"""

import os
import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from core.extractor import MeetingExtraction, ActionItem, Decision, OpenQuestion
from core.summarizer import Summarizer


# ==============================================================================
# SAMPLE DATA
# ==============================================================================

SAMPLE_EXTRACTION = MeetingExtraction(
    meeting_topic="Sprint Planning — Q3",
    participants=["Alice", "Bob", "Sarah"],
    action_items=[
        ActionItem(task="Complete login API", owner="Bob", deadline="Thursday", priority="High"),
        ActionItem(task="Review UI mockups", owner="Sarah", deadline="Wednesday", priority="Medium"),
    ],
    decisions=[
        Decision(description="Use PostgreSQL instead of MySQL", rationale="Better performance at scale"),
    ],
    open_questions=[
        OpenQuestion(question="Do we need Redis for caching at launch?", raised_by="Alice"),
    ],
)

SAMPLE_EMAIL = """Subject: Meeting Summary — Sprint Planning Q3

Hi team,

This is a summary of our Sprint Planning meeting held on 2025-06-19.

KEY DECISIONS
- Use PostgreSQL instead of MySQL (Better performance at scale)

ACTION ITEMS
Task | Owner | Deadline | Priority
Complete login API | Bob | Thursday | High
Review UI mockups | Sarah | Wednesday | Medium

OPEN QUESTIONS
- Do we need Redis for caching at launch? (Raised by: Alice)

Please action the above items by their respective deadlines."""

EMPTY_EXTRACTION = MeetingExtraction(
    meeting_topic=None,
    participants=[],
    action_items=[],
    decisions=[],
    open_questions=[],
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_summarizer():
    """
    Summarizer with Groq fully mocked.
    chain.invoke() returns a fake AIMessage with SAMPLE_EMAIL content.
    No API call, no network needed.
    """
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"}):
        with patch("core.summarizer.ChatGroq"):
            summarizer = Summarizer.__new__(Summarizer)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(content=SAMPLE_EMAIL)
            summarizer.chain = mock_chain
            return summarizer


# ==============================================================================
# TEST 1 — Missing API key
# ==============================================================================

def test_raises_if_no_api_key():
    """Must fail at init — not silently at generate() call time."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            Summarizer()


# ==============================================================================
# TEST 2 — Return type and basic output
# ==============================================================================

def test_generate_returns_string(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert isinstance(result, str)


def test_generate_returns_non_empty_string(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert len(result.strip()) > 0


def test_generate_output_is_stripped(mock_summarizer):
    """No leading/trailing whitespace in the output."""
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert result == result.strip()


# ==============================================================================
# TEST 3 — Email structure checks
# ==============================================================================

def test_output_contains_subject_line(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert "Subject:" in result


def test_output_contains_action_items_section(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert "ACTION ITEMS" in result


def test_output_contains_decisions_section(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert "KEY DECISIONS" in result


def test_output_contains_open_questions_section(mock_summarizer):
    result = mock_summarizer.generate(SAMPLE_EXTRACTION)
    assert "OPEN QUESTIONS" in result


# ==============================================================================
# TEST 4 — Prompt is built correctly (chain.invoke called with right keys)
# ==============================================================================

def test_chain_invoked_with_correct_keys(mock_summarizer):
    """Verify all required template variables are passed to chain."""
    mock_summarizer.generate(SAMPLE_EXTRACTION, meeting_date="2025-06-19")

    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert "topic" in call_kwargs
    assert "date" in call_kwargs
    assert "participants" in call_kwargs
    assert "decisions" in call_kwargs
    assert "action_items" in call_kwargs
    assert "open_questions" in call_kwargs


def test_topic_passed_to_prompt(mock_summarizer):
    mock_summarizer.generate(SAMPLE_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["topic"] == "Sprint Planning — Q3"


def test_meeting_date_passed_to_prompt(mock_summarizer):
    mock_summarizer.generate(SAMPLE_EXTRACTION, meeting_date="2025-06-19")
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["date"] == "2025-06-19"


def test_participants_joined_as_string(mock_summarizer):
    mock_summarizer.generate(SAMPLE_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["participants"] == "Alice, Bob, Sarah"


# ==============================================================================
# TEST 5 — Default date fallback
# ==============================================================================

def test_default_date_is_today(mock_summarizer):
    """If no date passed, should default to today's ISO date."""
    mock_summarizer.generate(SAMPLE_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["date"] == date.today().isoformat()


# ==============================================================================
# TEST 6 — Fallback values for missing fields
# ==============================================================================

def test_empty_topic_falls_back_to_team_meeting(mock_summarizer):
    extraction = MeetingExtraction(
        meeting_topic=None,
        participants=["Alice"],
        action_items=[],
        decisions=[],
        open_questions=[],
    )
    mock_summarizer.generate(extraction)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["topic"] == "Team Meeting"


def test_empty_participants_falls_back_to_not_specified(mock_summarizer):
    extraction = MeetingExtraction(
        meeting_topic="Sync",
        participants=[],
        action_items=[],
        decisions=[],
        open_questions=[],
    )
    mock_summarizer.generate(extraction)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["participants"] == "Not specified"


def test_empty_action_items_shows_none_identified(mock_summarizer):
    mock_summarizer.generate(EMPTY_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["action_items"] == "None identified."


def test_empty_decisions_shows_none_identified(mock_summarizer):
    mock_summarizer.generate(EMPTY_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["decisions"] == "None identified."


def test_empty_open_questions_shows_none(mock_summarizer):
    mock_summarizer.generate(EMPTY_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert call_kwargs["open_questions"] == "None."


# ==============================================================================
# TEST 7 — Action item serialization format
# ==============================================================================

def test_action_items_serialized_with_pipe_format(mock_summarizer):
    """Each action item must be serialized as: task | Owner: X | Deadline: Y | Priority: Z"""
    mock_summarizer.generate(SAMPLE_EXTRACTION)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    action_text = call_kwargs["action_items"]

    assert "Complete login API" in action_text
    assert "Owner: Bob" in action_text
    assert "Deadline: Thursday" in action_text
    assert "Priority: High" in action_text


def test_missing_owner_shows_tbd(mock_summarizer):
    extraction = MeetingExtraction(
        meeting_topic="Sync",
        participants=["Alice"],
        action_items=[
            ActionItem(task="Write docs", owner=None, deadline=None, priority=None)
        ],
        decisions=[],
        open_questions=[],
    )
    mock_summarizer.generate(extraction)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert "Owner: TBD" in call_kwargs["action_items"]


def test_missing_deadline_shows_not_set(mock_summarizer):
    extraction = MeetingExtraction(
        meeting_topic="Sync",
        participants=["Alice"],
        action_items=[
            ActionItem(task="Write docs", owner="Alice", deadline=None, priority=None)
        ],
        decisions=[],
        open_questions=[],
    )
    mock_summarizer.generate(extraction)
    call_kwargs = mock_summarizer.chain.invoke.call_args[0][0]
    assert "Deadline: Not set" in call_kwargs["action_items"]


# ==============================================================================
# TEST 8 — Groq returns empty string → must raise
# ==============================================================================

def test_empty_Groq_response_raises():
    """If Groq returns blank content, must raise ValueError — not return empty string."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.summarizer.ChatGroq"):
            summarizer = Summarizer.__new__(Summarizer)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(content="   ")
            summarizer.chain = mock_chain

            with pytest.raises(ValueError, match="empty summary"):
                summarizer.generate(SAMPLE_EXTRACTION)


# ==============================================================================
# TEST 9 — Groq API failure propagates
# ==============================================================================

def test_Groq_failure_propagates():
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.summarizer.ChatGroq"):
            summarizer = Summarizer.__new__(Summarizer)
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("Groq 429 rate limit")
            summarizer.chain = mock_chain

            with pytest.raises(Exception, match="Groq 429 rate limit"):
                summarizer.generate(SAMPLE_EXTRACTION)


# ==============================================================================
# TEST 10 — LIVE TEST (real Groq API call)
# Only runs when Groq_API_KEY is set AND quota is available.
# ==============================================================================

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set Groq_API_KEY env var to run live Groq test."
)
def test_live_summary_with_real_Groq():
    """
    Real API call to Groq.
    Run with:
        set Groq_API_KEY=your_real_key   (Windows)
        python -m pytest tests/test_summarizer.py::test_live_summary_with_real_Groq -v
    """
    summarizer = Summarizer()
    result = summarizer.generate(SAMPLE_EXTRACTION, meeting_date="2025-06-19")

    print(f"\n--- LIVE SUMMARY EMAIL ---\n{result}\n--------------------------")

    # Structural checks — not exact wording
    assert isinstance(result, str)
    assert len(result) > 100, "Summary is suspiciously short."
    assert "Subject:" in result, "Missing subject line."
    assert "Bob" in result or "login" in result.lower(), "Action item content missing."
    assert "PostgreSQL" in result or "database" in result.lower(), "Decision content missing."