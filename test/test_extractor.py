"""
tests/test_extractor.py

Tests for core/extractor.py
Run with: python -m pytest tests/test_extractor.py -v

Two types of tests:
- Unit tests : No API call. Mocks GROQ. Tests logic, validation, error handling.
- Live test  : Real GROQ API call. Only runs when GROQ_API_KEY is set.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from core.extractor import (
    Extractor,
    MeetingExtraction,
    ActionItem,
    Decision,
    OpenQuestion,
)


# ==============================================================================
# SAMPLE DATA — reused across tests
# ==============================================================================

SAMPLE_TRANSCRIPT = """
Alice: Good morning everyone. Let's go through the sprint update.
Bob: I'll finish the login API by Thursday. It's high priority.
Alice: Great. Sarah, can you review the UI mockups by Wednesday?
Sarah: Sure, I'll get that done.
Bob: We also decided to drop MySQL and go with PostgreSQL for the main database.
Alice: Agreed. One open question — do we need Redis for caching at launch, or can we defer it?
Sarah: Not sure, let's revisit next week.
Alice: Alright. That's it for today.
"""

SAMPLE_EXTRACTION = MeetingExtraction(
    meeting_topic="Sprint Update Meeting",
    participants=["Alice", "Bob", "Sarah"],
    action_items=[
        ActionItem(task="Finish login API", owner="Bob", deadline="Thursday", priority="High"),
        ActionItem(task="Review UI mockups", owner="Sarah", deadline="Wednesday", priority=None),
    ],
    decisions=[
        Decision(description="Use PostgreSQL instead of MySQL", rationale=None),
    ],
    open_questions=[
        OpenQuestion(question="Do we need Redis for caching at launch?", raised_by="Alice"),
    ],
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_extractor():
    """
    Extractor with GROQ fully mocked.
    The chain.invoke() returns SAMPLE_EXTRACTION directly.
    No API call, no network, no key needed.
    """
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"}):
        with patch("core.extractor.ChatGoogleGenerativeAI") as MockLLM:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = SAMPLE_EXTRACTION

            extractor = Extractor.__new__(Extractor)
            extractor.chain = mock_chain
            return extractor


# ==============================================================================
# TEST 1 — Missing API key
# ==============================================================================

def test_raises_if_no_api_key():
    """Must fail at init — not silently later when extract() is called."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            Extractor()


# ==============================================================================
# TEST 2 — Empty / blank transcript input
# ==============================================================================

def test_extract_raises_on_empty_string(mock_extractor):
    with pytest.raises(ValueError, match="empty"):
        mock_extractor.extract("")


def test_extract_raises_on_whitespace_only(mock_extractor):
    with pytest.raises(ValueError, match="empty"):
        mock_extractor.extract("     ")


# ==============================================================================
# TEST 3 — Return type is always MeetingExtraction
# ==============================================================================

def test_extract_returns_meeting_extraction_type(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert isinstance(result, MeetingExtraction)


# ==============================================================================
# TEST 4 — Action items parsed correctly
# ==============================================================================

def test_action_items_are_list(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert isinstance(result.action_items, list)


def test_action_items_count(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert len(result.action_items) == 2


def test_action_item_has_required_task_field(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for item in result.action_items:
        assert isinstance(item, ActionItem)
        assert item.task  # must not be empty string or None


def test_action_item_owner_is_string_or_none(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for item in result.action_items:
        assert item.owner is None or isinstance(item.owner, str)


def test_action_item_deadline_is_string_or_none(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for item in result.action_items:
        assert item.deadline is None or isinstance(item.deadline, str)


def test_action_item_priority_is_valid_or_none(mock_extractor):
    valid_priorities = {"High", "Medium", "Low", None}
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for item in result.action_items:
        assert item.priority in valid_priorities


def test_bob_owns_login_api_task(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    login_tasks = [a for a in result.action_items if "login" in a.task.lower() or "API" in a.task]
    assert len(login_tasks) >= 1
    assert login_tasks[0].owner == "Bob"


# ==============================================================================
# TEST 5 — Decisions parsed correctly
# ==============================================================================

def test_decisions_are_list(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert isinstance(result.decisions, list)


def test_decisions_count(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert len(result.decisions) == 1


def test_decision_has_description(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for d in result.decisions:
        assert isinstance(d, Decision)
        assert d.description


def test_decision_rationale_is_string_or_none(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for d in result.decisions:
        assert d.rationale is None or isinstance(d.rationale, str)


def test_postgresql_decision_present(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    descriptions = [d.description.lower() for d in result.decisions]
    assert any("postgresql" in d for d in descriptions)


# ==============================================================================
# TEST 6 — Open questions parsed correctly
# ==============================================================================

def test_open_questions_are_list(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert isinstance(result.open_questions, list)


def test_open_questions_count(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert len(result.open_questions) == 1


def test_open_question_has_question_field(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for q in result.open_questions:
        assert isinstance(q, OpenQuestion)
        assert q.question


def test_redis_question_present(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    questions = [q.question.lower() for q in result.open_questions]
    assert any("redis" in q or "caching" in q for q in questions)


# ==============================================================================
# TEST 7 — Participants parsed correctly
# ==============================================================================

def test_participants_are_list(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert isinstance(result.participants, list)


def test_participants_are_strings(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    for p in result.participants:
        assert isinstance(p, str)


def test_known_participants_present(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert "Alice" in result.participants
    assert "Bob" in result.participants
    assert "Sarah" in result.participants


# ==============================================================================
# TEST 8 — Meeting topic
# ==============================================================================

def test_meeting_topic_is_string_or_none(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    assert result.meeting_topic is None or isinstance(result.meeting_topic, str)


def test_meeting_topic_not_empty_string(mock_extractor):
    result = mock_extractor.extract(SAMPLE_TRANSCRIPT)
    if result.meeting_topic is not None:
        assert result.meeting_topic.strip() != ""


# ==============================================================================
# TEST 9 — Empty meeting (no action items / decisions in transcript)
# ==============================================================================

def test_empty_meeting_returns_empty_lists():
    """
    If the transcript has no action items or decisions,
    the result must return empty lists — not crash.
    """
    empty_extraction = MeetingExtraction(
        meeting_topic="Casual Sync",
        participants=["Alice", "Bob"],
        action_items=[],
        decisions=[],
        open_questions=[],
    )

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.extractor.ChatGroq"):
            extractor = Extractor.__new__(Extractor)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = empty_extraction
            extractor.chain = mock_chain

            result = extractor.extract("Alice: Hi. Bob: Hi. Nothing else happened.")

    assert result.action_items == []
    assert result.decisions == []
    assert result.open_questions == []


# ==============================================================================
# TEST 10 — GROQ API failure is propagated
# ==============================================================================

def test_GROQ_failure_propagates():
    """If GROQ throws, it must not be silently swallowed."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
        with patch("core.extractor.ChatGoogleGenerativeAI"):
            extractor = Extractor.__new__(Extractor)
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("GROQ rate limit hit")
            extractor.chain = mock_chain

            with pytest.raises(Exception, match="GROQ rate limit hit"):
                extractor.extract(SAMPLE_TRANSCRIPT)


# ==============================================================================
# TEST 11 — LIVE TEST (real GROQ API call)
# Only runs when GROQ_API_KEY is set in environment.
# ==============================================================================

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set GROQ_API_KEY env var to run live GROQ test."
)
def test_live_extraction_with_real_GROQ():
    """
    Real API call to GROQ. Validates the full pipeline end to end.
    Run with:
        export GROQ_API_KEY=your_real_key
        python -m pytest tests/test_extractor.py::test_live_extraction_with_real_GROQ -v
    """
    extractor = Extractor()
    result = extractor.extract(SAMPLE_TRANSCRIPT)

    print(f"\n--- LIVE EXTRACTION RESULT ---")
    print(f"Topic     : {result.meeting_topic}")
    print(f"Participants: {result.participants}")
    print(f"Actions   : {len(result.action_items)}")
    for a in result.action_items:
        print(f"  - {a.task} | {a.owner} | {a.deadline} | {a.priority}")
    print(f"Decisions : {len(result.decisions)}")
    for d in result.decisions:
        print(f"  - {d.description}")
    print(f"Questions : {len(result.open_questions)}")
    for q in result.open_questions:
        print(f"  - {q.question}")
    print(f"------------------------------")

    # Structural checks — not hardcoded values, GROQ may phrase things differently
    assert isinstance(result, MeetingExtraction)
    assert len(result.action_items) >= 1, "Expected at least one action item"
    assert len(result.decisions) >= 1, "Expected at least one decision"
    assert len(result.open_questions) >= 1, "Expected at least one open question"
    assert len(result.participants) >= 2, "Expected at least 2 participants"