"""
tests/test_memory.py

Tests for core/memory.py
Run with: python -m pytest tests/test_memory.py -v

Strategy:
- No mocking of ChromaDB or sentence-transformers.
  Both run locally with no API calls — fast and free to use in tests.
- Each test gets a fresh isolated ChromaDB in a temp directory.
  No test pollutes another. No leftover data in your real chroma_db.
- The embedding model downloads once and is cached locally after that.
  First run may take 30-60 seconds to download all-MiniLM-L6-v2.
"""

import os
import pytest
import tempfile

from core.extractor import MeetingExtraction, ActionItem, Decision, OpenQuestion
from core.memory import MeetingMemory, MemoryError


# ==============================================================================
# SAMPLE DATA — reused across tests
# ==============================================================================

SAMPLE_TRANSCRIPT = """
Alice: Good morning. Let's go through the sprint update.
Bob: I'll finish the login API by Thursday. High priority.
Alice: Sarah, can you review the UI mockups by Wednesday?
Sarah: Sure.
Bob: We decided to use PostgreSQL instead of MySQL.
Alice: Open question — do we need Redis for caching at launch?
Sarah: Not sure, let's defer it.
"""

SAMPLE_EXTRACTION = MeetingExtraction(
    meeting_topic="Sprint Planning",
    participants=["Alice", "Bob", "Sarah"],
    action_items=[
        ActionItem(task="Finish login API", owner="Bob", deadline="Thursday", priority="High"),
        ActionItem(task="Review UI mockups", owner="Sarah", deadline="Wednesday", priority="Medium"),
    ],
    decisions=[
        Decision(description="Use PostgreSQL instead of MySQL", rationale=None),
    ],
    open_questions=[
        OpenQuestion(question="Do we need Redis for caching at launch?", raised_by="Alice"),
    ],
)

SECOND_TRANSCRIPT = """
Dave: Quick sync on the mobile app.
Eve: We decided to use React Native instead of Flutter.
Dave: I'll set up the project repo by Friday.
Eve: Open question — do we support iOS 15 or drop it?
"""

SECOND_EXTRACTION = MeetingExtraction(
    meeting_topic="Mobile App Sync",
    participants=["Dave", "Eve"],
    action_items=[
        ActionItem(task="Set up project repo", owner="Dave", deadline="Friday", priority="High"),
    ],
    decisions=[
        Decision(description="Use React Native instead of Flutter", rationale=None),
    ],
    open_questions=[
        OpenQuestion(question="Do we support iOS 15?", raised_by="Eve"),
    ],
)


# ==============================================================================
# FIXTURE — fresh isolated ChromaDB per test
# ==============================================================================

@pytest.fixture
def memory(tmp_path):
    """Fresh isolated ChromaDB per test — points to pytest tmp_path, not real DB."""
    return MeetingMemory(path=str(tmp_path))


# ==============================================================================
# TEST 1 — Init
# ==============================================================================

def test_memory_initializes(memory):
    """ChromaDB + embedding model loads without errors."""
    assert memory is not None


def test_initial_count_is_zero(memory):
    assert memory.count() == 0


# ==============================================================================
# TEST 2 — store()
# ==============================================================================

def test_store_returns_string_id(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert isinstance(meeting_id, str)
    assert len(meeting_id) > 0


def test_store_returns_valid_uuid(memory):
    import uuid
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    # Must be a valid UUID — will raise if not
    uuid.UUID(meeting_id)


def test_store_increments_count(memory):
    assert memory.count() == 0
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert memory.count() == 1
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)
    assert memory.count() == 2


def test_store_empty_transcript_raises(memory):
    with pytest.raises(ValueError, match="empty"):
        memory.store("", SAMPLE_EXTRACTION)


def test_store_whitespace_transcript_raises(memory):
    with pytest.raises(ValueError, match="empty"):
        memory.store("    ", SAMPLE_EXTRACTION)


def test_store_with_date(memory):
    meeting_id = memory.store(
        SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION, meeting_date="2025-06-19"
    )
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["date"] == "2025-06-19"


def test_store_default_date_is_today(memory):
    from datetime import date
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["date"] == date.today().isoformat()


def test_store_with_project_name(memory):
    meeting_id = memory.store(
        SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION, project_name="backend"
    )
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["project"] == "backend"


def test_store_default_project_is_general(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["project"] == "general"


def test_store_metadata_action_count(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["action_count"] == 2


def test_store_metadata_decision_count(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["decision_count"] == 1


def test_store_metadata_topic(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["topic"] == "Sprint Planning"


def test_store_metadata_participants(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert "Alice" in record["metadata"]["participants"]
    assert "Bob" in record["metadata"]["participants"]


def test_store_unknown_topic_fallback(memory):
    extraction = MeetingExtraction(
        meeting_topic=None,
        participants=[],
        action_items=[],
        decisions=[],
        open_questions=[],
    )
    meeting_id = memory.store("Some transcript content.", extraction)
    record = memory.get_by_id(meeting_id)
    assert record["metadata"]["topic"] == "Unknown"


# ==============================================================================
# TEST 3 — get_by_id()
# ==============================================================================

def test_get_by_id_returns_correct_meeting(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert record is not None
    assert record["meeting_id"] == meeting_id


def test_get_by_id_returns_none_for_missing(memory):
    result = memory.get_by_id("nonexistent-id-12345")
    assert result is None


def test_get_by_id_document_contains_transcript(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert "login API" in record["document"]


def test_get_by_id_document_contains_decisions(memory):
    """_build_document must include decisions for better search recall."""
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert "PostgreSQL" in record["document"]


def test_get_by_id_document_contains_topic(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    record = memory.get_by_id(meeting_id)
    assert "Sprint Planning" in record["document"]


# ==============================================================================
# TEST 4 — search()
# ==============================================================================

def test_search_empty_query_raises(memory):
    with pytest.raises(ValueError, match="empty"):
        memory.search("")


def test_search_whitespace_query_raises(memory):
    with pytest.raises(ValueError, match="empty"):
        memory.search("   ")


def test_search_on_empty_db_returns_empty_list(memory):
    results = memory.search("PostgreSQL database decision")
    assert results == []


def test_search_returns_list(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    results = memory.search("database decision")
    assert isinstance(results, list)


def test_search_returns_correct_fields(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    results = memory.search("database")
    assert len(results) > 0
    result = results[0]
    assert "meeting_id" in result
    assert "date" in result
    assert "topic" in result
    assert "participants" in result
    assert "excerpt" in result
    assert "relevance_score" in result


def test_search_relevance_score_between_0_and_1(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    results = memory.search("PostgreSQL database")
    for r in results:
        assert 0 <= r["relevance_score"] <= 1


def test_search_finds_relevant_meeting(memory):
    """Semantic search must find PostgreSQL meeting when asked about databases."""
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)

    results = memory.search("PostgreSQL database decision")
    assert len(results) > 0
    # Top result should be the sprint meeting, not the mobile app meeting
    assert results[0]["topic"] == "Sprint Planning"


def test_search_finds_mobile_meeting_for_react_query(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)

    results = memory.search("React Native mobile framework")
    assert len(results) > 0
    assert results[0]["topic"] == "Mobile App Sync"


def test_search_respects_n_results(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)

    results = memory.search("meeting", n_results=1)
    assert len(results) == 1


def test_search_n_results_does_not_exceed_stored_count(memory):
    """Asking for 10 results when only 1 stored must not crash."""
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    results = memory.search("meeting", n_results=10)
    assert len(results) == 1


def test_search_excerpt_max_500_chars(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    results = memory.search("login API")
    for r in results:
        assert len(r["excerpt"]) <= 500


# ==============================================================================
# TEST 5 — Project filter
# ==============================================================================

def test_search_project_filter_isolates_results(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION, project_name="backend")
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION, project_name="mobile")

    results = memory.search("meeting decisions", project="backend")
    assert all(r["project"] == "backend" for r in results)


def test_search_wrong_project_returns_empty(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION, project_name="backend")
    results = memory.search("PostgreSQL", project="mobile")
    assert results == []


# ==============================================================================
# TEST 6 — list_meetings()
# ==============================================================================

def test_list_meetings_returns_list(memory):
    result = memory.list_meetings()
    assert isinstance(result, list)


def test_list_meetings_empty_when_no_data(memory):
    assert memory.list_meetings() == []


def test_list_meetings_returns_all(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)
    meetings = memory.list_meetings()
    assert len(meetings) == 2


def test_list_meetings_sorted_by_date_descending(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION, meeting_date="2025-06-01")
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION, meeting_date="2025-06-19")

    meetings = memory.list_meetings()
    assert meetings[0]["date"] == "2025-06-19"
    assert meetings[1]["date"] == "2025-06-01"


def test_list_meetings_respects_limit(memory):
    memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.store(SECOND_TRANSCRIPT, SECOND_EXTRACTION)

    meetings = memory.list_meetings(limit=1)
    assert len(meetings) == 1


# ==============================================================================
# TEST 7 — delete()
# ==============================================================================

def test_delete_existing_meeting_returns_true(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    result = memory.delete(meeting_id)
    assert result is True


def test_delete_removes_from_count(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert memory.count() == 1
    memory.delete(meeting_id)
    assert memory.count() == 0


def test_delete_nonexistent_returns_false(memory):
    result = memory.delete("nonexistent-id")
    assert result is False


def test_deleted_meeting_not_retrievable(memory):
    meeting_id = memory.store(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    memory.delete(meeting_id)
    assert memory.get_by_id(meeting_id) is None


# ==============================================================================
# TEST 8 — _build_document()
# ==============================================================================

def test_build_document_contains_transcript(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "login API" in doc


def test_build_document_contains_topic(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "Sprint Planning" in doc


def test_build_document_contains_decisions(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "PostgreSQL" in doc


def test_build_document_contains_action_items(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "Finish login API" in doc


def test_build_document_contains_open_questions(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "Redis" in doc


def test_build_document_contains_participants(memory):
    doc = memory._build_document(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
    assert "Alice" in doc
    assert "Bob" in doc


def test_build_document_empty_extraction_still_works(memory):
    """Empty extraction must not crash _build_document."""
    extraction = MeetingExtraction(
        meeting_topic=None,
        participants=[],
        action_items=[],
        decisions=[],
        open_questions=[],
    )
    doc = memory._build_document("Some transcript.", extraction)
    assert "Some transcript." in doc