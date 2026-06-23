"""
core/memory.py

Responsibility: Store meeting transcripts + metadata in ChromaDB.
Enables: Semantic search across all past meetings ("what did we decide about X?")

Embeddings: sentence-transformers (all-MiniLM-L6-v2) — local, free, no API needed.
Vector DB: ChromaDB — persistent on disk, no server needed.
"""

import os
import uuid
import logging
from datetime import date
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

from core.extractor import MeetingExtraction

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
COLLECTION_NAME = "meetings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 90MB, downloads once on first run


class MemoryError(Exception):
    """Raised when store or search operations fail."""
    pass


class MeetingMemory:
    """
    Stores and searches meeting transcripts using ChromaDB + local embeddings.

    Each meeting stored as one document with enriched content:
    transcript + decisions + action items + open questions combined.
    This improves search recall — searching "Redis caching" finds meetings
    that mentioned it anywhere, not just in the raw transcript.

    Usage:
        memory = MeetingMemory()
        meeting_id = memory.store(transcript, extraction, date="2025-06-19")
        results = memory.search("what did we decide about the database?")
        record  = memory.get_by_id(meeting_id)
    """

    def __init__(self, path: Optional[str] = None):
        chroma_path = path or os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")

        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready. Path: {chroma_path} | Documents: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store(
        self,
        transcript: str,
        extraction: MeetingExtraction,
        meeting_date: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> str:
        """
        Stores a meeting transcript + extraction in ChromaDB.

        Args:
            transcript: Full plain text transcript.
            extraction: Structured MeetingExtraction from extractor.py
            meeting_date: ISO date string e.g. "2025-06-19". Defaults to today.
            project_name: Optional tag for filtering. e.g. "backend", "product"

        Returns:
            meeting_id (UUID string) — use this to retrieve the meeting later.

        Raises:
            ValueError: If transcript is empty.
            MemoryError: If ChromaDB operation fails.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript is empty. Nothing to store.")

        meeting_id = str(uuid.uuid4())
        meeting_date = meeting_date or date.today().isoformat()

        # Combine transcript + structured data into one rich document.
        # Richer document = better semantic search recall.
        document = self._build_document(transcript, extraction)

        metadata = {
            "meeting_id": meeting_id,
            "date": meeting_date,
            "topic": extraction.meeting_topic or "Unknown",
            "participants": ", ".join(extraction.participants) if extraction.participants else "Unknown",
            "project": project_name or "general",
            "action_count": len(extraction.action_items),
            "decision_count": len(extraction.decisions),
            "question_count": len(extraction.open_questions),
        }

        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[meeting_id],
            )
        except Exception as e:
            raise MemoryError(f"Failed to store meeting in ChromaDB: {e}") from e

        logger.info(
            f"Meeting stored. ID: {meeting_id} | "
            f"Topic: {metadata['topic']} | "
            f"Date: {meeting_date} | "
            f"Project: {metadata['project']}"
        )
        return meeting_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        project: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search across all stored meetings.

        Args:
            query: Natural language question.
                   e.g. "what did we decide about authentication?"
            n_results: Max number of results to return.
            project: Optional project name filter. Only searches within that project.

        Returns:
            List of dicts sorted by relevance. Each dict has:
            meeting_id, date, topic, participants, project,
            excerpt (first 500 chars), relevance_score (0-1, higher = more relevant)

        Raises:
            ValueError: If query is empty.
            MemoryError: If ChromaDB operation fails.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        total = self.collection.count()
        if total == 0:
            logger.info("No meetings stored yet. Search returned empty.")
            return []

        where_filter = {"project": project} if project else None
        safe_n = min(n_results, total)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=safe_n,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise MemoryError(f"ChromaDB search failed: {e}") from e

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "meeting_id": meta["meeting_id"],
                "date": meta["date"],
                "topic": meta["topic"],
                "participants": meta["participants"],
                "project": meta.get("project", "general"),
                "action_count": meta.get("action_count", 0),
                "decision_count": meta.get("decision_count", 0),
                "excerpt": doc[:500],  # preview — first 500 chars
                "relevance_score": round(1 - dist, 4),  # cosine: 1=identical, 0=unrelated
            })

        logger.info(f"Search '{query[:50]}' returned {len(output)} results.")
        return output

    def get_by_id(self, meeting_id: str) -> Optional[dict]:
        """
        Retrieve a specific meeting by its ID.

        Args:
            meeting_id: UUID string returned by store().

        Returns:
            Dict with keys: meeting_id, document, metadata.
            None if meeting not found.
        """
        try:
            result = self.collection.get(
                ids=[meeting_id],
                include=["documents", "metadatas"],
            )
        except Exception as e:
            raise MemoryError(f"Failed to retrieve meeting '{meeting_id}': {e}") from e

        if not result["ids"]:
            return None

        return {
            "meeting_id": meeting_id,
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def list_meetings(self, limit: int = 50) -> list[dict]:
        """
        Returns metadata for all stored meetings. No embeddings, no search.
        Used by the frontend Meeting History page.

        Args:
            limit: Max number of meetings to return.

        Returns:
            List of metadata dicts sorted by date descending.
        """
        try:
            result = self.collection.get(
                limit=limit,
                include=["metadatas"],
            )
        except Exception as e:
            raise MemoryError(f"Failed to list meetings: {e}") from e

        meetings = result["metadatas"]

        # Sort by date descending — most recent first
        meetings.sort(key=lambda m: m.get("date", ""), reverse=True)
        return meetings

    def delete(self, meeting_id: str) -> bool:
        """
        Delete a meeting by ID.

        Returns:
            True if deleted. False if meeting not found.
        """
        existing = self.get_by_id(meeting_id)
        if not existing:
            return False

        try:
            self.collection.delete(ids=[meeting_id])
            logger.info(f"Meeting deleted. ID: {meeting_id}")
            return True
        except Exception as e:
            raise MemoryError(f"Failed to delete meeting '{meeting_id}': {e}") from e

    def count(self) -> int:
        """Returns total number of meetings stored."""
        return self.collection.count()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document(self, transcript: str, extraction: MeetingExtraction) -> str:
        """
        Builds one enriched string for embedding.

        Why combine everything into one document:
        - Raw transcript alone misses structured context.
        - Searching "PostgreSQL decision" needs to match even if
          the word "decision" only appears in the extraction, not transcript.
        - One document per meeting keeps ChromaDB simple.
        """
        parts = [f"TRANSCRIPT:\n{transcript.strip()}"]

        if extraction.meeting_topic:
            parts.insert(0, f"TOPIC: {extraction.meeting_topic}")

        if extraction.decisions:
            decisions_text = "\n".join(
                f"- {d.description}" for d in extraction.decisions
            )
            parts.append(f"DECISIONS:\n{decisions_text}")

        if extraction.action_items:
            actions_text = "\n".join(
                f"- {a.task} (Owner: {a.owner or 'TBD'}, Deadline: {a.deadline or 'None'})"
                for a in extraction.action_items
            )
            parts.append(f"ACTION ITEMS:\n{actions_text}")

        if extraction.open_questions:
            questions_text = "\n".join(
                f"- {q.question}" for q in extraction.open_questions
            )
            parts.append(f"OPEN QUESTIONS:\n{questions_text}")

        if extraction.participants:
            parts.append(f"PARTICIPANTS: {', '.join(extraction.participants)}")

        return "\n\n".join(parts)