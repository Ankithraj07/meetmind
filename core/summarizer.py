"""
core/summarizer.py

Responsibility: Turn a MeetingExtraction into a formatted email summary.
Output: Plain-text email string, ready to copy-paste or send.
API: Gemini 2.0 Flash via langchain-google-genai 4.x
"""

import os
import logging
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from core.extractor import MeetingExtraction

load_dotenv()

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """You are an executive assistant writing a professional meeting summary email.

Use the structured meeting data below. Write clean, concise, enterprise-grade output.

Format (follow exactly):
Subject: <subject line>

Hi team,

<2-3 line intro stating what the meeting was about and when>

KEY DECISIONS
- <decision 1>
- <decision 2>

ACTION ITEMS
Task | Owner | Deadline | Priority
<row per action item>

OPEN QUESTIONS
- <question 1>
- <question 2>

<one-line professional closing>

Meeting Data:
Topic: {topic}
Date: {date}
Participants: {participants}
Decisions: {decisions}
Action Items: {action_items}
Open Questions: {open_questions}"""


class Summarizer:
    """
    Generates a professional meeting summary email from extracted meeting data.

    Usage:
        summarizer = Summarizer()
        email_text = summarizer.generate(extraction, meeting_date="2025-06-19")
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com and add it to your .env file."
        )

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,  # 0.3 — slight variation is fine for email writing
            api_key=api_key,          # api_key not google_api_key — changed in v4.x
        )
        self.prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
        self.chain = self.prompt | self.llm

    def generate(
        self,
        extraction: MeetingExtraction,
        meeting_date: Optional[str] = None,
    ) -> str:
        """
        Generates a formatted email summary from structured meeting data.

        Args:
            extraction: MeetingExtraction object from extractor.py
            meeting_date: ISO date string e.g. "2025-06-19". Defaults to today.

        Returns:
            Email string with subject line and body. Ready to copy-paste.

        Raises:
            EnvironmentError: If GROQ_API_KEY is missing.
            Exception: Propagates Gemini API errors.
        """
        meeting_date = meeting_date or date.today().isoformat()

        # Serialize Pydantic models into readable strings for the prompt.
        # The LLM gets plain text, not JSON blobs.
        action_items_text = "\n".join(
            f"- {a.task} | Owner: {a.owner or 'TBD'} | "
            f"Deadline: {a.deadline or 'Not set'} | Priority: {a.priority or 'Unknown'}"
            for a in extraction.action_items
        ) or "None identified."

        decisions_text = "\n".join(
            f"- {d.description}" + (f" (Rationale: {d.rationale})" if d.rationale else "")
            for d in extraction.decisions
        ) or "None identified."

        questions_text = "\n".join(
            f"- {q.question}" + (f" (Raised by: {q.raised_by})" if q.raised_by else "")
            for q in extraction.open_questions
        ) or "None."

        response = self.chain.invoke({
            "topic": extraction.meeting_topic or "Team Meeting",
            "date": meeting_date,
            "participants": ", ".join(extraction.participants) if extraction.participants else "Not specified",
            "decisions": decisions_text,
            "action_items": action_items_text,
            "open_questions": questions_text,
        })

        # LangChain wraps Gemini response in AIMessage — .content gives the string
        email_text = response.content.strip()

        if not email_text:
            raise ValueError("Gemini returned an empty summary. Check your extraction input.")

        logger.info(f"Summary email generated. Length: {len(email_text)} chars.")
        return email_text