"""
core/extractor.py

Responsibility: Extract structured data from raw transcript text.
Output: ActionItem, Decision, OpenQuestion — typed Pydantic models.
Uses: LangChain + Gemini 1.5 Flash with structured output via .with_structured_output()

Why .with_structured_output() instead of JsonOutputParser:
- JsonOutputParser relies on the LLM following JSON instructions in the prompt.
  Gemini frequently wraps output in ```json fences, breaking the parser.
- .with_structured_output() uses Gemini's native function-calling under the hood.
  Schema enforcement happens at the API level, not prompt level. More reliable.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ==============================================================================
# OUTPUT SCHEMA
# These are contracts — downstream code (api, frontend) depends on these shapes.
# Do not rename fields without updating api/main.py and frontend/app.py.
# ==============================================================================

class ActionItem(BaseModel):
    task: str = Field(description="Concrete task that needs to be done.")
    owner: Optional[str] = Field(None, description="Person responsible. None if not mentioned.")
    deadline: Optional[str] = Field(None, description="Deadline if stated. e.g. 'by Friday', '2025-06-30'. None if not stated.")
    priority: Optional[str] = Field(None, description="High / Medium / Low. Infer from urgency in transcript.")


class Decision(BaseModel):
    description: str = Field(description="What was definitively decided in the meeting.")
    rationale: Optional[str] = Field(None, description="Why this decision was made, if explicitly stated.")


class OpenQuestion(BaseModel):
    question: str = Field(description="Unresolved question, blocker, or deferred item.")
    raised_by: Optional[str] = Field(None, description="Person who raised it. None if not named.")


class MeetingExtraction(BaseModel):
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    participants: list[str] = Field(
        default_factory=list,
        description="Names of people who spoke or were mentioned in the transcript."
    )
    meeting_topic: Optional[str] = Field(
        None,
        description="One-line topic or title inferred from the transcript."
    )


# ==============================================================================
# PROMPT
# Keep it tight. Gemini does not need verbose instructions.
# The schema handles structure — the prompt handles behaviour.
# ==============================================================================

EXTRACTION_PROMPT = """You are a meeting analyst. Extract structured information from the transcript.

Rules:
- Only extract what is explicitly stated or clearly implied. Do NOT invent details.
- If owner, deadline, or rationale is not mentioned — leave it null.
- Action items = concrete tasks someone must do. Not vague statements.
- Decisions = definitive conclusions the group reached.
- Open questions = unresolved items, blockers, things deferred to later.
- Participants = named people who spoke or were mentioned.

Transcript:
{transcript}"""


# ==============================================================================
# EXTRACTOR CLASS
# ==============================================================================

class Extractor:
    """
    Extracts structured data from a meeting transcript using Gemini 1.5 Flash.

    Usage:
        extractor = Extractor()
        result: MeetingExtraction = extractor.extract(transcript_text)
        print(result.action_items)
        print(result.decisions)
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com and add it to your .env file."
        )

        llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

        # .with_structured_output() uses Gemini's native function-calling.
        # More reliable than JsonOutputParser for Gemini — avoids markdown fence issues.
        self.llm = llm.with_structured_output(MeetingExtraction)
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        self.chain = self.prompt | self.llm

    def extract(self, transcript: str) -> MeetingExtraction:
        """
        Runs extraction on a plain text transcript.

        Args:
            transcript: Raw meeting transcript string.

        Returns:
            MeetingExtraction object — typed, validated, ready to use.

        Raises:
            ValueError: If transcript is empty.
            Exception: Propagates Gemini API or schema validation errors.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript is empty. Nothing to extract.")

        logger.info(f"Starting extraction. Transcript length: {len(transcript)} chars.")

        result: MeetingExtraction = self.chain.invoke({"transcript": transcript})

        logger.info(
            f"Extraction complete — "
            f"Actions: {len(result.action_items)} | "
            f"Decisions: {len(result.decisions)} | "
            f"Questions: {len(result.open_questions)} | "
            f"Participants: {len(result.participants)}"
        )

        return result