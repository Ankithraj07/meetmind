"""
core/transcriber.py

Responsibility: Convert audio files OR raw text into clean transcript strings.
Supports: .mp3, .wav, .m4a, .webm, .mp4, .mpeg, .mpga (via Groq Whisper), .txt (passthrough)
API: Groq — model: whisper-large-v3-turbo
"""

import os
import logging
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga"}
SUPPORTED_TEXT_FORMATS = {".txt"}


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class Transcriber:
    """
    Converts audio files or text files into plain transcript strings.

    Uses Groq Whisper for audio — free tier, no credit card needed.
    Text files are read directly without any API call.

    Usage:
        transcriber = Transcriber()
        text = transcriber.transcribe("path/to/audio.mp3")
        text = transcriber.transcribe("path/to/notes.txt")
        text = transcriber.transcribe_raw("Hello this is raw text")
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        self.client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transcribe(self, file_path: str) -> str:
        """
        Main entry point. Accepts a file path.
        Routes to audio or text handler based on file extension.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Clean transcript string.

        Raises:
            FileNotFoundError: If the file does not exist.
            TranscriptionError: If Groq Whisper fails or format is unsupported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext in SUPPORTED_AUDIO_FORMATS:
            logger.info(f"Audio file detected ({ext}). Sending to Groq Whisper.")
            return self._transcribe_audio(path)

        if ext in SUPPORTED_TEXT_FORMATS:
            logger.info(f"Text file detected ({ext}). Reading directly — no API call.")
            return self._read_text_file(path)

        raise TranscriptionError(
            f"Unsupported file format: '{ext}'. "
            f"Supported audio: {SUPPORTED_AUDIO_FORMATS}. "
            f"Supported text: {SUPPORTED_TEXT_FORMATS}."
        )

    def transcribe_raw(self, text: str) -> str:
        """
        Passthrough for raw text strings already transcribed or pasted by user.
        No API call. Just cleans and returns.

        Args:
            text: Raw transcript string.

        Returns:
            Stripped, clean string.

        Raises:
            ValueError: If text is empty or whitespace only.
        """
        if not text or not text.strip():
            raise ValueError("Raw text input is empty.")
        return text.strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe_audio(self, path: Path) -> str:
        """
        Sends audio file to Groq Whisper API.
        Model: whisper-large-v3-turbo — best accuracy on Groq free tier.

        Returns:
            Plain transcript string.

        Raises:
            TranscriptionError: Wraps any Groq API or file I/O failure.
        """
        try:
            with open(path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    response_format="text",   # plain string, not JSON
                    language="en",            # remove this line if meetings are multilingual
                )

            # response is already a plain string when response_format="text"
            transcript = response.strip()

            if not transcript:
                raise TranscriptionError(
                    f"Groq returned an empty transcript for '{path.name}'. "
                    "Check if the audio file has actual speech."
                )

            logger.info(f"Groq Whisper complete. File: {path.name} | Characters: {len(transcript)}")
            return transcript

        except TranscriptionError:
            raise  # don't double-wrap our own errors

        except Exception as e:
            raise TranscriptionError(
                f"Groq Whisper failed for '{path.name}': {e}"
            ) from e

    def _read_text_file(self, path: Path) -> str:
        """
        Reads a .txt file and returns its content as a string.

        Raises:
            TranscriptionError: If file is empty or not valid UTF-8.
        """
        try:
            content = path.read_text(encoding="utf-8").strip()

            if not content:
                raise TranscriptionError(f"Text file is empty: {path.name}")

            logger.info(f"Text file read complete. File: {path.name} | Characters: {len(content)}")
            return content

        except UnicodeDecodeError as e:
            raise TranscriptionError(
                f"Could not read '{path.name}' as UTF-8. "
                f"Save the file with UTF-8 encoding and retry. Error: {e}"
            ) from e