# MeetMind — AI Meeting Intelligence

Transcribe meetings → extract action items and decisions → search past meetings → generate email summaries.

Built with Groq (Whisper + Llama), ChromaDB, FastAPI, and Streamlit.

---

## What it does

- Upload audio (.mp3, .wav, .m4a) or paste a transcript
- Extracts structured data: action items with owners and deadlines, decisions, open questions, participants
- Stores everything in a local vector database for semantic search
- Generates a professional email summary ready to send
- Search across all past meetings in plain English

---

## Stack

| Layer | Technology |
|---|---|
| Transcription | Groq Whisper (`whisper-large-v3-turbo`) |
| Extraction + Summary | Groq LLM (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, no API |
| Vector DB | ChromaDB — persistent on disk |
| API | FastAPI |
| UI | Streamlit |

All LLM calls use Groq free tier. No OpenAI key needed.

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/yourusername/meetmind.git
cd meetmind
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq key at https://console.groq.com — no credit card required.

### 4. Create data directories

```bash
# Windows (PowerShell)
New-Item -Path "data\meetings\.gitkeep" -ItemType File -Force

# Mac/Linux
mkdir -p data/meetings && touch data/meetings/.gitkeep
```

### 5. Run the API

```bash
python -m uvicorn api.main:app --reload
```

API runs at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 6. Run the frontend (separate terminal)

```bash
streamlit run frontend/app.py
```

UI runs at `http://localhost:8501`.

---

## Run with Docker

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

---

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/process` | Full pipeline — text input |
| POST | `/process/audio` | Full pipeline — audio file |
| POST | `/transcribe/audio` | Audio → transcript only |
| POST | `/transcribe/text` | Text passthrough |
| POST | `/extract` | Transcript → structured JSON |
| POST | `/store` | Save meeting to ChromaDB |
| GET | `/summary/{id}` | Generate email summary |
| GET | `/search?q=...` | Semantic search |
| GET | `/meetings` | List all meetings |
| DELETE | `/meetings/{id}` | Delete a meeting |

---

## Run tests

```bash
# All tests
python -m pytest test/ -v

# Individual modules
python -m pytest test/test_transcriber.py -v
python -m pytest test/test_extractor.py -v
python -m pytest test/test_summarizer.py -v
python -m pytest test/test_memory.py -v
python -m pytest test/test_api.py -v
```

181 unit tests. No API calls needed for unit tests — all mocked.

Live tests (real Groq API calls) are gated behind env vars:

```bash
# Windows
set GROQ_API_KEY=your_key
python -m pytest test/test_extractor.py::test_live_extraction_with_real_GROQ -v

# Audio live test
set TEST_AUDIO_FILE=path\to\audio.mp3
python -m pytest test/test_transcriber.py::test_live_transcription_with_real_audio -v
```

---

## Project structure

```
meetmind/
├── core/
│   ├── transcriber.py    # Audio/text → transcript (Groq Whisper)
│   ├── extractor.py      # Transcript → structured JSON (Groq LLM)
│   ├── summarizer.py     # Structured data → email (Groq LLM)
│   └── memory.py         # Store + search meetings (ChromaDB)
├── api/
│   └── main.py           # FastAPI endpoints
├── frontend/
│   └── app.py            # Streamlit UI
├── test/
│   ├── test_transcriber.py
│   ├── test_extractor.py
│   ├── test_summarizer.py
│   ├── test_memory.py
│   └── test_api.py
├── data/
│   └── meetings/         # Raw transcript storage (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key for Whisper + LLM |
| `CHROMA_PERSIST_PATH` | No | `./data/chroma_db` | Where ChromaDB stores data |
| `API_BASE_URL` | No | `http://localhost:8000` | Frontend → API URL |

---

## Known limitations

- ChromaDB is local only — no multi-user support
- Groq free tier: 14,400 requests/day on Llama, 2 hours/day on Whisper
- Audio files over 25MB may hit Groq's file size limit — split long recordings first
- Switching embedding models requires deleting `data/chroma_db/` and re-processing all meetings
