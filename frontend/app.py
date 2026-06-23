"""
frontend/app.py - Premium UI
"""

import os
import requests
import pandas as pd
import streamlit as st
from datetime import date

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MeetMind — AI Meeting Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

    * { box-sizing: border-box; }
    html, body, [data-testid="stAppViewContainer"] {
        background: #09090B !important;
        color: #FAFAFA;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 0 2rem 3rem 2rem !important;
        max-width: 1100px !important;
    }
    body, p, div, span, label {
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background: #09090B !important;
        border-right: 1px solid #27272A !important;
        width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.25rem;
    }
    .logo-wrap { margin-bottom: 2.5rem; }
    .logo-name {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .logo-tagline {
        font-size: 0.7rem;
        color: #71717A;
        font-weight: 400;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.9rem;
        background: #18181B;
        border-radius: 8px;
        border: 1px solid #27272A;
        margin-top: 1rem;
    }
    .dot-green { width:8px; height:8px; border-radius:50%; background:#22C55E; flex-shrink:0; }
    .dot-red   { width:8px; height:8px; border-radius:50%; background:#EF4444; flex-shrink:0; }
    .status-text { font-size: 0.75rem; color: #A1A1AA; }
    .page-hero {
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #18181B;
        margin-bottom: 2rem;
    }
    .page-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.75rem;
        font-weight: 800;
        color: #FAFAFA;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0;
    }
    .page-sub {
        font-size: 0.875rem;
        color: #71717A;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .fmt-tag {
        background: #27272A;
        color: #A1A1AA;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stDateInput > div > div > input {
        background: #27272A !important;
        border: 1px solid #3F3F46 !important;
        border-radius: 8px !important;
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.15) !important;
    }
    .stTextInput > label,
    .stTextArea > label,
    .stDateInput > label,
    .stSelectbox > label {
        color: #A1A1AA !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stSelectbox > div > div {
        background: #27272A !important;
        border: 1px solid #3F3F46 !important;
        border-radius: 8px !important;
        color: #FAFAFA !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.6rem 1.25rem !important;
        transition: opacity 0.15s !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #27272A !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #71717A !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.825rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FAFAFA !important;
        border-bottom-color: #7C3AED !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding: 1.25rem 0 !important;
    }
    .stDataFrame {
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 1.25rem 0;
    }
    .metric-card {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        text-align: center;
    }
    .metric-num {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: #71717A;
        font-weight: 500;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .step-bar {
        display: flex;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }
    .step-item {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        background: #18181B;
        border: 1px solid #22C55E33;
        font-size: 0.75rem;
        font-weight: 600;
        color: #22C55E;
    }
    .r-card {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .r-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .r-topic { font-weight: 600; font-size: 0.95rem; color: #FAFAFA; }
    .r-score {
        background: linear-gradient(135deg, #7C3AED22, #4F46E522);
        border: 1px solid #7C3AED44;
        color: #A78BFA;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }
    .r-meta { font-size: 0.78rem; color: #71717A; margin-top: 0.3rem; }
    .r-excerpt {
        font-size: 0.8rem;
        color: #52525B;
        margin-top: 0.6rem;
        padding-top: 0.6rem;
        border-top: 1px solid #27272A;
        line-height: 1.5;
    }
    .stAlert { border-radius: 8px !important; }
    [data-testid="stInfo"] {
        background: #1E1B4B !important;
        border: 1px solid #3730A3 !important;
        border-radius: 8px !important;
        color: #C7D2FE !important;
    }
    [data-testid="stSuccess"] {
        background: #052E16 !important;
        border: 1px solid #166534 !important;
        border-radius: 8px !important;
    }
    [data-testid="stError"] {
        background: #450A0A !important;
        border: 1px solid #991B1B !important;
        border-radius: 8px !important;
    }
    [data-testid="stWarning"] {
        background: #431407 !important;
        border: 1px solid #9A3412 !important;
        border-radius: 8px !important;
    }
    .stExpander {
        background: #18181B !important;
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
    }
    .stExpander summary { color: #A1A1AA !important; }
    [data-testid="stFileUploader"] {
        background: #18181B !important;
        border: 2px dashed #3F3F46 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Browse files" !important;
        font-size: 0.875rem !important;
    }
    .stSpinner > div { border-top-color: #7C3AED !important; }
    .stRadio > div { gap: 0.5rem !important; }
    .stRadio label {
        background: #18181B !important;
        border: 1px solid #27272A !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        color: #A1A1AA !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
    }
    .stRadio label:has(input:checked) {
        background: #3730A3 !important;
        border-color: #4F46E5 !important;
        color: #FAFAFA !important;
    }
    hr { border-color: #27272A !important; }
    .stDownloadButton > button {
        background: #27272A !important;
        color: #A1A1AA !important;
        border: 1px solid #3F3F46 !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }
    .stDownloadButton > button:hover {
        background: #3F3F46 !important;
        color: #FAFAFA !important;
    }
    .stTextArea > div > div > textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 0.82rem !important;
        line-height: 1.7 !important;
        background: #18181B !important;
        color: #D4D4D8 !important;
        border: 1px solid #27272A !important;
    }
    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        color: #52525B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #27272A;
    }
    .decision-item {
        background: #18181B;
        border: 1px solid #27272A;
        border-left: 3px solid #7C3AED;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: #FAFAFA;
        font-weight: 500;
    }
    .decision-rationale {
        font-size: 0.78rem;
        color: #71717A;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    .question-item {
        background: #18181B;
        border: 1px solid #27272A;
        border-left: 3px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: #FAFAFA;
    }
    .question-raised {
        font-size: 0.78rem;
        color: #71717A;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================

def api_post(endpoint, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", **kwargs)
        return r.status_code, r.json()
    except requests.ConnectionError:
        st.error("Cannot reach API. Start it with: `uvicorn api.main:app --reload`")
        st.stop()

def api_get(endpoint, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", **kwargs)
        return r.status_code, r.json()
    except requests.ConnectionError:
        st.error("Cannot reach API. Start it with: `uvicorn api.main:app --reload`")
        st.stop()

def api_delete(endpoint):
    try:
        r = requests.delete(f"{API_BASE}{endpoint}")
        return r.status_code, r.json()
    except requests.ConnectionError:
        st.error("Cannot reach API.")
        st.stop()

def check_api_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except:
        return False


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.1rem">
            <div style="
                width:32px;height:32px;border-radius:8px;
                background:linear-gradient(135deg,#7C3AED,#4F46E5);
                display:flex;align-items:center;justify-content:center;
                font-size:1rem;flex-shrink:0;
            ">🧠</div>
            <div class="logo-name">MeetMind</div>
        </div>
        <div class="logo-tagline">AI Meeting Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📥  New Meeting", "🔍  Search", "📋  History"],
        label_visibility="collapsed",
    )

    healthy = check_api_health()
    if healthy:
        st.markdown("""
        <div class="status-row">
            <div class="dot-green"></div>
            <span class="status-text">API connected</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-row">
            <div class="dot-red"></div>
            <span class="status-text">API offline</span>
        </div>""", unsafe_allow_html=True)
        st.caption("`uvicorn api.main:app --reload`")


# ==============================================================================
# PAGE 1 — NEW MEETING
# ==============================================================================

if page == "📥  New Meeting":

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">New Meeting</div>
        <div class="page-sub">Paste a transcript or upload audio — get action items, decisions, and a summary email in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Input type",
        ["📋  Paste Transcript", "🎵  Upload Audio"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_date, col_proj = st.columns(2)
    with col_date:
        meeting_date = st.date_input(
            "Meeting Date *",
            value=date.today(),
            help="Date the meeting took place",
        ).isoformat()
    with col_proj:
        project_name = st.text_input(
            "Project / Team (optional)",
            placeholder="e.g. backend, product, mobile",
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # PATH A — Paste transcript
    if "Paste" in input_mode:
        transcript_input = st.text_area(
            "Transcript *",
            height=240,
            placeholder="John: We need to decide on the database. Sarah suggested PostgreSQL.\nJohn: Agreed. Sarah will set it up by Friday.\nSarah: Should we add Redis for caching at launch?\nJohn: Let's defer that — open question for next sprint.",
            help="Paste your raw meeting notes or auto-generated transcript here",
        )

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        if st.button("⚡ Process Meeting", type="primary", use_container_width=True):
            if not transcript_input.strip():
                st.warning("Transcript cannot be empty.")
            else:
                with st.spinner("Analysing transcript — extracting structure and generating summary..."):
                    status, data = api_post(
                        "/process",
                        json={
                            "text": transcript_input,
                            "meeting_date": meeting_date,
                            "project_name": project_name or None,
                        },
                    )
                if status == 200:
                    st.session_state["result"] = data
                    st.success("Meeting processed and saved.")
                else:
                    st.error(f"Processing failed: {data.get('detail', 'Unknown error')}")

    # PATH B — Upload audio
    else:
        
        st.markdown("""
        <div style="margin-bottom:0.5rem">
            <div class="input-label">Audio File <span class="required-star">✱</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Upload Audio File",
            type=["mp3", "wav", "m4a", "webm", "mp4"],
            help="Transcribed via Groq Whisper — free, no cost per minute",
        )

        st.markdown("""
        <div style="display:flex;gap:0.4rem;margin:0.5rem 0 1rem 0;flex-wrap:wrap">
            <span class="fmt-tag">mp3</span>
            <span class="fmt-tag">wav</span>
            <span class="fmt-tag">m4a</span>
            <span class="fmt-tag">webm</span>
            <span class="fmt-tag">mp4</span>
        </div>
        """, unsafe_allow_html=True)

        if uploaded:
            st.markdown(f"<div style='font-size:0.8rem;color:#A78BFA;margin-bottom:0.75rem'>Attached: {uploaded.name} — {round(len(uploaded.getvalue())/1024, 1)} KB</div>", unsafe_allow_html=True)

            if st.button("🎙️ Transcribe & Process", type="primary", use_container_width=True):
                with st.spinner(f"Transcribing {uploaded.name} via Groq Whisper..."):
                    status, data = api_post(
                        "/process/audio",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        params={"meeting_date": meeting_date, "project_name": project_name or None},
                    )
                if status == 200:
                    st.session_state["result"] = data
                    st.success("Transcription and processing complete.")
                else:
                    st.error(f"Failed: {data.get('detail', 'Unknown error')}")

    # RESULTS
    if "result" in st.session_state:
        result = st.session_state["result"]
        extraction = result.get("extraction", {})

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("""
        <div class="step-bar">
            <div class="step-item"><span>✓</span> Transcribed</div>
            <div class="step-item"><span>✓</span> Extracted</div>
            <div class="step-item"><span>✓</span> Stored</div>
            <div class="step-item"><span>✓</span> Summarized</div>
        </div>
        """, unsafe_allow_html=True)

        n_a = len(extraction.get("action_items", []))
        n_d = len(extraction.get("decisions", []))
        n_q = len(extraction.get("open_questions", []))

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-num">{n_a}</div>
                <div class="metric-lbl">Action Items</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{n_d}</div>
                <div class="metric-lbl">Decisions</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{n_q}</div>
                <div class="metric-lbl">Open Questions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        topic = extraction.get("meeting_topic") or "Meeting"
        participants = extraction.get("participants", [])
        st.markdown(f"**{topic}**  ·  <span style='color:#71717A;font-size:0.875rem'>{', '.join(participants) if participants else 'Participants not detected'}</span>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["✅  Action Items", "⚖️  Decisions", "❓  Open Questions", "📧  Email Summary"])

        with tab1:
            items = extraction.get("action_items", [])
            if items:
                rows = [{"Task": a.get("task",""), "Owner": a.get("owner") or "TBD", "Deadline": a.get("deadline") or "Not set", "Priority": a.get("priority") or "—"} for a in items]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No action items identified in this transcript.")

        with tab2:
            decisions = extraction.get("decisions", [])
            if decisions:
                for d in decisions:
                    rationale_html = f'<div class="decision-rationale">Rationale: {d["rationale"]}</div>' if d.get("rationale") else ""
                    st.markdown(f'<div class="decision-item">{d["description"]}{rationale_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No decisions identified in this transcript.")

        with tab3:
            questions = extraction.get("open_questions", [])
            if questions:
                for q in questions:
                    raised_html = f'<div class="question-raised">Raised by: {q["raised_by"]}</div>' if q.get("raised_by") else ""
                    st.markdown(f'<div class="question-item">{q["question"]}{raised_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No open questions identified in this transcript.")

        with tab4:
            email = result.get("email_summary", "")
            if email:
                st.text_area("", value=email, height=380, label_visibility="collapsed")
                st.download_button(
                    "Download .txt",
                    data=email,
                    file_name=f"meeting_summary_{result.get('meeting_date','today')}.txt",
                    mime="text/plain",
                )
            else:
                st.info("No summary available.")

        with st.expander("View raw transcript"):
            st.text_area(
                "",
                value=result.get("transcript", "No transcript available."),
                height=300,
                label_visibility="collapsed",
                disabled=True,
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Clear — start new meeting", key="clear_btn"):
            del st.session_state["result"]
            st.rerun()


# ==============================================================================
# PAGE 2 — SEARCH
# ==============================================================================

elif page == "🔍  Search":

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">Search Meetings</div>
        <div class="page-sub">Ask anything in plain English. Semantic search finds relevant meetings even when exact words don't match.</div>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search query *",
            placeholder="What did we decide about the database?",
        )
    with col_n:
        n_results = st.selectbox("Results", [3, 5, 10], index=1)

    col_proj, col_btn = st.columns([3, 1])
    with col_proj:
        project_filter = st.text_input(
            "Filter by project (optional)",
            placeholder="e.g. backend",
        )
    with col_btn:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_btn")

    # Run search and store results in session state
    if search_clicked:
        if not query.strip():
            st.warning("Enter a search query first.")
        else:
            params = {"q": query, "n": n_results}
            if project_filter.strip():
                params["project"] = project_filter.strip()

            with st.spinner("Searching across all meetings..."):
                status, data = api_get("/search", params=params)

            if status == 200:
                # Store in session state so results survive button reruns
                st.session_state["search_results"] = data.get("results", [])
                st.session_state["search_query"] = query
            else:
                st.error(f"Search failed: {data.get('detail')}")

    # Render results from session state — persists across reruns
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        query_used = st.session_state.get("search_query", "")
        count = len(results)

        if count == 0:
            st.info("No results found. Try a different query or process some meetings first.")
        else:
            st.markdown(f"<div style='font-size:0.8rem;color:#71717A;margin-bottom:1rem'>{count} result{'s' if count!=1 else ''} for <strong style='color:#A78BFA'>\"{query_used}\"</strong></div>", unsafe_allow_html=True)

            for r in results:
                score_pct = int(r.get("relevance_score", 0) * 100)
                st.markdown(f"""
                <div class="r-card">
                    <div class="r-top">
                        <div class="r-topic">{r.get('topic','Unknown')}</div>
                        <div class="r-score">{score_pct}% match</div>
                    </div>
                    <div class="r-meta">
                       📅 {r.get('date','—')} &nbsp;·&nbsp;
                       👥 {r.get('participants','—')} &nbsp;·&nbsp;
                       🏷️ {r.get('project','general')} &nbsp;·&nbsp;
                       ✅ {r.get('action_count',0)} actions &nbsp;·&nbsp;
                       ⚖️ {r.get('decision_count',0)} decisions
                    </div>
                    <div class="r-excerpt">{r.get('excerpt','')[:280]}...</div>
                </div>
                """, unsafe_allow_html=True)

                mid = r.get("meeting_id")
                if st.button("Get email summary", key=f"sum_{mid}"):
                    with st.spinner("Generating summary..."):
                        ss, sd = api_get(f"/summary/{mid}")
                    if ss == 200:
                        st.text_area(
                            "",
                            value=sd["email_summary"],
                            height=360,
                            label_visibility="collapsed",
                            key=f"email_{mid}",
                        )
                    else:
                        st.error(f"Summary failed: {sd.get('detail')}")


# ==============================================================================
# PAGE 3 — HISTORY
# ==============================================================================

elif page == "📋  History":

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">Meeting History</div>
        <div class="page-sub">All stored meetings — sorted by date, most recent first.</div>
    </div>
    """, unsafe_allow_html=True)

    status, data = api_get("/meetings", params={"limit": 100})

    if status != 200:
        st.error(f"Could not load meetings: {data.get('detail')}")
        st.stop()

    meetings = data.get("meetings", [])
    total = data.get("count", 0)

    if total == 0:
        st.info("No meetings stored yet. Go to New Meeting to process your first one.")
        st.stop()

    total_actions = sum(m.get("action_count", 0) for m in meetings)
    total_decisions = sum(m.get("decision_count", 0) for m in meetings)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-num">{total}</div>
            <div class="metric-lbl">Meetings Stored</div>
        </div>
        <div class="metric-card">
            <div class="metric-num">{total_actions}</div>
            <div class="metric-lbl">Action Items</div>
        </div>
        <div class="metric-card">
            <div class="metric-num">{total_decisions}</div>
            <div class="metric-lbl">Decisions Made</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>All Meetings</div>", unsafe_allow_html=True)

    df_rows = [{
        "Date": m.get("date","—"),
        "Topic": m.get("topic","—"),
        "Participants": m.get("participants","—"),
        "Project": m.get("project","general"),
        "Actions": m.get("action_count",0),
        "Decisions": m.get("decision_count",0),
    } for m in meetings]

    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Meeting Actions</div>", unsafe_allow_html=True)

    col_sel, col_act = st.columns([3, 2])
    with col_sel:
        selected_id = st.selectbox(
            "Select meeting",
            options=[m.get("meeting_id") for m in meetings],
            format_func=lambda mid: next(
                (f"{m['date']} — {m['topic']}" for m in meetings if m["meeting_id"] == mid), mid
            ),
        )
    with col_act:
        action = st.radio("Action", ["Get Summary", "Delete Meeting"], horizontal=True)

    if st.button("Run", type="primary", key="history_run"):
        if action == "Get Summary":
            with st.spinner("Generating summary..."):
                s, d = api_get(f"/summary/{selected_id}")
            if s == 200:
                st.text_area("", value=d["email_summary"], height=380, label_visibility="collapsed")
            else:
                st.error(f"Failed: {d.get('detail')}")
        elif action == "Delete Meeting":
            s, d = api_delete(f"/meetings/{selected_id}")
            if s == 200:
                st.success("Meeting deleted.")
                st.rerun()
            else:
                st.error(f"Delete failed: {d.get('detail')}")