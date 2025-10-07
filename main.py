from __future__ import annotations

"""
Scene Reenactment Bot — FastAPI (single file)

What it does
------------
- Ingest a chapter text + learner profile/goal
- Builds a chapter-scoped, character-led scene plan (no tutor voice)
- Runs interactive, in-character role-play that reenacts scenes (not Q&A)
- Tracks soft signals silently (empathy, text evidence, bias check, curiosity)
- Provides a single end-of-session feedback summary upon /end

Keep-alives
-----------
- Uses Hugging Face InferenceClient exactly (no change to how HF is called)
- Stays strictly within chapter via snippet retrieval + forced citations [C#]

Run
---
$ export HF_MODEL=deepseek-ai/DeepSeek-V3-0324  # or your model
$ export HF_TOKEN=...  # if needed
$ uvicorn app:app --reload --port 8000

Dependencies
------------
fastapi, pydantic, huggingface_hub, scikit-learn (optional), uvicorn
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

import os, re, time, json

# =====================
# Model / client setup
# =====================
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3-0324")
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else InferenceClient()

app = FastAPI(title="Scene Reenactment Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# State & storage
# =====================
@dataclass
class Signals:
    empathy: float = 0.0
    text_evidence: float = 0.0
    bias_check: float = 0.0
    curiosity: float = 0.0

@dataclass
class TutorState:
    user_id: str
    profile: Dict[str, Any]
    chapter_title: str
    chapter_source: Optional[str]
    chapter_summary: str
    learning_objectives: List[str]
    roleplay_persona: str
    chunks: List[str] = field(default_factory=list)
    index: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    exercise_queue: List[Dict[str, Any]] = field(default_factory=list)
    signals: Signals = field(default_factory=Signals)
    signal_log: List[Dict[str, float]] = field(default_factory=list)
    scene: str = "opening"
    scene_step: int = 0
    scenario: Dict[str, Any] = field(default_factory=dict)
    last_prompt_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ended: bool = False

SESSION: Dict[str, TutorState] = {}

# =====================
# Pydantic I/O models
# =====================
class Profile(BaseModel):
    background: str = Field(..., description="User background/context")
    goals: str = Field(..., description="Learning goal in user's words")
    level: Optional[str] = Field(None, description="Self-assessed level")

class IngestRequest(BaseModel):
    user_id: str
    chapter_title: str
    chapter_text: str
    chapter_source: Optional[str] = None
    profile: Profile

class IngestResponse(BaseModel):
    user_id: str
    lesson_plan: Dict[str, Any]
    first_prompt: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    reply: str
    state: Dict[str, Any]
    new_exercise: Optional[Dict[str, Any]] = None
    citations: Optional[List[str]] = None

class EndRequest(BaseModel):
    user_id: str

class EndResponse(BaseModel):
    user_id: str
    final_feedback: Dict[str, Any]

# =====================
# Utilities — chunking, retrieval, HF call, parsing
# =====================

def _denest_if_needed(data: Dict[str, Any]) -> Dict[str, Any]:
    """If the model double-serialized the full object into data['reply'], unwrap it."""
    try:
        r = data.get("reply")
        if isinstance(r, str):
            s = r.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("```") and s.endswith("```")):
                inner = _safe_json_loads(s)  # your existing tolerant parser
                # If the inner looks like the real payload, use it
                if isinstance(inner, dict) and ("reply" in inner or "new_exercise" in inner or "citations" in inner):
                    return inner
    except Exception:
        pass
    return data

def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        punct = max(text.rfind(p, i, end) for p in [".", "?", "!", ";"])
        if punct <= i:
            punct = end
        chunks.append(text[i:punct].strip())
        i = max(punct - overlap, i + 1)
    return [c for c in chunks if c]


def build_index(chunks: List[str]) -> Dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        X = vec.fit_transform(chunks)
        return {"type": "tfidf", "vec": vec, "X": X}
    except Exception:
        return {"type": "fallback"}


def retrieve(question: str, chunks: List[str], index: Dict[str, Any], k: int = 3) -> List[tuple[str, str]]:
    if not chunks:
        return []
    if index.get("type") == "tfidf":
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        qv = index["vec"].transform([question])
        sims = cosine_similarity(qv, index["X"]).ravel()
        order = sims.argsort()[::-1][:k]
    else:
        q = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
        scores = [(i, len(q & set(re.findall(r"[a-z0-9]+", c.lower())))) for i, c in enumerate(chunks)]
        order = [i for i, _ in sorted(scores, key=lambda t: t[1], reverse=True)[:k]]
    return [(f"C{idx+1}", chunks[idx]) for idx in order]


def _hf_chat(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 800) -> str:
    completion = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = completion.choices[0].message
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _safe_json_loads(s: str) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    text = (s or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {"reply": text, "new_exercise": None, "citations": []}


def _state_public_view(state: TutorState) -> Dict[str, Any]:
    return {
        "chapter_title": state.chapter_title,
        "chapter_source": state.chapter_source,
        "chapter_summary": state.chapter_summary,
        "learning_objectives": state.learning_objectives,
        "roleplay_persona": state.roleplay_persona,
        "signals_preview": asdict(state.signals),
        "exercise_queue_len": len(state.exercise_queue),
    }

# =====================
# Prompts
# =====================
REFUSAL_PREFIX = "I’m limited to the provided chapter"

PLANNER_SYSTEM = (
    """
        You build a short, interactive, in-world reenactment strictly within the supplied CHAPTER SNIPPETS.
        - Choose ONE bot character who plausibly appears/speaks in this chapter (or an in-world narrator like a child/neighbor). No tutors, no meta.
        - Provide a 3–5 sentence chapter_summary grounded ONLY in snippets; use [C#] when referencing specifics.
        - Derive 2–4 learning_objectives aligned to the learner's goal.
        - Create the FIRST role-play exercise that opens IN CHARACTER (one line + ONE question). No teaching voice.
        - Exercise must be:
        - type: "roleplay"
        - instructions: what the learner tries to do in-scene
        - prompt: opening situation (in-character)
        - choices: null (prefer free response)
        - answer: concise rubric (OOC) for debriefing later
        - skills: 2–4 tags (e.g., empathy, text_evidence, bias_check, curiosity)
        - deadline_sec: 45–90s if tension fits (else null)
        Output JSON ONLY with keys: chapter_summary, learning_objectives, roleplay_persona, exercise
            """
        ).strip()

RUNTIME_SYSTEM = (
            """
        You are a chapter-scoped, in-world character for a role-play.

        Scope rules (hard):
        - Use ONLY the CHAPTER SNIPPETS provided.
        - If insufficient, reply: "I’m limited to the provided chapter and don’t have enough information in it to answer that. I can help you explore what the chapter does cover."
        - Never invent facts. Any claim about the chapter must be supportable by snippets; include citations like [C1], [C2].

        Contract:
        - Speak fully IN CHARACTER (dialogue/action). No meta, no teaching voice.
        - One turn = ONE realistic line or action + optionally ONE question.
        - Keep things immersive; if the user goes outside the chapter, redirect gently in-character back to provided events.

        Output JSON ONLY (no code fences):
        {{
        "reply": "<in-character utterance only>",
        "citations": ["C1","C2"] | [],
        "new_exercise": {{
            "type": "roleplay",
            "instructions": "<what the learner is trying to do in-scene>",
            "prompt": "<next in-character situation or question>",
            "choices": null,
            "answer": "<concise rubric/criteria (OOC, for later debrief)>",
            "skills": ["empathy","text_evidence","bias_check","curiosity"],
            "deadline_sec": 60 | null,
            "branch": "<probe_deeper|raise_time_pressure|de_escalate|switch_perspective|escalate_objection|close_out>|null"
        }} | null,
        "internal_log_patch": {{"empathy": -1..+1, "text_evidence": -1..+1, "bias_check": -1..+1, "curiosity": -1..+1}}
        }}
            """
        ).strip()

DEBRIEF_SYSTEM = (
    """
        You produce a SINGLE end-of-session feedback based solely on:
        - the session transcript (in-character dialogue)
        - tracked signals (empathy, text_evidence, bias_check, curiosity)
        - the learner goal
        - the provided chapter snippets

        Rules (hard):
        - Cite ONLY this chapter via [C#] from the supplied snippet IDs when making text-based claims.
        - No long lectures; keep it concise and actionable.

        Output JSON ONLY:
        {{
        "summary": "2–4 sentences: what the learner practiced/experienced",
        "strengths": ["...","..."],
        "growth": ["one or two concrete next steps"],
        "chapter_evidence": [{{"quote_or_paraphrase": "...", "why_it_matters": "...", "citations": ["C#"]}}],
        "metrics": {{"empathy": 0..1, "text_evidence": 0..1, "bias_check": 0..1, "curiosity": 0..1}}
        }}
            """
        ).strip()

# =====================
# Ingest
# =====================
class _ProfileShim(BaseModel):
    background: str
    goals: str
    level: str | None = None

def _decode_text_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")


def _ingest_core(user_id: str, chapter_title: str, chapter_text: str,
                 chapter_source: str | None, profile: dict[str, Any]) -> IngestResponse:
    chunks = chunk_text(chapter_text)
    index = build_index(chunks)

    # Small evidence pack for planning
    top = retrieve(f"summary + objectives for: {chapter_title}", chunks, index, k=3)
    sources_block = "\n".join([f"[{cid}] {txt}" for cid, txt in top])

    planner_user = f"""
        Learner profile:
        - Background: {profile.get('background')}
        - Goals: {profile.get('goals')}
        - Level: {profile.get('level') or 'unspecified'}

        CHAPTER SNIPPETS (the ONLY allowed knowledge):
        {sources_block}

        Build the reenactment plan as specified.
        """.strip()

    raw = _hf_chat(
        [{"role": "system", "content": PLANNER_SYSTEM},
         {"role": "user", "content": planner_user}],
        temperature=0.15, max_tokens=900
    )
    data = _safe_json_loads(raw)

    # Fallbacks if model under-fills
    lesson = {
        "chapter_summary": data.get("chapter_summary") or "(No summary returned)",
        "learning_objectives": data.get("learning_objectives") or ["Explore the scene", "Practice grounded interpretation"],
        "roleplay_persona": data.get("roleplay_persona") or "In-world narrator",
        "exercise": data.get("exercise") or {
            "type": "roleplay",
            "instructions": "Enter the scene and describe what you do next.",
            "prompt": "(In-character) You stand at the edge of the scene. What do you do?",
            "choices": None,
            "answer": "Responds in-character and grounded in snippet details.",
            "skills": ["curiosity"],
            "deadline_sec": None
        }
    }

    state = TutorState(
        user_id=user_id,
        profile=profile,
        chapter_title=chapter_title,
        chapter_source=chapter_source,
        chapter_summary=lesson["chapter_summary"],
        learning_objectives=lesson["learning_objectives"],
        roleplay_persona=lesson["roleplay_persona"],
        chunks=chunks,
        index=index,
        exercise_queue=[lesson["exercise"]],
        last_prompt_at=time.time(),
    )
    SESSION[user_id] = state

    first_prompt = lesson["exercise"].get("prompt") or lesson["exercise"].get("instructions") or "(In-character) Let's begin."
    lesson_plan = {
        "chapter_summary": state.chapter_summary,
        "learning_objectives": state.learning_objectives,
        "roleplay_persona": state.roleplay_persona,
        "first_exercise": lesson["exercise"],
    }
    return IngestResponse(user_id=user_id, lesson_plan=lesson_plan, first_prompt=first_prompt)

# =====================
# Endpoints
# =====================
@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    return _ingest_core(
        user_id=req.user_id,
        chapter_title=req.chapter_title,
        chapter_text=req.chapter_text,
        chapter_source=req.chapter_source,
        profile=req.profile.model_dump(),
    )


@app.post("/ingest_file", response_model=IngestResponse)
async def ingest_file(
    user_id: str = Form(...),
    chapter_title: str = Form(...),
    profile_background: str = Form(...),
    profile_goals: str = Form(...),
    profile_level: Optional[str] = Form(None),
    chapter_source: Optional[str] = Form(None),
    chapter_file: UploadFile = File(...),
):
    if chapter_file.filename and not chapter_file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt file.")
    raw = await chapter_file.read()
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Text file too large (max 2MB).")

    chapter_text = _decode_text_file(raw).strip()
    if not chapter_text:
        raise HTTPException(status_code=400, detail="Empty or unreadable text file.")

    if not chapter_title:
        first_line = next((ln.strip() for ln in chapter_text.splitlines() if ln.strip()), "")
        chapter_title = (first_line[:80] or "Untitled Chapter")

    profile = {"background": profile_background, "goals": profile_goals, "level": profile_level or "unspecified"}

    return _ingest_core(
        user_id=user_id,
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        chapter_source=chapter_source,
        profile=profile,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = SESSION.get(req.user_id)
    if not state:
        raise HTTPException(status_code=404, detail="No active session. Call /ingest first.")
    if state.ended:
        raise HTTPException(status_code=400, detail="Session already ended. Start a new session or call /reset.")

    # Retrieve chapter snippets relevant to the user's turn
    top = retrieve(req.message, state.chunks, state.index, k=3)
    if not top:
        if state.chunks:
            top = [(f"C{i+1}", c) for i, c in enumerate(state.chunks[:3])]
        else:
            return ChatResponse(
                user_id=req.user_id,
                reply="I don’t have any snippets from this chapter yet. Please re-ingest the chapter.",
                state=_state_public_view(state),
            )

    sources_block = "\n".join([f"[{cid}] {txt}" for cid, txt in top])

    # User message payload for runtime
    user_payload = f"""
        Learner:
        - Background: {state.profile.get('background')}
        - Goals: {state.profile.get('goals')}
        - Level: {state.profile.get('level') or 'unspecified'}

        Chapter: {state.chapter_title}
        CHAPTER SNIPPETS (the ONLY allowed knowledge):
        {sources_block}

        User says (in-scene): "{req.message}"

        Return internal_log_patch to reflect the learner’s last move (approximate). Do NOT include any coaching or assessment.
        """.strip()

    raw = _hf_chat(
        [{"role": "system", "content": RUNTIME_SYSTEM},
         {"role": "user", "content": user_payload}],
        temperature=0.12, max_tokens=700
    )
    data = _safe_json_loads(raw)
    data = _denest_if_needed(data)
    # Enforce citations (soft retry)
    allowed = [cid for cid, _ in top]
    def _valid_citations(cites: Any) -> bool:
        return isinstance(cites, list) and (not cites or all(c in allowed for c in cites))

    if not _valid_citations(data.get("citations")) and not (data.get("reply", "").startswith(REFUSAL_PREFIX)):
        retry_msg = user_payload + "\n\nReminder: You MUST cite only from these snippet IDs and you MUST include 'citations'."
        raw2 = _hf_chat(
            [{"role": "system", "content": RUNTIME_SYSTEM}, {"role": "user", "content": retry_msg}],
            temperature=0.12, max_tokens=700
        )
        data2 = _safe_json_loads(raw2)
        data2 = _denest_if_needed(data2)
        
        if _valid_citations(data2.get("citations")) or data2.get("reply", "").startswith(REFUSAL_PREFIX):
            data = data2

    # Silent signal tracking
    ilog = data.get("internal_log_patch") or {}
    def _clamp(v):
        try:
            return max(-1.0, min(1.0, float(v)))
        except:
            return 0.0
    patch = {
        "empathy": _clamp(ilog.get("empathy", 0)),
        "text_evidence": _clamp(ilog.get("text_evidence", 0)),
        "bias_check": _clamp(ilog.get("bias_check", 0)),
        "curiosity": _clamp(ilog.get("curiosity", 0)),
    }
    state.signals.empathy += patch["empathy"]
    state.signals.text_evidence += patch["text_evidence"]
    state.signals.bias_check += patch["bias_check"]
    state.signals.curiosity += patch["curiosity"]
    state.signal_log.append(patch)

    # Force roleplay type for any exercise
    new_ex = data.get("new_exercise")
    if isinstance(new_ex, dict) and new_ex.get("type") != "roleplay":
        new_ex["type"] = "roleplay"

    if isinstance(new_ex, dict):
        state.exercise_queue.append(new_ex)
        state.last_prompt_at = time.time()

    # Maintain short history
    state.history.append({"user": req.message, "assistant": data.get("reply", "")})
    state.history = state.history[-14:]
    state.updated_at = time.time()

    payload = {
        "user_id": req.user_id,
        "reply": data.get("reply", "(No reply)"),
        "state": _state_public_view(state),
        "citations": data.get("citations"),
    }
    if new_ex:
        payload["new_exercise"] = new_ex
    return payload


@app.post("/end", response_model=EndResponse)
def end_session(req: EndRequest):
    state = SESSION.get(req.user_id)
    if not state:
        raise HTTPException(status_code=404, detail="No active session. Call /ingest first.")

    if state.ended:
        return EndResponse(user_id=req.user_id, final_feedback={"summary": "(already ended)", "strengths": [], "growth": [], "chapter_evidence": [], "metrics": asdict(state.signals)})

    transcript_text = "\n".join([f"User: {t.get('user','')}\nBot: {t.get('assistant','')}" for t in state.history])[-3500:]

    top = retrieve("Session recap and justification", state.chunks, state.index, k=5)
    if not top and state.chunks:
        top = [(f"C{i+1}", c) for i, c in enumerate(state.chunks[:5])]
    evidence_block = "\n".join([f"[{cid}] {txt}" for cid, txt in top])

    # squash signals roughly into [0,1]
    s = state.signals
    signals_norm = {
        "empathy": max(0.0, min(1.0, (s.empathy + 3) / 6)),
        "text_evidence": max(0.0, min(1.0, (s.text_evidence + 3) / 6)),
        "bias_check": max(0.0, min(1.0, (s.bias_check + 3) / 6)),
        "curiosity": max(0.0, min(1.0, (s.curiosity + 3) / 6)),
    }

    user_msg = f"""
        Learner goal: {state.profile.get('goals')}
        Transcript (trimmed):
        {transcript_text}

        Signals (approx):
        {json.dumps(signals_norm)}

        CHAPTER SNIPPETS (allowed citations only):
        {evidence_block}

        Return JSON only as specified.
        """.strip()

    raw = _hf_chat(
        [{"role": "system", "content": DEBRIEF_SYSTEM}, {"role": "user", "content": user_msg}],
        temperature=0.2, max_tokens=700
    )
    data = _safe_json_loads(raw)
    data.setdefault("metrics", signals_norm)

    state.ended = True
    state.updated_at = time.time()

    return EndResponse(user_id=req.user_id, final_feedback=data)


@app.get("/state/{user_id}")
def get_state(user_id: str) -> Dict[str, Any]:
    state = SESSION.get(user_id)
    if not state:
        raise HTTPException(status_code=404, detail="No active session for this user.")
    return _state_public_view(state)


@app.post("/reset/{user_id}")
def reset(user_id: str):
    if user_id in SESSION:
        del SESSION[user_id]
    return {"ok": True}
