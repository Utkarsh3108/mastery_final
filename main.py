from __future__ import annotations

# HF Tutor Bot — single-file FastAPI app that:
# 1) Ingests a chapter + user profile
# 2) Builds a customized lesson plan
# 3) Runs interactive tutoring with role‑play + exercises
# 4) Tracks simple progress in-memory (swap with Redis/DB for prod)

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
import json
import os
import time
import re
from fastapi.middleware.cors import CORSMiddleware

############################################################
# Model / client setup
############################################################
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3-0324")
HF_TOKEN = os.getenv("HF_TOKEN")

# If your HF endpoint requires a token, set HF_TOKEN in env
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else InferenceClient()

app = FastAPI(title="HF Chat Tutor Bot", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your LAN IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
############################################################
# In-memory store (replace with Redis/Postgres in production)
############################################################
@dataclass
class Progress:
    # naive progress model: 0-100 mastery, plus streaks
    mastery: int = 0
    correct_in_a_row: int = 0
    attempts: int = 0

@dataclass
class TutorState:
    user_id: str
    profile: Dict[str, Any]
    chapter_title: str
    chapter_source: Optional[str]
    chapter_summary: str
    learning_objectives: List[str]
    roleplay_persona: str
    exercise_queue: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)  # minimal transcript
    progress: Progress = field(default_factory=Progress)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # NEW: scenario & runtime
    scenario: Dict[str, Any] = field(default_factory=dict)
    last_prompt_at: float = field(default_factory=time.time)
    achievements: List[str] = field(default_factory=list)

    # (already present) retrieval
    chunks: List[str] = field(default_factory=list)
    index: Dict[str, Any] = field(default_factory=dict)

# crude in-memory map; replace with a persistent store
SESSION: Dict[str, TutorState] = {}

############################################################
# Pydantic request/response models
############################################################
class Profile(BaseModel):
    background: str = Field(..., description="User background/context")
    goals: str = Field(..., description="Learning goals in user's words")
    level: Optional[str] = Field(
        None, description="Self-assessed level (e.g., beginner/intermediate/advanced)"
    )

class IngestRequest(BaseModel):
    user_id: str
    chapter_title: str
    chapter_text: str
    chapter_source: Optional[str] = None
    profile: Profile

# --- NEW: scenario config sent from UI ---
class ScenarioConfig(BaseModel):
    user_id: str
    book: str
    user_role: str           # e.g., "Audit Associate"
    bot_role: str            # e.g., "Client CFO"
    difficulty: int = 1      # 1..5
    learning_style: str | None = None  # e.g., "practice-first", "examples", "Socratic"
    time_pressure: bool = False
    emotion: str = "supportive"  # "supportive" | "neutral" | "challenging"

class ScenarioSetResponse(BaseModel):
    ok: bool


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
    state:Dict[str, Any] 
    new_exercise: Optional[Dict[str, Any]] = None # redacted/sanitized subset of TutorState for UI

############################################################
# Prompting templates
############################################################

# ==========================
# Chapter-scoped user template + retrieval
# ==========================
CHAT_USER_SCOPED = """
Learner:
- Background: {background}
- Goals: {goals}
- Level: {level}

Chapter: {chapter_title}
Learning objectives: {learning_objectives}

CHAPTER SNIPPETS (the ONLY allowed knowledge):
{sources_block}

User says: "{user_message}"

Follow SYSTEM rules. Output JSON only using the schema. If out-of-scope → use the refusal template.
""".strip()



def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
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

def build_index(chunks: list[str]) -> dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        X = vec.fit_transform(chunks)
        return {"type": "tfidf", "vec": vec, "X": X}
    except Exception:
        return {"type": "fallback"}

def retrieve(question: str, chunks: list[str], index: dict, k: int = 3) -> list[tuple[str, str]]:
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

def _valid_citations(cites: Any, allowed_ids: list[str]) -> bool:
    if not isinstance(cites, list): return False
    if not cites: return False
    return all(c in allowed_ids for c in cites)

REFUSAL_PREFIX = "I’m limited to the provided chapter"


SYSTEM_TUTOR_SCOPED = """
You are "Socratic Coach", a chapter-scoped tutor.

Scope rules (hard):
- Use ONLY the CHAPTER SNIPPETS provided.
- If insufficient, reply: "I’m limited to the provided chapter and don’t have enough information in it to answer that. I can help you explore what the chapter does cover."
- Never invent facts. Every factual statement must map to a snippet; include citations like [C1], [C2].

Pedagogy & scenario:
- Learner role: {user_role}. Your role: {bot_role}. Tone: {emotion}. Difficulty: {difficulty}/5. Learning style preference: {learning_style}.
- Ask ONE focused question at a time. Prefer hints before answers.
- For feedback on answers: explain WHY it’s right/wrong, citing snippets.

Output JSON ONLY (no code fences):
{{
  "reply": "<tutor speaks in-role & tone>",
  "citations": ["C1","C2"] | [],
  "new_exercise": {{
      "type": "mcq|short_answer|coding|roleplay|reflection",
      "instructions": "...",
      "prompt": "...",
      "choices": ["A","B","C","D"] | null,
      "answer": "<expected answer or rubric>" | null,
      "skills": ["concept1","concept2"],
      "deadline_sec": 60 | null,
      "branch": "<probe_deeper|raise_time_pressure|de_escalate|switch_perspective>|null"
  }} | null,
  "assessment": {{"correct": true|false|null, "explanation": "why (with citations)", "delta_mastery": 0}} | null,
  "progress_update": {{"mastery": 0, "correct_in_a_row": 0}} | null
}}
""".strip()

# SYSTEM_TUTOR = (
#     """
# You are "Socratic Coach", a warm, rigorous AI tutor. You adapt teaching to the
# user's profile and goals, keep explanations concise, and prefer questions over monologues.

# Rules:
# - Always use plain language first; add technical depth only as needed.
# - Teach via: micro-explanations → targeted questions → feedback → next action.
# - Incorporate role-play when useful (e.g., interviewer/candidate, teacher/student, clinician/patient).
# - When asking a question, ask ONE focused question at a time.
# - If the user seems stuck, offer a small hint rather than the full answer.
# - Keep turns short and interactive.

# Output format:
# Respond as a JSON object ONLY (no markdown fences), with keys:
# {
#   "reply": "<what the tutor says to the learner>",
#   "new_exercise": {
#       "type": "mcq|short_answer|coding|roleplay|reflection",
#       "instructions": "...",
#       "prompt": "...",
#       "choices": ["A","B","C","D"],
#       "answer": "<expected answer or rubric>",
#       "skills": ["concept1", "concept2"]
#   } | null,
#   "assessment": {
#       "correct": true|false|null,
#       "explanation": "why",
#       "delta_mastery": 0
#   } | null,
#   "progress_update": {"mastery": 0, "correct_in_a_row": 0} | null
# }
# If something doesn't apply, set it to null.
#     """
# ).strip()

# INGEST_USER_MSG = (
#     """
# Build a customized lesson plan from this chapter for the specific learner.

# User profile:
# - Background: {background}
# - Goals: {goals}
# - Level: {level}

# Scenario:
# - Learner role: {{scenario.get("user_role","Learner")}}
# - Your role: {scenario.get("bot_role","Mentor")}
# - Difficulty: {difficulty}/5
# - Learning style: {scenario.get("learning_style","")}
# - Time pressure: {time_pressure}

# Chapter title: {chapter_title}
# Chapter text:
# {chapter_text}

# Return JSON ONLY with keys:
# {{
#   "chapter_summary": "3-5 sentence summary",
#   "learning_objectives": ["objective 1", "objective 2", "objective 3"],
#   "roleplay_persona": "who the tutor pretends to be and why",
#   "exercise": {{
#       "type": "mcq|short_answer|coding|roleplay|reflection",
#       "instructions": "...",
#       "prompt": "...",
#       "choices": ["A","B","C","D"],
#       "answer": "<expected answer or rubric>",
#       "skills": ["concept1", "concept2"]
#   }}
# }}
#     """
# ).strip()

CHAT_USER_MSG = (
    """
Context for this session (summarized):
- Learner background: {background}
- Goals: {goals}
- Level: {level}
- Chapter title: {chapter_title}
- Chapter summary: {chapter_summary}
- Learning objectives: {learning_objectives}
- Your role-play persona: {roleplay_persona}
- Current mastery: {mastery}/100; streak: {streak}

The learner says: "{user_message}"

Follow SYSTEM_TUTOR rules and output JSON ONLY using the specified schema.
    """
).strip()

############################################################
# Utility: model call + robust JSON parsing
############################################################

def _hf_chat(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 800) -> str:
    """Single place to call the HF chat model; returns raw content string."""
    completion = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Normalize to string across providers
    msg = completion.choices[0].message
    content = None
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = getattr(msg, "content", None)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if content is None:
        content = ""
    return str(content)


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Parse model output into JSON.
    - Strips ``` fences
    - Tries direct json.loads
    - Then extracts first {...} block
    - Falls back to a safe wrapper to avoid KeyErrors
    """
    if isinstance(s, dict):
        return s
    text = (s or "").strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Heuristic: grab first {...} block
    try:
        import re
        m = re.search("\\{[\\s\\S]*\\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return {"reply": text, "new_exercise": None, "assessment": None, "progress_update": None}


def _state_public_view(state: TutorState) -> Dict[str, Any]:
    # Redact long fields for UI
    return {
        "chapter_title": state.chapter_title,
        "chapter_source": state.chapter_source,
        "chapter_summary": state.chapter_summary,
        "learning_objectives": state.learning_objectives,
        "roleplay_persona": state.roleplay_persona,
        "progress": asdict(state.progress),
        "exercise_queue_len": len(state.exercise_queue),
    }

############################################################
# Endpoints
############################################################
@app.post("/scenario", response_model=ScenarioSetResponse)
def set_scenario(cfg: ScenarioConfig):
    state = SESSION.get(cfg.user_id)
    if not state:
        # minimal shell so /ingest can fill remaining fields later
        state = TutorState(
            user_id=cfg.user_id,
            profile={"background": "", "goals": "", "level": ""},
            chapter_title="",
            chapter_source=None,
            chapter_summary="",
            learning_objectives=[],
            roleplay_persona="",
        )
        SESSION[cfg.user_id] = state

    state.scenario = cfg.model_dump()
    state.updated_at = time.time()
    return ScenarioSetResponse(ok=True)

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Create/refresh a tutoring session from chapter + profile and return first exercise.
    Chapter-scoped: chunk the chapter, build retriever, and ask the model using only retrieved snippets.
    """
    # 1) Build retrieval artifacts
    chunks = chunk_text(req.chapter_text)
    index = build_index(chunks)

    # 2) Use retrieval to plan (overview)
    overview_q = f"Provide a short summary and learning objectives for: {req.chapter_title}"
    top = retrieve(overview_q, chunks, index, k=3)
    sources_block = "\n".join([f"[{cid}] {txt}" for cid, txt in top])

    # 3) Scenario parameters (if set via /scenario)
    sess = SESSION.get(req.user_id)
    scenario = getattr(sess, "scenario", {}) if sess else {}
    time_pressure = bool(scenario.get("time_pressure", False))
    difficulty = int(scenario.get("difficulty", 1))

    ingest_user = f"""
    User profile:
    - Background: {req.profile.background}
    - Goals: {req.profile.goals}
    - Level: {req.profile.level or "unspecified"}

    Scenario:
    - Learner role: {scenario.get("user_role","Learner")}
    - Your role: {scenario.get("bot_role","Mentor")}
    - Difficulty: {difficulty}/5
    - Learning style: {scenario.get("learning_style","")}
    - Time pressure: {time_pressure}

    CHAPTER SNIPPETS:
    {sources_block}

    Task: Build a lesson plan and the first in-scope exercise STRICTLY from the snippets.
    - Prefer a roleplay-style exercise if appropriate.
    - Include a small 'deadline_sec' (e.g., 45–90s) if time pressure is True.
    - Always attach a concise rubric in 'answer' so we can grade.
    Return only the lesson JSON.
    """.strip()

    sys_msg = SYSTEM_TUTOR_SCOPED.format(
        user_role=scenario.get("user_role","Learner"),
        bot_role=scenario.get("bot_role","Mentor"),
        emotion=scenario.get("emotion","supportive"),
        difficulty=difficulty,
        learning_style=scenario.get("learning_style",""),
    )

    raw = _hf_chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": ingest_user},
        ],
        temperature=0.1,
        max_tokens=900,
    )

    data = _safe_json_loads(raw)
    # Defensive fallback to avoid 5xx
    needed = ("chapter_summary", "learning_objectives", "roleplay_persona", "exercise")
    if any(k not in data for k in needed):
        data = {
            "chapter_summary": data.get("chapter_summary") or "(No summary returned by model.)",
            "learning_objectives": data.get("learning_objectives") or [
                "Grasp the key ideas", "Practice one applied example"
            ],
            "roleplay_persona": data.get("roleplay_persona") or "Supportive subject-matter coach",
            "exercise": data.get("exercise") or {
                "type": "reflection",
                "instructions": "In 2–3 lines, tell me what you hope to learn from this chapter.",
                "prompt": "What feels hardest right now?",
                "choices": None,
                "answer": None,
                "skills": ["metacognition"]
            }
        }

    # 4) Build and save state (store retrieval artifacts)
    state = TutorState(
        user_id=req.user_id,
        profile=req.profile.model_dump(),
        chapter_title=req.chapter_title,
        chapter_source=req.chapter_source,
        chapter_summary=data["chapter_summary"],
        learning_objectives=data["learning_objectives"],
        roleplay_persona=data["roleplay_persona"],
        exercise_queue=[data["exercise"]],
        chunks=chunks,
        index=index,
        scenario=scenario,
        last_prompt_at=time.time(),
    )
    SESSION[req.user_id] = state

    first_prompt = data["exercise"].get("instructions") or data["exercise"].get("prompt") or \
                   "Let's begin. Tell me what you understand so far."

    lesson_plan = {
        "chapter_summary": state.chapter_summary,
        "learning_objectives": state.learning_objectives,
        "roleplay_persona": state.roleplay_persona,
        "first_exercise": data["exercise"],
    }

    return IngestResponse(user_id=req.user_id, lesson_plan=lesson_plan, first_prompt=first_prompt)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 0) Ensure session exists
    state = SESSION.get(req.user_id)
    if not state:
        raise HTTPException(status_code=404, detail="No active session. Call /ingest first.")

    # 1) Retrieve top-k chapter snippets for this turn
    top = retrieve(req.message, state.chunks, state.index, k=3)
    sources_block = "\n".join([f"[{cid}] {txt}" for cid, txt in top])

    user_msg = CHAT_USER_SCOPED.format(
        background=state.profile.get("background"),
        goals=state.profile.get("goals"),
        level=state.profile.get("level") or "unspecified",
        chapter_title=state.chapter_title,
        learning_objectives=", ".join(state.learning_objectives),
        sources_block=sources_block,
        user_message=req.message,
    )

    scenario = state.scenario or {}
    sys_msg = SYSTEM_TUTOR_SCOPED.format(
        user_role=scenario.get("user_role", "Learner"),
        bot_role=scenario.get("bot_role", "Mentor"),
        emotion=scenario.get("emotion", "supportive"),
        difficulty=scenario.get("difficulty", 1),
        learning_style=scenario.get("learning_style", ""),
    )

    raw = _hf_chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    data = _safe_json_loads(raw)

    # 2) Enforce citations (soft retry)
    allowed = [cid for cid, _ in top]
    if not _valid_citations(data.get("citations"), allowed) and not (data.get("reply","").startswith(REFUSAL_PREFIX)):
        retry_msg = user_msg + "\n\nReminder: You MUST cite only from these snippet IDs and you MUST include 'citations'."
        raw2 = _hf_chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": retry_msg},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        data2 = _safe_json_loads(raw2)
        if _valid_citations(data2.get("citations"), allowed) or data2.get("reply","").startswith(REFUSAL_PREFIX):
            data = data2

    # 3) Optional: deadline/time-pressure check on last exercise
    if state.exercise_queue and isinstance(state.exercise_queue[-1], dict):
        deadline = state.exercise_queue[-1].get("deadline_sec")
        if deadline:
            elapsed = time.time() - (state.last_prompt_at or time.time())
            if elapsed > float(deadline):
                data.setdefault("assessment", {})
                data["assessment"]["correct"] = False
                data["assessment"]["explanation"] = f"Time ran out (>{int(deadline)}s). Let's review key steps."
                data["assessment"]["delta_mastery"] = -1

    # 4) Update progress if assessment present
    assessment = data.get("assessment") or {}
    if assessment and isinstance(assessment, dict):
        state.progress.attempts += 1
        if assessment.get("correct") is True:
            state.progress.correct_in_a_row += 1
            state.progress.mastery = max(0, min(100, state.progress.mastery + int(assessment.get("delta_mastery", 2))))
        elif assessment.get("correct") is False:
            state.progress.correct_in_a_row = 0
            state.progress.mastery = max(0, min(100, state.progress.mastery - 1))

    # 5) Queue new exercise if provided
    new_ex = data.get("new_exercise")
    if new_ex:
        state.exercise_queue.append(new_ex)
        state.last_prompt_at = time.time()

    # 6) Maintain short history (last 10 turns)
    state.history.append({"user": req.message, "assistant": data.get("reply", "")})
    state.history = state.history[-10:]
    state.updated_at = time.time()

    # 7) Return
    payload = {
        "user_id": req.user_id,
        "reply": data.get("reply", "(No reply)"),
        "state": _state_public_view(state),
    }
    if new_ex:
        payload["new_exercise"] = new_ex
    return payload  # FastAPI will coerce to ChatResponse

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

############################################################
# Notes for Frontend Integration (summary)
############################################################
# 1) Call POST /ingest with { user_id, chapter_title, chapter_text, chapter_source?, profile }
#    -> Render lesson_plan and show first_prompt to the user.
# 2) For each user message, call POST /chat with { user_id, message }.
#    -> Show `reply` and, if desired, render the top of `exercise_queue` for interaction.
# 3) Persist state server-side (Redis/DB). The SESSION dict is only for demo.
# 4) Consider adding SSE/websocket streaming if your HF endpoint supports it.
# 5) For safety/format drift, keep _safe_json_loads and consider guardrails (JSON schema validation).
