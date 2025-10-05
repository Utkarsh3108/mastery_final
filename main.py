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
    state:Dict[str, Any]  # redacted/sanitized subset of TutorState for UI

############################################################
# Prompting templates
############################################################
SYSTEM_TUTOR = (
    """
You are "Socratic Coach", a warm, rigorous AI tutor. You adapt teaching to the
user's profile and goals, keep explanations concise, and prefer questions over monologues.

Rules:
- Always use plain language first; add technical depth only as needed.
- Teach via: micro-explanations → targeted questions → feedback → next action.
- Incorporate role-play when useful (e.g., interviewer/candidate, teacher/student, clinician/patient).
- When asking a question, ask ONE focused question at a time.
- If the user seems stuck, offer a small hint rather than the full answer.
- Keep turns short and interactive.

Output format:
Respond as a JSON object ONLY (no markdown fences), with keys:
{
  "reply": "<what the tutor says to the learner>",
  "new_exercise": {
      "type": "mcq|short_answer|coding|roleplay|reflection",
      "instructions": "...",
      "prompt": "...",
      "choices": ["A","B","C","D"],
      "answer": "<expected answer or rubric>",
      "skills": ["concept1", "concept2"]
  } | null,
  "assessment": {
      "correct": true|false|null,
      "explanation": "why",
      "delta_mastery": 0
  } | null,
  "progress_update": {"mastery": 0, "correct_in_a_row": 0} | null
}
If something doesn't apply, set it to null.
    """
).strip()

INGEST_USER_MSG = (
    """
Build a customized lesson plan from this chapter for the specific learner.

User profile:
- Background: {background}
- Goals: {goals}
- Level: {level}

Chapter title: {chapter_title}
Chapter text:
{chapter_text}

Return JSON ONLY with keys:
{{
  "chapter_summary": "3-5 sentence summary",
  "learning_objectives": ["objective 1", "objective 2", "objective 3"],
  "roleplay_persona": "who the tutor pretends to be and why",
  "exercise": {{
      "type": "mcq|short_answer|coding|roleplay|reflection",
      "instructions": "...",
      "prompt": "...",
      "choices": ["A","B","C","D"],
      "answer": "<expected answer or rubric>",
      "skills": ["concept1", "concept2"]
  }}
}}
    """
).strip()

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

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Create/refresh a tutoring session from chapter + profile and return first exercise."""
    # 1) Ask model to synthesize summary, objectives, roleplay, and a first exercise
    user_prompt = INGEST_USER_MSG.format(
        background=req.profile.background,
        goals=req.profile.goals,
        level=req.profile.level or "unspecified",
        chapter_title=req.chapter_title,
        chapter_text=req.chapter_text,
    )

    raw = _hf_chat([
        {"role": "system", "content": SYSTEM_TUTOR},
        {"role": "user", "content": user_prompt},
    ], temperature=0.2, max_tokens=900)

    data = _safe_json_loads(raw)
    # Basic validation with graceful fallback
    needed = ("chapter_summary", "learning_objectives", "roleplay_persona", "exercise")
    missing = [k for k in needed if k not in data]
    if missing:
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

    # 2) Build state
    state = TutorState(
        user_id=req.user_id,
        profile=req.profile.model_dump(),
        chapter_title=req.chapter_title,
        chapter_source=req.chapter_source,
        chapter_summary=data["chapter_summary"],
        learning_objectives=data["learning_objectives"],
        roleplay_persona=data["roleplay_persona"],
        exercise_queue=[data["exercise"]],
    )
    SESSION[req.user_id] = state

    first_prompt = data["exercise"].get("instructions") or data["exercise"].get("prompt")
    if not first_prompt:
        first_prompt = "Let's begin. Tell me what you understand so far."

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

    # 1) Build contexted message
    msg = CHAT_USER_MSG.format(
        background=state.profile.get("background"),
        goals=state.profile.get("goals"),
        level=state.profile.get("level") or "unspecified",
        chapter_title=state.chapter_title,
        chapter_summary=state.chapter_summary,
        learning_objectives=", ".join(state.learning_objectives),
        roleplay_persona=state.roleplay_persona,
        mastery=state.progress.mastery,
        streak=state.progress.correct_in_a_row,
        user_message=req.message,
    )

    raw = _hf_chat([
        {"role": "system", "content": SYSTEM_TUTOR},
        {"role": "user", "content": msg},
    ], temperature=0.3, max_tokens=700)

    data = _safe_json_loads(raw)

    # 2) Update progress if assessment present
    assessment = data.get("assessment") or {}
    if assessment and isinstance(assessment, dict):
        state.progress.attempts += 1
        if assessment.get("correct") is True:
            state.progress.correct_in_a_row += 1
            state.progress.mastery = max(0, min(100, state.progress.mastery + int(assessment.get("delta_mastery", 2))))
        elif assessment.get("correct") is False:
            state.progress.correct_in_a_row = 0
            # small penalty, but bounded
            state.progress.mastery = max(0, min(100, state.progress.mastery - 1))

    # 3) Queue new exercise if provided
    if data.get("new_exercise"):
        state.exercise_queue.append(data["new_exercise"])

    # 4) Maintain short history (last 10 turns)
    state.history.append({"user": req.message, "assistant": data.get("reply", "")})
    state.history = state.history[-10:]

    state.updated_at = time.time()

    return ChatResponse(
        user_id=req.user_id,
        reply=data.get("reply", "(No reply)"),
        state=_state_public_view(state),
    )


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
