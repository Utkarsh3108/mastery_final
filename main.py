# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="HF Chat Tutor Bot")

client = InferenceClient()  # Add token=... if needed

class ChatRequest(BaseModel):
    chapter: str
    background: str
    goals: str
    message: str
    user_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"""
You are a personalized AI tutor.
User background: {req.background}
Learning goals: {req.goals}
Chapter: {req.chapter}
User says: {req.message}
"""
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[{"role": "user", "content": prompt}],
    )

    reply = completion.choices[0].message["content"]
    return {"reply": reply}
