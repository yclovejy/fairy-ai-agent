from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
import re

from agent_v5 import agent_answer  # V5版Agent
from news_intelligence import NewsIntelligenceService
from voice_service import whisper_service

app = FastAPI()
news_service = NewsIntelligenceService()

# 允许前端访问（端口3000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",  # 如果需要同源测试
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,  # 如果需要携带cookie
)

# -------- Frontend --------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.join(_BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

# -------- API --------
class Query(BaseModel):
    query: str
    history: list = Field(default_factory=list)
    model: str | None = None

class NewsArticle(BaseModel):
    title: str = ""
    content: str = ""


def strip_for_tts(text: str) -> str:
    cleaned = re.sub(r"[#*`>\-\[\]]", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@app.post("/chat")
def chat_endpoint(q: Query):
    answer = agent_answer(q.query, q.history)
    return {"answer": answer}

@app.post("/news/analyze")
def analyze_news(article: NewsArticle):
    if not article.title and not article.content:
        raise HTTPException(status_code=400, detail="请至少提供标题或正文。")

    insight = news_service.analyze_article(article.title, article.content)
    return {
        "title": insight.title,
        "category": insight.category,
        "sentiment": insight.sentiment,
        "summary": insight.summary,
        "score_detail": insight.score_detail,
    }


@app.post("/voice/chat")
async def voice_chat(
    transcript: str = Form(default=""),
    history: str = Form(default="[]"),
    audio: UploadFile | None = File(default=None),
):
    parsed_history = []
    if history:
        try:
            parsed_history = json.loads(history)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="history 不是合法 JSON。") from exc

    asr_provider = "browser"
    final_transcript = transcript.strip()

    if not final_transcript and audio is not None:
        audio_bytes = await audio.read()
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        try:
            result = whisper_service.transcribe_bytes(audio_bytes, suffix=suffix)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        final_transcript = result["text"]
        asr_provider = result["provider"]

    if not final_transcript:
        raise HTTPException(status_code=400, detail="没有收到可识别的语音或文本。")

    answer = agent_answer(final_transcript, parsed_history)
    return {
        "transcript": final_transcript,
        "answer": answer,
        "tts_text": strip_for_tts(answer),
        "asr_provider": asr_provider,
    }

# 健康检查
@app.get("/ping")
def ping():
    return {"message": "AI Agent 后端运行中"}
