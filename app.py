import json
import os
import re
import socket

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_v5 import agent_answer, get_agent_profiles  # V5版Agent
from news_intelligence import NewsIntelligenceService
from scheduler import start_background_scheduler
from transformer_bootstrap import ensure_news_transformer_model
from voice_service import whisper_service

app = FastAPI()
news_service = NewsIntelligenceService()
SUPPORTED_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}
DEFAULT_MODEL = "deepseek-v4-flash"


@app.on_event("startup")
def startup_tasks():
    if ensure_news_transformer_model():
        news_service.refresh_if_needed()
    start_background_scheduler()

def _parse_origin_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _build_allowed_origins() -> list[str]:
    defaults = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    extra = _parse_origin_list(os.getenv("CORS_ALLOW_ORIGINS", ""))
    merged: list[str] = []
    for origin in defaults + extra:
        if origin not in merged:
            merged.append(origin)
    return merged


def _get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


# 允许本机、局域网和显式配置的前端来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_allowed_origins(),
    allow_origin_regex=os.getenv(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$",
    ),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# -------- Frontend --------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.join(_BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


@app.get("/server-info")
def server_info():
    port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
    return {
        "local_url": f"http://127.0.0.1:{port}",
        "lan_url": f"http://{_get_local_ip()}:{port}",
    }

# -------- API --------
class Query(BaseModel):
    query: str
    history: list = Field(default_factory=list)
    model: str | None = None
    agent_id: str | None = None

class NewsArticle(BaseModel):
    title: str = ""
    content: str = ""
    url: str = ""


def strip_for_tts(text: str) -> str:
    cleaned = re.sub(r"[#*`>\-\[\]]", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def resolve_model(model: str | None) -> str:
    selected = (model or DEFAULT_MODEL).strip().lower()
    return selected if selected in SUPPORTED_MODELS else DEFAULT_MODEL


@app.post("/chat")
def chat_endpoint(q: Query):
    model = resolve_model(q.model)
    answer = agent_answer(q.query, q.history, q.agent_id, model)
    return {"answer": answer, "agent_id": q.agent_id or "auto", "model": model}


@app.get("/agents")
def list_agents():
    return {"agents": get_agent_profiles()}

@app.post("/news/analyze")
def analyze_news(article: NewsArticle):
    content = article.content
    article_fetch_status = "not_requested"
    if article.url and not content:
        from fetch_news import fetch_article_text

        content = fetch_article_text(article.url)
        article_fetch_status = "ok" if content else "empty"

    if not article.title and not content:
        raise HTTPException(status_code=400, detail="请至少提供标题、正文或可抓取的 URL。")

    insight = news_service.analyze_article(article.title, content)
    return {
        "title": insight.title,
        "category": insight.category,
        "sentiment": insight.sentiment,
        "summary": insight.summary,
        "score_detail": insight.score_detail,
        "article_fetch_status": article_fetch_status,
    }


@app.post("/voice/chat")
async def voice_chat(
    transcript: str = Form(default=""),
    history: str = Form(default="[]"),
    agent_id: str = Form(default="auto"),
    model: str = Form(default=DEFAULT_MODEL),
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

    selected_model = resolve_model(model)
    answer = agent_answer(final_transcript, parsed_history, agent_id, selected_model)
    return {
        "transcript": final_transcript,
        "answer": answer,
        "agent_id": agent_id or "auto",
        "model": selected_model,
        "tts_text": strip_for_tts(answer),
        "asr_provider": asr_provider,
    }

# 健康检查
@app.get("/ping")
def ping():
    return {"message": "AI Agent 后端运行中"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
    reload_enabled = os.getenv("APP_RELOAD", "true").strip().lower() in {"1", "true", "yes", "on"}

    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)
