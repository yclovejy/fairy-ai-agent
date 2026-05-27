import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import requests

from deepseek_client import deepseek_client

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ========= 配置 =========
NEWS_API = "http://api.xcvts.cn/api/hotlist/qq_news?type=new"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(_BASE_DIR, "data", "news.json")
EMBED_PATH = os.path.join(_BASE_DIR, "data", "news_embeddings.npy")
GENERATED_TRAIN_PATH = os.path.join(_BASE_DIR, "data", "news_train_generated.jsonl")

SENTENCE_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = None

CATEGORY_KEYWORDS = {
    "体育": ["比赛", "球队", "联赛", "球员", "进球", "冠军", "篮球", "足球", "体育", "夺冠", "赛季", "奥运"],
    "科技": ["AI", "人工智能", "芯片", "大模型", "算法", "机器人", "科技", "数据", "算力", "互联网", "软件", "智能"],
    "财经": ["股市", "经济", "财经", "投资", "融资", "市场", "基金", "银行", "企业", "盈利", "债券", "金融"],
    "娱乐": ["电影", "明星", "票房", "综艺", "娱乐", "演唱会", "电视剧", "粉丝", "艺人", "上映", "导演", "音乐"],
}


def env_flag(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_embedding_model():
    global model
    if model is not None:
        return model
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(SENTENCE_MODEL_NAME, local_files_only=True)
        return model
    except Exception as e:
        print("⚠️ 未找到本地 SentenceTransformer 模型，跳过 embedding 生成:", e)
        return None

def load_old_news():
    if not os.path.exists(SAVE_PATH):
        return []
    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def fetch_news():
    print("🚀 正在抓取新闻...")

    try:
        res = requests.get(NEWS_API, timeout=10)
        res.raise_for_status()
        data = res.json()

        news_list = []

        for item in data.get("data", []):
            title = item.get("title", "") or ""
            content = item.get("desc", "") or "" # 腾讯API字段

            news_list.append({
                "title": title,
                "content": content
            })

        print(f"✅ 抓取成功: {len(news_list)}条")

        return news_list

    except Exception as e:
        print("❌ 抓取失败:", e)
        return []


def extract_json_payload(text):
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.append(cleaned[array_start: array_end + 1])

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidates.append(cleaned[object_start: object_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response does not contain valid JSON.")


def normalize_llm_news_items(payload):
    if isinstance(payload, dict):
        items = payload.get("news") or payload.get("items") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    normalized = []
    seen_titles = set()
    fetched_at = datetime.now(timezone.utc).isoformat()
    for item in items:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        content = str(item.get("content") or item.get("summary") or item.get("desc") or "").strip()
        if not title or title in seen_titles:
            continue

        normalized.append({
            "title": title,
            "content": content,
            "source": str(item.get("source", "DeepSeek fallback")).strip() or "DeepSeek fallback",
            "published_at": str(item.get("published_at", "")).strip(),
            "fetched_at": fetched_at,
            "provider": "deepseek_fallback",
        })
        seen_titles.add(title)

    return normalized


def fetch_news_with_llm():
    if not env_flag("NEWS_LLM_FALLBACK_ENABLED", True):
        print("⏭️ LLM 新闻兜底已关闭")
        return []

    if not deepseek_client.is_enabled():
        print("⚠️ 未配置 DEEPSEEK_API_KEY，无法启用 LLM 新闻兜底")
        return []

    current_date = datetime.now().strftime("%Y-%m-%d")
    max_items = max(1, env_int("NEWS_LLM_FALLBACK_MAX_ITEMS", 12))
    prompt = (
        f"今天是 {current_date}。主新闻爬虫不可用，请整理 {max_items} 条尽可能新的中文热点新闻。"
        "优先覆盖科技、财经、国际、中国社会、文旅娱乐和体育。"
        "只返回 JSON，不要 Markdown，不要解释。格式为："
        "{\"news\":[{\"title\":\"标题\",\"content\":\"80到160字摘要\",\"source\":\"来源或线索\","
        "\"published_at\":\"可为空\"}]}"
    )

    try:
        raw = deepseek_client.chat(
            [
                {
                    "role": "system",
                    "content": "你是严谨的新闻检索兜底服务，只输出可解析 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2200,
        )
        news_list = normalize_llm_news_items(extract_json_payload(raw))
        print(f"✅ LLM 兜底生成新闻: {len(news_list)}条")
        return news_list
    except Exception as e:
        print("❌ LLM 新闻兜底失败:", e)
        return []

def merge_news(old_news, new_news):
    print("合并新闻并去重...")

    merged = old_news + new_news

    seen = set()
    unique_news = []

    for news in merged:
        title = news.get("title", "").strip()

        if title and title not in seen:
            seen.add(title)
            unique_news.append(news)

    print(f"去重后剩余: {len(unique_news)}条")
    return unique_news

def save_news(news_list):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)

    print("💾 已保存 news.json")

def build_embeddings(news_list):
    embedding_model = get_embedding_model()
    if embedding_model is None:
        return

    print("🧠 正在生成embedding...")

    texts = []
    for news in news_list:
        text = (news.get("title") or "") + " " + (news.get("content") or "")
        texts.append(text)

    if not texts:
        print("⚠️ 没有新闻文本，跳过embedding生成")
        return

    embeddings = embedding_model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(EMBED_PATH, embeddings)

    print("💾 已保存 embeddings")


def infer_label_from_keywords(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    scores = {}
    for label, keywords in CATEGORY_KEYWORDS.items():
        scores[label] = sum(1 for keyword in keywords if keyword in cleaned)

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return None

    tied_labels = [label for label, score in scores.items() if score == best_score]
    if len(tied_labels) > 1:
        return None
    return best_label


def build_generated_training_rows(news_list):
    rows = []
    seen_texts = set()

    for news in news_list:
        title = (news.get("title") or "").strip()
        content = (news.get("content") or "").strip()
        text = f"{title} {content}".strip()
        if not text or text in seen_texts:
            continue

        label = infer_label_from_keywords(text)
        if not label:
            continue

        rows.append({"text": text, "label": label})
        seen_texts.add(text)

    print(f"🧪 生成弱监督训练样本: {len(rows)}条")
    return rows


def save_generated_training_rows(rows):
    os.makedirs(os.path.dirname(GENERATED_TRAIN_PATH), exist_ok=True)
    with open(GENERATED_TRAIN_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("💾 已保存 news_train_generated.jsonl")


def train_transformer_if_needed(train_enabled, rows):
    if not train_enabled:
        print("⏭️ 已关闭自动 Transformer 训练")
        return

    if not rows:
        print("⚠️ 没有可用的新增训练样本，跳过 Transformer 训练")
        return

    extra_epochs = os.getenv("NEWS_TRAIN_EPOCHS", "4").strip() or "4"
    batch_size = os.getenv("NEWS_TRAIN_BATCH_SIZE", "8").strip() or "8"

    cmd = [
        sys.executable,
        "train_transformer_news.py",
        "--data-path",
        os.path.join("data", "news_train.jsonl"),
        "--data-path",
        os.path.join("data", "news_train_generated.jsonl"),
        "--epochs",
        extra_epochs,
        "--batch-size",
        batch_size,
    ]

    print("🧠 开始训练 Transformer 分类器...")
    completed = subprocess.run(cmd, cwd=_BASE_DIR, check=False)
    if completed.returncode == 0:
        print("✅ Transformer 训练完成")
    else:
        print(f"❌ Transformer 训练失败，退出码: {completed.returncode}")


def run_news_pipeline(train_transformer=False):
    old_news = load_old_news()
    new_news = fetch_news()
    if not new_news:
        print("🔁 主新闻源不可用，尝试 LLM 新闻兜底...")
        new_news = fetch_news_with_llm()

    if not new_news:
        print("⚠️ 没有数据，不更新")
        return False

    merged_news = merge_news(old_news, new_news)
    if merged_news == old_news:
        print("ℹ️ 新闻列表无新增内容，跳过重建与训练")
        return False

    save_news(merged_news)
    build_embeddings(merged_news)

    generated_rows = build_generated_training_rows(merged_news)
    save_generated_training_rows(generated_rows)
    train_transformer_if_needed(train_transformer, generated_rows)
    return True

# ========= 主流程 =========
if __name__ == "__main__":
    train_enabled = os.getenv("NEWS_AUTO_TRAIN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    run_news_pipeline(train_transformer=train_enabled)
