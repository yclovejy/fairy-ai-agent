from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from fairy_core.paths import DATA_DIR

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


VECTOR_STORE_PATH = str(DATA_DIR / "news_vector_store.npz")
VECTOR_META_PATH = str(DATA_DIR / "news_vector_store_meta.json")
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def safe_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def normalize_vector_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors.shape) == 1:
        vectors = np.expand_dims(vectors, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def news_item_text(news: dict[str, Any]) -> str:
    fields = [
        "title",
        "content",
        "summary",
        "desc",
        "source",
        "published_at",
        "fetched_at",
        "article_text",
        "url",
    ]
    return " ".join(safe_text(news.get(field, "")) for field in fields if news.get(field)).strip()


def weighted_news_text(news: dict[str, Any]) -> str:
    title = safe_text(news.get("title") or "")
    content = safe_text(news.get("content") or news.get("summary") or news.get("desc") or "")
    article_text = safe_text(news.get("article_text") or "")
    source = safe_text(news.get("source") or "")
    return f"{title} {title} {title} {source} {content} {article_text[:1800]}".strip()


def news_id(news: dict[str, Any]) -> str:
    raw = "|".join(
        [
            safe_text(news.get("title") or ""),
            safe_text(news.get("url") or ""),
            safe_text(news.get("published_at") or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def build_signature(news_items: list[dict[str, Any]], provider: str, model_name: str) -> dict[str, Any]:
    ids = [news_id(item) for item in news_items]
    digest = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()
    return {
        "version": 2,
        "provider": provider,
        "model_name": model_name,
        "count": len(news_items),
        "digest": digest,
        "ids": ids,
    }


def save_vector_store(vectors: np.ndarray, news_items: list[dict[str, Any]], provider: str, model_name: str) -> None:
    if not env_flag("NEWS_VECTOR_STORE_ENABLED", True):
        return

    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    signature = build_signature(news_items, provider, model_name)
    np.savez_compressed(
        VECTOR_STORE_PATH,
        embeddings=normalize_vector_matrix(vectors),
        ids=np.asarray(signature["ids"]),
    )
    with open(VECTOR_META_PATH, "w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存持久化向量库: {VECTOR_STORE_PATH}")


def load_vector_store(news_items: list[dict[str, Any]], provider: str, model_name: str) -> np.ndarray | None:
    if not env_flag("NEWS_VECTOR_STORE_ENABLED", True):
        return None
    if not (os.path.exists(VECTOR_STORE_PATH) and os.path.exists(VECTOR_META_PATH)):
        return None

    try:
        with open(VECTOR_META_PATH, "r", encoding="utf-8") as f:
            saved_signature = json.load(f)
        current_signature = build_signature(news_items, provider, model_name)
        comparable_keys = ["version", "provider", "model_name", "count", "digest"]
        if any(saved_signature.get(key) != current_signature.get(key) for key in comparable_keys):
            return None

        data = np.load(VECTOR_STORE_PATH, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        if embeddings.shape[0] != len(news_items):
            return None
        print("加载持久化新闻向量库")
        return normalize_vector_matrix(embeddings)
    except Exception as exc:
        print("持久化向量库加载失败，回退重建:", exc)
        return None


@dataclass
class RankedNews:
    score: float
    item: dict[str, Any]


class CrossEncoderReranker:
    def __init__(self) -> None:
        self.model = None
        self.model_name = os.getenv("NEWS_RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL_NAME).strip()
        self.enabled = env_flag("NEWS_RERANKER_ENABLED", True)
        self._attempted = False

    def _load(self) -> None:
        if self._attempted or not self.enabled:
            return
        self._attempted = True
        if CrossEncoder is None:
            print("cross-encoder reranker 不可用，跳过重排")
            return
        try:
            try:
                self.model = CrossEncoder(self.model_name, local_files_only=True)
            except TypeError:
                self.model = CrossEncoder(
                    self.model_name,
                    automodel_args={"local_files_only": True},
                    tokenizer_args={"local_files_only": True},
                )
            print(f"已加载本地 cross-encoder reranker: {self.model_name}")
        except Exception as exc:
            self.model = None
            print("未找到本地 cross-encoder reranker，保留原始检索排序:", exc)

    def rerank(self, query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        self._load()
        if not self.model or len(items) <= 1:
            return items[:top_k]

        max_candidates = max(top_k, env_int("NEWS_RERANKER_MAX_CANDIDATES", 30))
        candidates = items[:max_candidates]
        pairs = [(query, news_item_text(item)[:2200]) for item in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as exc:
            print("cross-encoder reranker 执行失败，保留原始检索排序:", exc)
            return items[:top_k]

        ranked = [
            RankedNews(
                score=float(score) + float(item.get("source_credibility_score", 0.5)) * 0.05,
                item=item,
            )
            for score, item in zip(scores, candidates)
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return [result.item for result in ranked[:top_k]]


def tfidf_rank(query: str, news_items: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], np.ndarray, TfidfVectorizer]:
    texts = [weighted_news_text(item) for item in news_items]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    query_embedding = vectorizer.transform([query]).toarray()[0].astype(np.float32)
    if np.linalg.norm(query_embedding) == 0:
        return [], embeddings, vectorizer
    embeddings = normalize_vector_matrix(embeddings)
    scores = np.dot(embeddings, query_embedding / np.linalg.norm(query_embedding))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [news_items[i] for i in top_indices if scores[i] > 0], embeddings, vectorizer


def source_credibility_badge(news: dict[str, Any]) -> str:
    score = float(news.get("source_credibility_score", 0.5))
    if score >= 0.82:
        return "高"
    if score >= 0.62:
        return "中"
    return "待核验"


def source_credibility_line(news: dict[str, Any]) -> str:
    source = safe_text(news.get("source") or news.get("provider") or "未知来源")
    provider = safe_text(news.get("provider") or "")
    score = float(news.get("source_credibility_score", 0.5))
    badge = source_credibility_badge(news)
    return f"{source} / {provider} / 可信度{badge}({score:.2f})"
