from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


LABELS = ["体育", "科技", "财经", "娱乐"]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "news_transformer")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


CATEGORY_KEYWORDS = {
    "体育": ["比赛", "球队", "联赛", "球员", "进球", "冠军", "篮球", "足球", "体育", "夺冠", "赛季", "奥运"],
    "科技": ["AI", "人工智能", "芯片", "大模型", "算法", "机器人", "科技", "数据", "算力", "互联网", "软件", "智能"],
    "财经": ["股市", "经济", "财经", "投资", "融资", "市场", "基金", "银行", "企业", "盈利", "债券", "金融"],
    "娱乐": ["电影", "明星", "票房", "综艺", "娱乐", "演唱会", "电视剧", "粉丝", "艺人", "上映", "导演", "音乐"],
}

POSITIVE_WORDS = {
    "突破", "增长", "提升", "回暖", "合作", "创新", "利好", "成功", "稳步", "领先",
    "改善", "看好", "提振", "复苏", "高效", "点赞", "向好", "增产", "夺冠", "刷新",
}
NEGATIVE_WORDS = {
    "下滑", "亏损", "风险", "压力", "争议", "暴跌", "裁员", "危机", "处罚", "拖累",
    "波动", "事故", "失败", "质疑", "担忧", "制裁", "下跌", "违规", "冲突", "悲观",
}
RISK_WORDS = {
    "风险", "争议", "冲突", "处罚", "危机", "制裁", "下滑", "亏损", "事故", "担忧",
    "波动", "失业", "裁员", "暴跌", "违规", "谣言",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"[。！？!?；;\n]", cleaned)
    return [part.strip(" ，,") for part in parts if part.strip(" ，,")]


def pick_keywords(texts: list[str], top_k: int = 6) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for keyword_group in CATEGORY_KEYWORDS.values():
            for keyword in keyword_group:
                if keyword in text:
                    tokens.append(keyword)

        tokens.extend(re.findall(r"[A-Za-z]{2,}", text))

    counter = Counter(tokens)
    return [word for word, _ in counter.most_common(top_k)]


def extractive_summary(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "暂无可用摘要。"

    if len(sentences) <= max_sentences:
        return "。".join(sentences) + "。"

    joined = " ".join(sentences)
    keywords = pick_keywords([joined], top_k=8)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        score = len(sentence) * 0.02
        for keyword in keywords:
            if keyword and keyword in sentence:
                score += 1.5
        scored.append((score, sentence))

    best_sentences = {sentence for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]}
    ordered = [sentence for sentence in sentences if sentence in best_sentences][:max_sentences]
    return "。".join(ordered) + "。"


def keyword_category_scores(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    lowered = normalize_text(text)
    for label, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in lowered:
                score += 1.0
        scores[label] = score

    if not any(scores.values()):
        return {label: 1 / len(LABELS) for label in LABELS}

    total = sum(scores.values()) or 1.0
    return {label: value / total for label, value in scores.items()}


def sentiment_scores(text: str) -> tuple[int, int, int]:
    cleaned = normalize_text(text)
    positive = sum(cleaned.count(word) for word in POSITIVE_WORDS)
    negative = sum(cleaned.count(word) for word in NEGATIVE_WORDS)
    risk = sum(cleaned.count(word) for word in RISK_WORDS)
    return positive, negative, risk


def sentiment_label(text: str) -> tuple[str, dict[str, int]]:
    positive, negative, risk = sentiment_scores(text)
    if positive == 0 and negative == 0:
        label = "中立"
    elif positive - negative >= 2:
        label = "正面"
    elif negative - positive >= 1:
        label = "负面"
    else:
        label = "中立"
    return label, {"positive": positive, "negative": negative, "risk": risk}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerNewsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.position(x)
        encoded = self.encoder(x, src_key_padding_mask=~attention_mask.bool())

        mask = attention_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(pooled))
        return logits


class CharTokenizer:
    def __init__(self, vocab: dict[str, int], max_len: int = 128) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    @classmethod
    def from_file(cls, path: str, max_len: int = 128) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.vocab.get(char, self.unk_id) for char in normalize_text(text)[: self.max_len]]
        if not tokens:
            tokens = [self.unk_id]
        attention_mask = [1] * len(tokens)
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        return torch.tensor([tokens], dtype=torch.long), torch.tensor([attention_mask], dtype=torch.long)


@dataclass
class NewsInsight:
    title: str
    category: str
    sentiment: str
    summary: str
    score_detail: dict[str, Any]


class NewsIntelligenceService:
    def __init__(self) -> None:
        self.model: TransformerNewsClassifier | None = None
        self.tokenizer: CharTokenizer | None = None
        self.device = torch.device("cpu")
        self.config: dict[str, Any] = {}
        self._model_signature: tuple[Any, ...] | None = None
        self._load_classifier()

    def _build_model_signature(self) -> tuple[Any, ...]:
        signature = []
        for path in [MODEL_PATH, VOCAB_PATH, CONFIG_PATH]:
            try:
                signature.append((path, os.path.getmtime(path), os.path.getsize(path)))
            except OSError:
                signature.append((path, None, None))
        return tuple(signature)

    def _load_classifier(self) -> None:
        self.model = None
        self.tokenizer = None
        self.config = {}

        if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(CONFIG_PATH)):
            self._model_signature = self._build_model_signature()
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.tokenizer = CharTokenizer.from_file(VOCAB_PATH, max_len=self.config.get("max_len", 128))
        self.model = TransformerNewsClassifier(
            vocab_size=self.config["vocab_size"],
            num_classes=len(self.config["labels"]),
            embed_dim=self.config.get("embed_dim", 128),
            num_heads=self.config.get("num_heads", 4),
            num_layers=self.config.get("num_layers", 2),
            hidden_dim=self.config.get("hidden_dim", 256),
            dropout=self.config.get("dropout", 0.1),
            max_len=self.config.get("max_len", 128),
        ).to(self.device)
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._model_signature = self._build_model_signature()

    def refresh_if_needed(self) -> None:
        signature = self._build_model_signature()
        if signature != self._model_signature:
            self._load_classifier()

    def has_trained_classifier(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict_category(self, text: str) -> tuple[str, dict[str, float], str]:
        self.refresh_if_needed()
        if self.model is None or self.tokenizer is None:
            scores = keyword_category_scores(text)
            label = max(scores, key=scores.get)
            return label, scores, "keyword"

        input_ids, attention_mask = self.tokenizer.encode(text)
        with torch.no_grad():
            logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        labels = self.config.get("labels", LABELS)
        scores = {label: float(prob) for label, prob in zip(labels, probs)}
        label = max(scores, key=scores.get)
        return label, scores, "transformer"

    def analyze_article(self, title: str, content: str) -> NewsInsight:
        full_text = normalize_text(f"{title} {content}")
        category, category_scores, provider = self.predict_category(full_text)
        sentiment, sentiment_detail = sentiment_label(full_text)
        summary = extractive_summary(content or title)
        return NewsInsight(
            title=title,
            category=category,
            sentiment=sentiment,
            summary=summary,
            score_detail={
                "category_scores": category_scores,
                "sentiment_detail": sentiment_detail,
                "category_provider": provider,
            },
        )

    def analyze_news_list(self, news_items: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
        if not news_items:
            return {
                "insights": [],
                "sentiment_distribution": {},
                "category_distribution": {},
                "keywords": [],
                "public_opinion": "没有检索到相关新闻，暂时无法形成舆情判断。",
            }

        selected = news_items[:top_k]
        insights = [
            self.analyze_article(item.get("title", ""), item.get("content", ""))
            for item in selected
        ]
        for insight, item in zip(insights, selected):
            insight.score_detail["source"] = item.get("source") or item.get("provider") or "未知来源"
            insight.score_detail["provider"] = item.get("provider") or ""
            insight.score_detail["source_domain"] = item.get("source_domain") or ""
            insight.score_detail["source_credibility_score"] = item.get("source_credibility_score", 0.5)

        sentiment_counter = Counter(item.sentiment for item in insights)
        category_counter = Counter(item.category for item in insights)
        keywords = pick_keywords(
            [f"{item.get('title', '')} {item.get('content', '')}" for item in selected],
            top_k=6,
        )

        dominant_category = category_counter.most_common(1)[0][0]
        dominant_sentiment = sentiment_counter.most_common(1)[0][0]
        risk_total = sum(item.score_detail["sentiment_detail"]["risk"] for item in insights)

        public_opinion = (
            f"当前检索结果主要集中在{dominant_category}主题，整体舆情偏{dominant_sentiment}。"
            f"关键词包括：{'、'.join(keywords) if keywords else '暂无明显高频词'}。"
        )
        if risk_total >= 3:
            public_opinion += " 同时出现了较多风险或争议词，适合进一步做社会舆情跟踪。"

        return {
            "insights": [item.__dict__ for item in insights],
            "sentiment_distribution": dict(sentiment_counter),
            "category_distribution": dict(category_counter),
            "keywords": keywords,
            "public_opinion": public_opinion,
        }


def format_analysis_block(analysis: dict[str, Any]) -> str:
    if not analysis.get("insights"):
        return "未检索到相关新闻。"

    lines = ["新闻智能分析："]
    for index, insight in enumerate(analysis["insights"], start=1):
        lines.append(
            f"{index}. {insight['title']}\n"
            f"   类别：{insight['category']} | 情感：{insight['sentiment']}\n"
            f"   来源：{insight.get('score_detail', {}).get('source', '未知来源')} | "
            f"可信度：{float(insight.get('score_detail', {}).get('source_credibility_score', 0.5)):.2f}\n"
            f"   摘要：{insight['summary']}"
        )

    sentiment_distribution = analysis.get("sentiment_distribution", {})
    category_distribution = analysis.get("category_distribution", {})
    keywords = analysis.get("keywords", [])

    lines.append(
        "舆情统计："
        f"\n- 情感分布：{sentiment_distribution or '暂无'}"
        f"\n- 类别分布：{category_distribution or '暂无'}"
        f"\n- 高频关键词：{'、'.join(keywords) if keywords else '暂无'}"
    )
    lines.append(f"舆情判断：{analysis.get('public_opinion', '暂无判断。')}")
    return "\n".join(lines)
