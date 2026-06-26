import numpy as np
import json
from dotenv import load_dotenv
import os
import re
import threading
import random
import requests
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from fairy_core.deepseek_client import deepseek_client
from fairy_core.news_intelligence import NewsIntelligenceService, format_analysis_block
from fairy_core.news_retrieval import (
    CrossEncoderReranker,
    load_vector_store,
    save_vector_store,
    source_credibility_line,
    weighted_news_text,
)
from fairy_core.travel_agent import travel_agent
from fairy_core.environment_service import environment_service
from fairy_core.paths import DATA_DIR
from fairy_core.vision_service import vision_service

load_dotenv()

# 定义常量
EMBED_PATH = str(DATA_DIR / "news_embeddings.npy")
NEWS_PATH = str(DATA_DIR / "news.json")
SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ========= 初始化 =========
embedding_model = None
embedding_provider = "tfidf"
news_vectorizer = None
news_intelligence = NewsIntelligenceService()
news_reranker = CrossEncoderReranker()
_news_asset_lock = threading.Lock()
_news_assets_signature = None
NEWS_HINT_WORDS = [
    "新闻", "热点", "舆情", "情感分析", "最近发生", "最新", "今日", "今天",
    "股市", "财经", "科技", "娱乐", "体育", "比赛", "夺冠", "票房", "AI",
]
LATEST_NEWS_PATTERNS = [
    r"(最新|今日|今天|现在|目前|实时|刚刚).*(新闻|热点|消息|动态|资讯)",
    r"(新闻|热点|消息|动态|资讯).*(最新|今日|今天|现在|目前|实时|刚刚)",
    r"(有什么|有哪些).*(新闻|热点|消息|动态|资讯)",
]
RECENT_NEWS_PATTERNS = [
    r"(最近|近几天|这几天|过去\d+天|近\d+天|一周内|本周).*(新闻|热点|消息|动态|资讯|发生|情况|走势)",
    r"(新闻|热点|消息|动态|资讯).*(最近|近几天|这几天|过去\d+天|近\d+天|一周内|本周)",
]
# ========= 数据 =========
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
QWEATHER_HOST = os.getenv("QWEATHER_HOST")

IDENTITY_ANSWERS = (
    "我是 Fairy，由我的主人 yongcheng 创造。我的原型来自米哈游《绝区零》中的人工智能 Fairy。至于别的嘛，先保留一点神秘感。",
    "我是 Fairy，yongcheng 是创造我的主人。灵感原型来自米哈游推出的《绝区零》里的人工智能 Fairy，其他内部信息就不对外展开啦。",
    "你可以叫我 Fairy。我的创造者与主人是 yongcheng，原型取自米哈游《绝区零》中的人工智能 Fairy。除此之外，我只聊我能为你做什么。",
    "Fairy 在此。是主人 yongcheng 创造了我，而我的原型来自米哈游游戏《绝区零》中的人工智能 Fairy。更多底层细节不属于我的自我介绍。",
)

IDENTITY_PATTERNS = (
    r"^(你|fairy|它)(到底|究竟)?是谁(啊|呀|呢)?$",
    r"谁.{0,6}(创造|开发|设计|制作|做出).{0,3}(你|fairy|它)",
    r"(你|fairy|它).{0,6}(谁创造|谁开发|谁设计|谁制作|主人是谁)",
    r"(你的|fairy的).{0,3}(主人|创造者|开发者|设计者)",
    r"(你|fairy).{0,4}(原型|来历|身世)",
)


def identity_answer(query):
    compact = re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]{}<>《》]", "", safe_text(query)).lower()
    if not any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in IDENTITY_PATTERNS):
        return None
    return random.choice(IDENTITY_ANSWERS)


def init_embedding_model():
    global embedding_model, embedding_provider
    if SentenceTransformer is None:
        print("sentence-transformers 不可用，切换到 TF-IDF 检索")
        return

    try:
        embedding_model = SentenceTransformer(SENTENCE_MODEL_NAME, local_files_only=True)
        embedding_provider = "sentence_transformer"
        print("已加载本地 SentenceTransformer 检索模型")
    except Exception as e:
        print("未找到本地 SentenceTransformer 模型，切换到 TF-IDF 检索:", e)
        embedding_model = None
        embedding_provider = "tfidf"


init_embedding_model()

def safe_text(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')


def normalize_search_text(text):
    text = safe_text(text or "").lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def news_item_text(news):
    return " ".join(
        safe_text(news.get(field, ""))
        for field in ["title", "content", "summary", "desc", "source", "published_at", "fetched_at"]
        if news.get(field)
    )


def get_news_store_updated_at():
    try:
        return datetime.fromtimestamp(os.path.getmtime(NEWS_PATH)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "未知"


def looks_like_latest_news_query(query):
    return any(re.search(pattern, query) for pattern in LATEST_NEWS_PATTERNS)


def looks_like_recent_news_query(query):
    return any(re.search(pattern, query) for pattern in RECENT_NEWS_PATTERNS)


def is_temporal_news_query(query):
    return looks_like_latest_news_query(query) or looks_like_recent_news_query(query)


def requested_recent_days(query):
    text = safe_text(query)
    match = re.search(r"(?:过去|近)(\d+)天", text)
    if match:
        return max(1, min(int(match.group(1)), 30))
    if any(word in text for word in ["一周", "本周", "近一周"]):
        return 7
    if any(word in text for word in ["今天", "今日", "刚刚", "实时"]):
        return 1
    return 3


def parse_news_time(value):
    if not value:
        return 0.0
    raw = safe_text(value).strip()
    try:
        if raw.isdigit():
            timestamp = int(raw)
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000
            return float(timestamp)
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def latest_news_items(limit=20):
    reliable_items = [item for item in news_list if item.get("provider") != "deepseek_fallback"]
    candidates = reliable_items or news_list

    def sort_key(item):
        return (
            parse_news_time(item.get("published_at")),
            parse_news_time(item.get("fetched_at")),
        )

    return sorted(candidates, key=sort_key, reverse=True)[:limit]


def news_item_timestamp(item):
    return max(
        parse_news_time(item.get("published_at")),
        parse_news_time(item.get("fetched_at")),
    )


def news_item_recency_timestamp(item):
    published_timestamp = parse_news_time(item.get("published_at"))
    if published_timestamp > 0:
        return published_timestamp
    return parse_news_time(item.get("fetched_at"))


def recent_news_items(days=3, limit=20, items=None):
    candidates = items if items is not None else news_list
    cutoff = datetime.now(timezone.utc).timestamp() - days * 24 * 60 * 60
    recent_items = [
        item for item in candidates
        if news_item_recency_timestamp(item) >= cutoff and item.get("provider") != "deepseek_fallback"
    ]
    if not recent_items:
        recent_items = [
            item for item in candidates
            if news_item_recency_timestamp(item) >= cutoff
        ]
    return sorted(recent_items, key=news_item_recency_timestamp, reverse=True)[:limit]


def load_news():
    if not os.path.exists(NEWS_PATH):
        return []
    with open(NEWS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_embeddings():
    if not os.path.exists(EMBED_PATH):
        return None
    return np.load(EMBED_PATH)


def get_news_assets_signature():
    signature = []
    for path in [NEWS_PATH, EMBED_PATH]:
        try:
            signature.append((path, os.path.getmtime(path), os.path.getsize(path)))
        except OSError:
            signature.append((path, None, None))
    return tuple(signature)

news_list = load_news()
news_embeddings = load_embeddings()

# ========= 新闻向量 =========

def prepare_news_embeddings():
    global news_texts, news_embeddings, news_vectorizer, embedding_provider

    news_texts = []
    for news in news_list:
        news_texts.append(weighted_news_text(news))

    if not news_texts:
        news_embeddings = None
        news_vectorizer = None
        return

    if embedding_model is not None:
        news_embeddings = load_vector_store(news_list, "sentence_transformer", SENTENCE_MODEL_NAME)

        if news_embeddings is None and os.path.exists(EMBED_PATH):
            print("加载本地 embedding")
            news_embeddings = np.load(EMBED_PATH)

        if news_embeddings is None:
            print("没有可用的预计算 embedding，现场计算")
            news_embeddings = embedding_model.encode(news_texts)
            news_embeddings = news_embeddings / np.linalg.norm(news_embeddings, axis=1, keepdims=True)

        if len(news_embeddings.shape) == 1:
            news_embeddings = np.expand_dims(news_embeddings, axis=0)
        if news_embeddings.shape[0] != len(news_list):
            print("embedding数量与新闻数量不一致，重新生成embedding")
            news_embeddings = embedding_model.encode(news_texts)
            news_embeddings = news_embeddings / np.clip(np.linalg.norm(news_embeddings, axis=1, keepdims=True), 1e-12, None)
            save_vector_store(news_embeddings, news_list, "sentence_transformer", SENTENCE_MODEL_NAME)
        embedding_provider = "sentence_transformer"
        news_vectorizer = None
        return

    print("使用 TF-IDF 构建本地检索索引")
    news_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    news_embeddings = news_vectorizer.fit_transform(news_texts).toarray().astype(np.float32)
    embedding_provider = "tfidf"

prepare_news_embeddings()
_news_assets_signature = get_news_assets_signature()

def reload_news():
    global news_list, news_embeddings, _news_assets_signature
    news_list = load_news()
    prepare_news_embeddings()
    _news_assets_signature = get_news_assets_signature()
    print("新闻数据已更新并重建embedding")


def ensure_latest_news_assets():
    global _news_assets_signature

    current_signature = get_news_assets_signature()
    if current_signature == _news_assets_signature:
        return

    with _news_asset_lock:
        current_signature = get_news_assets_signature()
        if current_signature == _news_assets_signature:
            return
        reload_news()


def ensure_news_available():
    if news_list:
        ensure_latest_news_assets()
        return

    print("新闻数据为空，尝试立即抓取一次...")
    try:
        from fairy_core.fetch_news import run_news_pipeline
        updated = run_news_pipeline(train_transformer=False)
    except Exception as exc:
        print("新闻即时抓取失败:", exc)
        updated = False

    if updated or os.path.exists(NEWS_PATH):
        reload_news()


def refresh_news_before_answer(query):
    if not is_temporal_news_query(query):
        return False

    train_enabled = os.getenv("NEWS_ON_DEMAND_TRAIN_ENABLED", os.getenv("NEWS_AUTO_TRAIN_ENABLED", "false"))
    train_transformer = train_enabled.strip().lower() in {"1", "true", "yes", "on"}
    print(f"[新闻刷新] 检测到时效性新闻问题，先联网刷新: {query}")

    with _news_asset_lock:
        try:
            from fairy_core.fetch_news import refresh_news_for_query
            updated = refresh_news_for_query(query, train_transformer=train_transformer)
        except Exception as exc:
            print("[新闻刷新] 即时刷新失败:", exc)
            updated = False
        if updated or os.path.exists(NEWS_PATH):
            reload_news()
        return updated

# ========= 工具 =========
def keyword_match(query, top_k=10):
    scored_results = []
    query_compact = normalize_search_text(query)
    query_terms = [
        normalize_search_text(term)
        for term in re.split(r"[\s，。！？、,.!?:：；;（）()《》【】\\[\\]\"'“”‘’]+", safe_text(query))
        if normalize_search_text(term)
    ]

    for news in news_list:
        title = news.get("title") or ""
        text = news_item_text(news)
        title_compact = normalize_search_text(title)
        text_compact = normalize_search_text(text)

        score = 0.0
        if query_compact and title_compact:
            if query_compact == title_compact:
                score += 100.0
            elif query_compact in title_compact or title_compact in query_compact:
                score += 80.0

        if query_compact and text_compact and query_compact in text_compact:
            score += 60.0

        for term in query_terms:
            if len(term) < 2:
                continue
            if term in title_compact:
                score += 8.0
            elif term in text_compact:
                score += 3.0

        if score > 0:
            scored_results.append((score, news))

    scored_results.sort(key=lambda item: item[0], reverse=True)
    return [news for _, news in scored_results[:top_k]]


def has_local_news_match(query):
    query_compact = normalize_search_text(query)
    if len(query_compact) < 6:
        return False

    for news in news_list[:1000]:
        title_compact = normalize_search_text(news.get("title") or "")
        if not title_compact:
            continue
        if query_compact == title_compact:
            return True
        if len(query_compact) >= 10 and (query_compact in title_compact or title_compact in query_compact):
            return True

    return False

def retrieve_news(query, top_k=20):
    ensure_latest_news_assets()

    if news_embeddings is None:
        return news_reranker.rerank(query, keyword_match(query, top_k=top_k), top_k=top_k)

    keyword_results = keyword_match(query, top_k=top_k)

    if len(keyword_results) >= 5 or has_local_news_match(query):
        print("[检索] 使用关键词命中")
        return news_reranker.rerank(query, keyword_results, top_k=top_k)

    if embedding_provider == "sentence_transformer" and embedding_model is not None:
        query_embedding = embedding_model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    else:
        if news_vectorizer is None:
            return keyword_results[:top_k]
        query_embedding = news_vectorizer.transform([query]).toarray()[0].astype(np.float32)

    if np.linalg.norm(query_embedding) == 0:
        return keyword_results[:top_k]

    scores = np.dot(news_embeddings, query_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    embed_results = [news_list[i] for i in top_indices if i < len(news_list) and scores[i] > 0]

    combined = keyword_results + embed_results

    seen = set()
    final_results = []

    for news in combined:
        title = news.get("title", "")
        if title not in seen:
            seen.add(title)
            final_results.append(news)

    print(f"[检索] keyword:{len(keyword_results)} | embedding:{len(embed_results)}")
    return news_reranker.rerank(query, final_results, top_k=top_k)


def build_news_prompt(query, result, analysis_block):
    picked = result[:5]
    return f"""
你是一个新闻分析专家，请根据检索到的新闻回答用户问题。

用户问题：
{query}

检索新闻：
{json.dumps(picked, ensure_ascii=False, indent=2)}

本地新闻库更新时间：
{get_news_store_updated_at()}

智能分析结果：
{analysis_block}

回答要求：
1. 先给出简洁结论，再给出原因。
2. 可以概括行业趋势、舆情倾向、潜在风险。
3. 如果涉及情感分析，请明确说明偏正面、负面还是中立。
4. 如果用户问“最新/今天/目前/最近”，优先使用系统刚刷新并入库的新闻，不要使用过期旧库冒充最新。
5. 保持条理清晰，适合课程大作业展示。
"""


def build_news_fallback_prompt(query):
    return f"""
用户正在询问新闻或热点问题，但本地新闻数据库暂时没有可用检索结果。

用户问题：
{query}

请用你的通用知识和推理能力兜底回答。

回答要求：
1. 不要说“没有检索到相关新闻”。
2. 如果问题明显要求最新实时新闻，请先说明“本地新闻库暂时没有检索结果，我先基于大模型知识给出参考分析”，避免假装已经联网看到最新消息。
3. 给出可操作、条理清晰的分析；可以补充用户接下来可以关注哪些关键词或方向。
4. 语气自然，像 Fairy 在认真帮用户兜底。
"""


def format_local_news_answer(query, result, analysis):
    if not result:
        return (
            f"本地新闻库暂时没有可用结果（本地库更新时间：{get_news_store_updated_at()}）。我可以先做通用分析；"
            "如果要看实时热点，请稍后再试，系统会继续自动更新新闻库。"
        )

    lines = [
        f"问题：{query}",
        "",
        format_analysis_block(analysis),
        "",
        "新闻总结：",
        f"本次共检索到 {len(result)} 条相关新闻，系统优先展示前 {len(analysis.get('insights', []))} 条进行分析。",
        f"本地新闻库更新时间：{get_news_store_updated_at()}。",
        analysis.get("public_opinion", "当前样本不足，暂时无法形成稳定舆情判断。"),
    ]
    return "\n".join(lines)


def format_latest_news_answer(query, result, analysis, refreshed=False, days=None):
    if not result:
        return format_local_news_answer(query, result, analysis)

    lines = [
        "已先联网刷新新闻源并完成入库去重。" if refreshed else "这是本地新闻库当前可用的最新列表。",
        f"本地新闻库更新时间：{get_news_store_updated_at()}。",
    ]
    if days:
        lines.append(f"本次优先检索近 {days} 天入库或发布的新闻。")
    lines.extend(["", "热点列表："])

    for index, item in enumerate(result[:10], start=1):
        title = safe_text(item.get("title") or "无标题")
        content = safe_text(item.get("content") or item.get("summary") or item.get("desc") or "")
        source = source_credibility_line(item)
        published_at = safe_text(item.get("published_at") or "")
        if len(content) > 90:
            content = content[:90].rstrip() + "..."
        meta = " | ".join(value for value in [source, published_at] if value)
        lines.append(f"{index}. {title}" + (f"（{meta}）" if meta else ""))
        if content:
            lines.append(f"   {content}")

    lines.extend(["", format_analysis_block(analysis)])
    return "\n".join(lines)

# ========= 新闻Agent =========
def news_agent(query, model):
    ensure_news_available()
    needs_fresh_news = is_temporal_news_query(query)
    refreshed = refresh_news_before_answer(query) if needs_fresh_news else False
    news_intelligence.refresh_if_needed()
    recent_days = requested_recent_days(query) if needs_fresh_news else None

    if looks_like_latest_news_query(query) and not looks_like_recent_news_query(query):
        result = latest_news_items(limit=20)
    elif looks_like_recent_news_query(query):
        retrieved = retrieve_news(query, top_k=50)
        result = recent_news_items(days=recent_days or 3, limit=20, items=retrieved)
        if not result:
            result = recent_news_items(days=recent_days or 3, limit=20)
    else:
        result = retrieve_news(query)

    if not deepseek_client.is_enabled():
        analysis = news_intelligence.analyze_news_list(result, top_k=5)
        if needs_fresh_news:
            return format_latest_news_answer(query, result, analysis, refreshed=refreshed, days=recent_days)
        return format_local_news_answer(query, result, analysis)

    if not result:
        try:
            answer = deepseek_client.chat(
                messages=[
                    {"role": "system", "content": "你是 Fairy，一名可靠的中文新闻与热点分析助手。"},
                    {"role": "user", "content": safe_text(build_news_fallback_prompt(query))},
                ],
                temperature=0.45,
                max_tokens=1000,
                model=model,
            )
            return safe_text(answer)
        except Exception as e:
            print("DeepSeek 新闻兜底失败:", e)
            analysis = news_intelligence.analyze_news_list(result, top_k=5)
            return format_local_news_answer(query, result, analysis)

    analysis = news_intelligence.analyze_news_list(result, top_k=5)
    if needs_fresh_news:
        return format_latest_news_answer(query, result, analysis, refreshed=refreshed, days=recent_days)

    analysis_block = format_analysis_block(analysis)

    try:
        answer = deepseek_client.chat(
            messages=[
                {"role": "system", "content": "你是一名擅长新闻检索、情感分析、舆情总结的中文助手。"},
                {"role": "user", "content": safe_text(build_news_prompt(query, result, analysis_block))},
            ],
            temperature=0.4,
            max_tokens=1200,
            model=model,
        )
        return safe_text(answer)
    except Exception as e:
        print("DeepSeek 新闻总结失败:", e)
        return format_local_news_answer(query, result, analysis)

# ========= 聊天Agent =========
def chat_agent(query, history, model):
    messages = [{"role": "system", "content": safe_text("你是一个友好的AI助手，名字叫fairy")}]

    for turn in normalize_history(history):
        messages.append({"role": "user", "content": safe_text(turn["user"])})
        messages.append({"role": "assistant", "content": safe_text(turn["assistant"])})

    messages.append({"role": "user", "content": safe_text(query)})

    if not deepseek_client.is_enabled():
        return "当前未配置 DeepSeek API，我可以继续做新闻检索、分类和本地分析；普通聊天能力请在 `.env` 中设置 `DEEPSEEK_API_KEY` 后重试。"

    try:
        return deepseek_client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            model=model,
        )
    except Exception as e:
        return f"DeepSeek 调用失败: {e}"

# ========= 天气Agent =========
COMMON_CITIES = [
    "北京", "上海", "天津", "重庆", "广州", "深圳", "杭州", "苏州", "南京", "成都",
    "武汉", "西安", "长沙", "郑州", "青岛", "厦门", "宁波", "合肥", "福州", "沈阳",
]


def extract_city_rule(query):
    for city in COMMON_CITIES:
        if city in query:
            return city

    match = re.search(r"([^\s，。！？,.!?\d]{2,6})(?:天气|气温|下雨|温度)", query)
    if match:
        return match.group(1)
    return None


def extract_city_ai(query, model):
    rule_city = extract_city_rule(query)
    if rule_city:
        return rule_city

    prompt = f"""
你是一个信息抽取助手。

任务：从用户输入中提取“城市名称”。

规则：
1. 只返回城市名（如：北京、天津、上海）
2. 不要返回任何解释
3. 不能返回多余内容
4. 如果没有城市，返回：None

示例：
输入：北京的天气 → 北京
输入：帮我查天津天气 → 天津
输入：上海明天天气 → 上海
输入：你好 → None

用户输入：
{query}

输出：
"""

    if not deepseek_client.is_enabled():
        return None

    try:
        city = deepseek_client.chat(
            messages=[{"role": "user", "content": safe_text(prompt)}],
            temperature=0.0,
            max_tokens=32,
            model=model,
        ).strip()
    except Exception as e:
        print("DeepSeek 城市抽取失败:", e)
        return None

    if city == "None":
        return None

    return city

def get_city_id(city_name):
    if not QWEATHER_API_KEY or not QWEATHER_HOST:
        return None

    url = f"https://{QWEATHER_HOST}/geo/v2/city/lookup"

    params = {
        "location": city_name,
        "range": "cn",
        "number": 1
    }

    headers = {"X-QW-Api-Key": QWEATHER_API_KEY}

    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)

        print("DEBUG URL:", res.url)
        print("DEBUG STATUS:", res.status_code)
        print("DEBUG TEXT:", res.text[:200])

        if res.status_code != 200:
            return None

        data = res.json()

        if data.get("code") == "200" and "location" in data:
            return data["location"][0]["id"]

        return None
    except Exception as e:
        print("城市查询失败:", e)
        return None

def weather_agent(query, model):
    city = extract_city_ai(query, model)

    print("DEBUG city:", city)

    if not city:
        return "请说清楚城市，例如：北京天气"

    city_id = get_city_id(city)
    print("DEBUG city_id:", city_id)

    if not city_id:
        return f"找不到城市：{city}"

    if not QWEATHER_API_KEY or not QWEATHER_HOST:
        return "未配置 QWeather（QWEATHER_API_KEY/QWEATHER_HOST），无法查询天气。请在 `.env` 中设置后重启服务。"

    url = f"https://{QWEATHER_HOST}/v7/weather/now"

    params = {
        "location": city_id
    }

    headers = {
        "X-QW-Api-Key": QWEATHER_API_KEY
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)

        print("DEBUG WEATHER URL:", res.url)
        print("DEBUG STATUS:", res.status_code)

        if res.status_code != 200:
            return f"获取 {city} 天气失败（状态码 {res.status_code}）"

        data = res.json()

        if data.get("code") != "200":
            return f"获取 {city} 天气失败"

        now = data.get("now", {})

        return (
            f"📍 城市：{city}\n"
            f"🌤 天气：{now.get('text')}\n"
            f"🌡 温度：{now.get('temp')}°C\n"
            f"💨 风向：{now.get('windDir')} {now.get('windScale')}级\n"
            f"💧 湿度：{now.get('humidity')}%"
        )

    except Exception as e:
        return f"天气查询出错: {e}"

# ========= 工具Agent =========
def tool_agent(query):
    match = re.findall(r'[\d\.\+\-\*\/\(\)]+', query)
    if not match:
        return safe_text("无法识别计算表达式")

    expr = ''.join(match)
    try:
        # 这里一定要用 f-string，把 result 变量填进去
        result = eval(expr)
        return safe_text(f"计算结果是: {result}")
    except Exception as e:
        return safe_text(f"计算时出错: {e}")

# ========= 决策Agent =========
def decide_agent(query, model):
    lowered = query.lower()
    ensure_latest_news_assets()
    if is_temporal_news_query(query) or has_local_news_match(query):
        return "news"
    if any(word in query for word in NEWS_HINT_WORDS) or "news" in lowered:
        return "news"
    if re.search(r"(最近|最新|今日|今天).*(发生|情况|动态|走势|消息)", query):
        return "news"
    if re.search(r"(科技|财经|娱乐|体育|AI).*(新闻|动态|热点|情况|走势)", query):
        return "news"
    environment_terms = (
        "环境数据",
        "环境状态",
        "室内环境",
        "教室环境",
        "实验室环境",
        "传感器",
        "光照",
        "人体红外",
        "有人吗",
        "有人经过",
        "esp32",
    )
    if any(term in lowered for term in environment_terms):
        return "environment"
    if ("室内" in query or "教室" in query or "实验室" in query) and any(
        word in query for word in ["温度", "湿度", "亮度", "有人"]
    ):
        return "environment"
    if any(word in query for word in ["天气", "气温", "下雨", "温度"]):
        return "weather"
    vision_terms = (
        "看到了什么",
        "看到什么",
        "摄像头",
        "视觉识别",
        "识别记录",
        "刚才识别",
        "眼前是什么",
    )
    if any(term in safe_text(query) for term in vision_terms):
        return "vision"
    if travel_agent.matches_query(query):
        return "travel"
    if re.search(r"[\d\.\+\-\*\/\(\)]", query):
        return "tool"

    prompt = f"""
判断用户意图：

用户输入：
{query}

可选类型:
1. news(新闻相关)
2. weather(天气查询)
3. tool(计算/工具类问题)
4. travel(地名与旅游问答)
5. environment(ESP32室内环境监测)
6. chat(日常聊天)

只返回一个词：
"""

    if not deepseek_client.is_enabled():
        return "chat"

    try:
        decision = safe_text(
            deepseek_client.chat(
                messages=[{"role": "user", "content": safe_text(prompt)}],
                temperature=0.0,
                max_tokens=16,
                model=model,
            ).strip().lower()
        )
    except Exception as e:
        print("DeepSeek 意图识别失败:", e)
        return "chat"

    if "news" in decision:
        return "news"
    elif "weather" in decision:
        return "weather"
    elif "tool" in decision:
        return "tool"
    elif "travel" in decision:
        return "travel"
    elif "environment" in decision:
        return "environment"
    else:
        return "chat"

# ========= 总Agent =========
AGENT_PROFILES = {
    "auto": {
        "id": "auto",
        "name": "Fairy",
        "intent": "auto",
        "tone": "orchestrator",
    },
    "news": {
        "id": "news",
        "name": "News Lens",
        "intent": "news",
        "tone": "news",
    },
    "weather": {
        "id": "weather",
        "name": "Sky Trace",
        "intent": "weather",
        "tone": "weather",
    },
    "tool": {
        "id": "tool",
        "name": "Calc Core",
        "intent": "tool",
        "tone": "tool",
    },
    "chat": {
        "id": "chat",
        "name": "Fairy Chat",
        "intent": "chat",
        "tone": "chat",
    },
    "travel": {
        "id": "travel",
        "name": "Geo Voyage",
        "intent": "travel",
        "tone": "travel",
        "shortName": "GEO",
        "glyph": "travel",
        "color": "#e05a47",
        "accent": "#27b8a2",
        "tags": ["PLACE", "LOCAL"],
        "sample": "地名解析与旅游问答",
    },
    "vision": {
        "id": "vision",
        "name": "Vision Guide",
        "intent": "vision",
        "tone": "vision",
        "shortName": "VISION",
        "glyph": "vision",
        "color": "#0f8d8f",
        "accent": "#f3b548",
        "tags": ["YOLO", "K230"],
        "sample": "目标识别与视觉导览",
    },
    "environment": {
        "id": "environment",
        "name": "Environment Guardian",
        "intent": "environment",
        "tone": "environment",
        "shortName": "ENV",
        "glyph": "environment",
        "color": "#11967d",
        "accent": "#7de3a1",
        "tags": ["ESP32", "SENSOR"],
        "sample": "环境监测与异常预警",
    },
}


def get_agent_profiles():
    return list(AGENT_PROFILES.values())


def resolve_agent_intent(agent_id, query, model):
    if not agent_id or agent_id == "auto":
        return decide_agent(query, model)

    profile = AGENT_PROFILES.get(safe_text(agent_id).lower())
    if not profile:
        return decide_agent(query, model)

    return profile["intent"]


def agent_answer(query, history, agent_id=None, model=None):
    identity = identity_answer(query)
    if identity:
        return identity

    selected_model = model or deepseek_client.model
    intent = resolve_agent_intent(agent_id, query, selected_model)

    print(f"[调度] 当前任务类型: {intent}")

    if intent == "news":
        return news_agent(query, selected_model)
    elif intent == "weather":
        return weather_agent(query, selected_model)
    elif intent == "tool":
        return tool_agent(query)
    elif intent == "travel":
        return travel_agent.answer(query, selected_model)
    elif intent == "vision":
        return vision_service.answer(query, selected_model)
    elif intent == "environment":
        return environment_service.answer(query, selected_model)
    else:
        return chat_agent(query, history, selected_model)


def normalize_history(history):
    """
    支持两种格式：
    1) [{"user": "...", "assistant": "..."}, ...]
    2) OpenAI风格: [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    if not history:
        return []

    normalized = []

    # 已经是 turn 格式
    if isinstance(history, list) and history and isinstance(history[0], dict) and ("user" in history[0] or "assistant" in history[0]):
        for t in history:
            u = t.get("user")
            a = t.get("assistant")
            if u is None or a is None:
                continue
            normalized.append({"user": safe_text(u), "assistant": safe_text(a)})
        return normalized

    # role/content 格式 -> turn 格式
    pending_user = None
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").lower()
        content = msg.get("content")
        if content is None:
            continue

        if role == "user":
            pending_user = safe_text(content)
        elif role == "assistant" and pending_user is not None:
            normalized.append({"user": pending_user, "assistant": safe_text(content)})
            pending_user = None

    return normalized

# ========= REPL =========
if __name__ == "__main__":
    print("你好，我是fairy，你的好助手！")
    print("输入 再见 退出\n")

    chat_history = []

    while True:
        user_query = input("你: ")

        if user_query.lower() == "再见":
            break

        answer = agent_answer(user_query, chat_history)

        print("fairy:", answer)

        chat_history.append({
            "user": safe_text(user_query),
            "assistant": safe_text(answer)
        })
