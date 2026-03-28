import numpy as np
import json
from dotenv import load_dotenv
import os
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from deepseek_client import deepseek_client
from news_intelligence import NewsIntelligenceService, format_analysis_block

load_dotenv()

# 定义常量
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_PATH = os.path.join(_BASE_DIR, "data", "news_embeddings.npy")
NEWS_PATH = os.path.join(_BASE_DIR, "data", "news.json")
SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ========= 初始化 =========
embedding_model = None
embedding_provider = "tfidf"
news_vectorizer = None
news_intelligence = NewsIntelligenceService()
# ========= 数据 =========
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
QWEATHER_HOST = os.getenv("QWEATHER_HOST")


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

def load_news():
    if not os.path.exists(NEWS_PATH):
        return []
    with open(NEWS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_embeddings():
    if not os.path.exists(EMBED_PATH):
        return None
    return np.load(EMBED_PATH)

news_list = load_news()
news_embeddings = load_embeddings()

# ========= 新闻向量 =========

def prepare_news_embeddings():
    global news_texts, news_embeddings, news_vectorizer, embedding_provider

    news_texts = []
    for news in news_list:
        title = news.get("title") or ""
        content = news.get("content") or ""
        news_texts.append((title * 3) + " " + content)

    if not news_texts:
        news_embeddings = None
        news_vectorizer = None
        return

    if embedding_model is not None and os.path.exists(EMBED_PATH):
        print("加载本地 embedding")
        news_embeddings = np.load(EMBED_PATH)
    else:
        news_embeddings = None

    if embedding_model is not None and news_embeddings is None:
        print("没有可用的预计算 embedding，现场计算")
        news_embeddings = embedding_model.encode(news_texts)
        news_embeddings = news_embeddings / np.linalg.norm(news_embeddings, axis=1, keepdims=True)

    if embedding_model is not None and news_embeddings is not None:
        if len(news_embeddings.shape) == 1:
            news_embeddings = np.expand_dims(news_embeddings, axis=0)
        if news_embeddings.shape[0] != len(news_list):
            print("embedding数量与新闻数量不一致，重新生成embedding")
            news_embeddings = embedding_model.encode(news_texts)
            news_embeddings = news_embeddings / np.linalg.norm(news_embeddings, axis=1, keepdims=True)
        embedding_provider = "sentence_transformer"
        news_vectorizer = None
        return

    print("使用 TF-IDF 构建本地检索索引")
    news_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    news_embeddings = news_vectorizer.fit_transform(news_texts).toarray().astype(np.float32)
    embedding_provider = "tfidf"

prepare_news_embeddings()

def reload_news():
    global news_list, news_embeddings
    news_list = load_news()
    prepare_news_embeddings()
    print("新闻数据已更新并重建embedding")

# ========= 工具 =========
def keyword_match(query, top_k=10):
    results = []

    for news in news_list:
        title = news.get("title") or ""
        content = news.get("content") or ""

        if query in title or query in content:
            results.append(news)

    return results[:top_k]

def retrieve_news(query, top_k=20):
    if news_embeddings is None:
        return []

    keyword_results = keyword_match(query)

    if len(keyword_results) >= 5:
        print("[检索] 使用关键词命中")
        return keyword_results[:top_k]

    if embedding_provider == "sentence_transformer" and embedding_model is not None:
        query_embedding = embedding_model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    else:
        if news_vectorizer is None:
            return keyword_results[:top_k]
        query_embedding = news_vectorizer.transform([query]).toarray()[0].astype(np.float32)

    scores = np.dot(news_embeddings, query_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    embed_results = [news_list[i] for i in top_indices if i < len(news_list)]

    combined = keyword_results + embed_results

    seen = set()
    final_results = []

    for news in combined:
        title = news.get("title", "")
        if title not in seen:
            seen.add(title)
            final_results.append(news)

    print(f"[检索] keyword:{len(keyword_results)} | embedding:{len(embed_results)}")
    return final_results[:top_k]


def build_news_prompt(query, result, analysis_block):
    picked = result[:5]
    return f"""
你是一个新闻分析专家，请根据检索到的新闻回答用户问题。

用户问题：
{query}

检索新闻：
{json.dumps(picked, ensure_ascii=False, indent=2)}

智能分析结果：
{analysis_block}

回答要求：
1. 先给出简洁结论，再给出原因。
2. 可以概括行业趋势、舆情倾向、潜在风险。
3. 如果涉及情感分析，请明确说明偏正面、负面还是中立。
4. 保持条理清晰，适合课程大作业展示。
"""


def format_local_news_answer(query, result, analysis):
    if not result:
        return "没有检索到相关新闻，你可以换个关键词再试试。"

    lines = [
        f"问题：{query}",
        "",
        format_analysis_block(analysis),
        "",
        "新闻总结：",
        f"本次共检索到 {len(result)} 条相关新闻，系统优先展示前 {len(analysis.get('insights', []))} 条进行分析。",
        analysis.get("public_opinion", "当前样本不足，暂时无法形成稳定舆情判断。"),
    ]
    return "\n".join(lines)

# ========= 新闻Agent =========
def news_agent(query):
    result = retrieve_news(query)
    analysis = news_intelligence.analyze_news_list(result, top_k=5)
    analysis_block = format_analysis_block(analysis)

    if not deepseek_client.is_enabled():
        return format_local_news_answer(query, result, analysis)

    try:
        answer = deepseek_client.chat(
            messages=[
                {"role": "system", "content": "你是一名擅长新闻检索、情感分析、舆情总结的中文助手。"},
                {"role": "user", "content": safe_text(build_news_prompt(query, result, analysis_block))},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        return safe_text(answer)
    except Exception as e:
        print("DeepSeek 新闻总结失败:", e)
        return format_local_news_answer(query, result, analysis)

# ========= 聊天Agent =========
def chat_agent(query, history):
    messages = [{"role": "system", "content": safe_text("你是一个友好的AI助手，名字叫fairy")}]

    for turn in normalize_history(history):
        messages.append({"role": "user", "content": safe_text(turn["user"])})
        messages.append({"role": "assistant", "content": safe_text(turn["assistant"])})

    messages.append({"role": "user", "content": safe_text(query)})

    if not deepseek_client.is_enabled():
        return "当前未配置 DeepSeek API，我可以继续做新闻检索、分类和本地分析；普通聊天能力请在 `.env` 中设置 `DEEPSEEK_API_KEY` 后重试。"

    try:
        return deepseek_client.chat(messages=messages, temperature=0.7, max_tokens=1024)
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


def extract_city_ai(query):
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

def weather_agent(query):
    city = extract_city_ai(query)

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
def decide_agent(query):
    lowered = query.lower()
    if any(word in query for word in ["新闻", "热点", "舆情", "情感分析", "AI行业", "最近发生"]) or "news" in lowered:
        return "news"
    if any(word in query for word in ["天气", "气温", "下雨", "温度"]):
        return "weather"
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
4. chat(日常聊天)

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
    else:
        return "chat"

# ========= 总Agent =========
def agent_answer(query, history):
    intent = decide_agent(query)

    print(f"[调度] 当前任务类型: {intent}")

    if intent == "news":
        return news_agent(query)
    elif intent == "weather":
        return weather_agent(query)
    elif intent == "tool":
        return tool_agent(query)
    else:
        return chat_agent(query, history)


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
