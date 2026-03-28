import requests
import json
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ========= 配置 =========
NEWS_API = "http://api.xcvts.cn/api/hotlist/qq_news?type=new"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(_BASE_DIR, "data", "news.json")
EMBED_PATH = os.path.join(_BASE_DIR, "data", "news_embeddings.npy")

SENTENCE_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = None


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

# ========= 主流程 =========
if __name__ == "__main__":
    old_news = load_old_news()
    new_news = fetch_news()

    if new_news:
        merged_news = merge_news(old_news, new_news) #合并去重
        save_news(merged_news)
        build_embeddings(merged_news)
    else:
        print("⚠️ 没有数据，不更新")
