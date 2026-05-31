import json
import html
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import quote_plus, urljoin, urlparse

import numpy as np
import requests

from deepseek_client import deepseek_client
from news_retrieval import save_vector_store, weighted_news_text

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ========= 配置 =========
NEWS_API = os.getenv(
    "NEWS_TENCENT_HOT_API_URL",
    "https://i.news.qq.com/web_feed/get_command_pagination?page=1",
)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(_BASE_DIR, "data", "news.json")
EMBED_PATH = os.path.join(_BASE_DIR, "data", "news_embeddings.npy")
GENERATED_TRAIN_PATH = os.path.join(_BASE_DIR, "data", "news_train_generated.jsonl")
DEFAULT_TENCENT_PAGE_URLS = [
    "https://news.qq.com/",
    "https://new.qq.com/",
    "https://new.qq.com/ch/tech/",
    "https://new.qq.com/ch/finance/",
    "https://new.qq.com/ch/sports/",
    "https://new.qq.com/ch/ent/",
]
DEFAULT_TENCENT_API_URLS = [
    NEWS_API,
    "https://i.news.qq.com/web_feed/get_command_pagination?page=1",
    "http://api.xcvts.cn/api/hotlist/qq_news?type=new",
]
DEFAULT_RSS_URLS = [
    "https://news.google.com/rss?hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    "https://feeds.bbci.co.uk/zhongwen/simp/rss.xml",
]
GOOGLE_NEWS_SEARCH_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
)
TRUSTED_DOMAIN_SCORES = {
    "qq.com": 0.82,
    "news.qq.com": 0.88,
    "new.qq.com": 0.86,
    "bbc.co.uk": 0.86,
    "bbc.com": 0.86,
    "news.google.com": 0.74,
}
PROVIDER_BASE_SCORES = {
    "tencent_hot_api": 0.82,
    "tencent_web": 0.84,
    "rss": 0.68,
    "deepseek_fallback": 0.25,
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}

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


def env_list(name, default):
    configured = os.getenv(name, "").strip()
    if configured:
        return [value.strip() for value in configured.split(",") if value.strip()]
    return list(default)


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
            news = json.load(f)
        if not env_flag("NEWS_KEEP_LLM_FALLBACK_ITEMS", False):
            news = [item for item in news if item.get("provider") != "deepseek_fallback"]
        return news
    except:
        return []

def normalized_utc_now():
    return datetime.now(timezone.utc).isoformat()


def strip_html(text):
    if not isinstance(text, str):
        text = str(text or "")
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_published_at(raw_value):
    if raw_value is None:
        return ""

    if isinstance(raw_value, (int, float)):
        try:
            value = float(raw_value)
            if value > 10_000_000_000:
                value = value / 1000
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        except Exception:
            return str(raw_value).strip()

    cleaned = str(raw_value).strip()
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return normalize_published_at(int(cleaned))
    return cleaned


def compact_title_key(title):
    return re.sub(r"[\W_]+", "", (title or "").lower(), flags=re.UNICODE)


def source_domain(url):
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def calculate_source_credibility_score(item, provider, default_source):
    url = str(item.get("url") or item.get("link") or item.get("jumpUrl") or "").strip()
    source = strip_html(str(item.get("source") or item.get("media") or item.get("author") or default_source))
    domain = source_domain(url)
    score = PROVIDER_BASE_SCORES.get(provider, 0.55)

    for trusted_domain, trusted_score in TRUSTED_DOMAIN_SCORES.items():
        if domain == trusted_domain or domain.endswith("." + trusted_domain):
            score = max(score, trusted_score)

    if source and source not in {"RSS", "DeepSeek fallback"}:
        score += 0.04
    if item.get("published_at") or item.get("publish_time") or item.get("pubDate") or item.get("time"):
        score += 0.03
    if provider == "deepseek_fallback":
        score = min(score, 0.35)
    return round(min(max(score, 0.0), 0.98), 2)


def normalize_news_item(item, provider, default_source):
    title = strip_html(
        item.get("title")
        or item.get("name")
        or item.get("ztTitle")
        or item.get("sTitle")
        or item.get("headline")
        or ""
    )
    content = strip_html(
        item.get("desc")
        or item.get("summary")
        or item.get("abstract")
        or item.get("intro")
        or item.get("content")
        or item.get("description")
        or ""
    )
    source = strip_html(
        item.get("source")
        or item.get("media")
        or item.get("author")
        or item.get("chlname")
        or default_source
    )
    url = str(
        item.get("url")
        or item.get("link")
        or item.get("jumpUrl")
        or item.get("vurl")
        or item.get("surl")
        or ""
    ).strip()
    published_at = normalize_published_at(
        item.get("published_at")
        or item.get("publish_time")
        or item.get("pub_time")
        or item.get("time")
        or item.get("ctime")
        or item.get("timestamp")
        or ""
    )

    if not title:
        return None

    return {
        "title": title,
        "content": content,
        "source": source or default_source,
        "url": url,
        "published_at": published_at,
        "fetched_at": normalized_utc_now(),
        "provider": provider,
        "source_domain": source_domain(url),
        "source_credibility_score": calculate_source_credibility_score(
            {**item, "url": url, "source": source, "published_at": published_at},
            provider,
            default_source,
        ),
    }


def iter_dicts(value, max_items=800):
    stack = [value]
    count = 0
    while stack and count < max_items:
        current = stack.pop()
        if isinstance(current, dict):
            count += 1
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def dedupe_new_items(items, limit=None):
    unique_items = []
    seen = set()
    for item in items:
        title_key = compact_title_key(item.get("title", ""))
        url_key = (item.get("url") or "").split("?")[0]
        dedupe_key = title_key or url_key
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_items.append(item)
        if limit and len(unique_items) >= limit:
            break
    return unique_items


def normalized_url_key(url):
    if not url:
        return ""
    parsed = urlparse(str(url).strip())
    if not parsed.netloc:
        return ""
    return f"{parsed.scheme or 'https'}://{parsed.netloc.lower().removeprefix('www.')}{parsed.path}".rstrip("/")


def news_identity_keys(item):
    title_key = compact_title_key(item.get("title", ""))
    url_key = normalized_url_key(item.get("url") or "")
    return {key for key in [title_key, url_key] if key}


def fetch_news_from_hot_api():
    print("🚀 正在抓取腾讯新闻热榜 API...")
    news_list = []
    for api_url in env_list("NEWS_TENCENT_HOT_API_URLS", DEFAULT_TENCENT_API_URLS):
        try:
            res = requests.get(api_url, timeout=10, headers=REQUEST_HEADERS)
            res.raise_for_status()
            data = res.json()

            for raw_item in iter_dicts(data):
                item = normalize_news_item(raw_item, "tencent_hot_api", "腾讯新闻")
                if item:
                    news_list.append(item)

            print(f"✅ 腾讯热榜 API 抓取成功: {len(news_list)}条 | {api_url}")
            if news_list:
                break
        except Exception as e:
            print(f"⚠️ 腾讯热榜 API 抓取失败: {api_url} | {e}")

    limit = max(1, env_int("NEWS_TENCENT_MAX_ITEMS", 30))
    return dedupe_new_items(news_list, limit=limit)


class TencentAnchorParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.anchors = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        href = (attr_map.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "#")):
            return
        self._current = {
            "href": urljoin(self.base_url, href),
            "title": attr_map.get("title") or attr_map.get("aria-label") or "",
            "text": "",
        }

    def handle_data(self, data):
        if self._current is not None:
            self._current["text"] += data

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._current is not None:
            self.anchors.append(self._current)
            self._current = None


def is_tencent_news_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not (host.endswith("qq.com") and ("news" in host or host.startswith("new."))):
        return False
    if any(blocked in host for blocked in ["video.qq.com", "mail.qq.com"]):
        return False
    path = parsed.path.lower()
    if path in {"", "/"}:
        return False
    return bool(re.search(r"(/rain/a/|/omn/|/a/|/cmsn/|/ch/|/news/|/article/|/content/)", path))


def parse_tencent_page_items(html_text, page_url):
    parser = TencentAnchorParser(page_url)
    try:
        parser.feed(html_text)
    except Exception:
        pass

    fetched_at = normalized_utc_now()
    items = []
    for anchor in parser.anchors:
        url = anchor["href"].split("#")[0]
        title = strip_html(anchor.get("title") or anchor.get("text") or "")
        if len(title) < 6 or not is_tencent_news_url(url):
            continue
        items.append(
            {
                "title": title,
                "content": title,
                "source": "腾讯新闻",
                "url": url,
                "published_at": "",
                "fetched_at": fetched_at,
                "provider": "tencent_web",
                "source_domain": source_domain(url),
                "source_credibility_score": calculate_source_credibility_score(
                    {"url": url, "source": "腾讯新闻"},
                    "tencent_web",
                    "腾讯新闻",
                ),
            }
        )

    return dedupe_new_items(items)


class ArticleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_script_like = False
        self.capture_stack = []
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.in_script_like = True
        if tag in {"p", "article", "section", "div"}:
            attr_text = " ".join(str(value or "") for _, value in attrs).lower()
            if tag == "p" or any(key in attr_text for key in ["article", "content", "text", "正文", "main"]):
                self.capture_stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.in_script_like = False
        if self.capture_stack and tag == self.capture_stack[-1]:
            self.capture_stack.pop()

    def handle_data(self, data):
        if self.in_script_like or not self.capture_stack:
            return
        cleaned = strip_html(data)
        if len(cleaned) >= 12:
            self.chunks.append(cleaned)


def extract_meta_description(html_text):
    patterns = [
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE)
        if match:
            return strip_html(match.group(1))
    return ""


def fetch_article_text(url):
    if not url or not env_flag("NEWS_FETCH_ARTICLE_TEXT_ENABLED", True):
        return ""
    try:
        res = requests.get(url, timeout=10, headers=REQUEST_HEADERS)
        res.raise_for_status()
        content_type = res.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return ""

        meta_desc = extract_meta_description(res.text)
        parser = ArticleTextParser()
        parser.feed(res.text)
        text = strip_html("。".join(parser.chunks))
        if meta_desc and meta_desc not in text:
            text = f"{meta_desc}。{text}"
        max_chars = max(300, env_int("NEWS_ARTICLE_TEXT_MAX_CHARS", 2400))
        return text[:max_chars].strip()
    except Exception as exc:
        print(f"⚠️ 原文抓取失败: {url} | {exc}")
        return ""


def enrich_news_items(news_list):
    if not news_list:
        return news_list

    max_items = max(0, env_int("NEWS_FETCH_ARTICLE_TEXT_MAX_ITEMS", 18))
    enriched = []
    for index, item in enumerate(news_list):
        item = dict(item)
        item["source_domain"] = item.get("source_domain") or source_domain(item.get("url") or "")
        item["source_credibility_score"] = item.get("source_credibility_score") or calculate_source_credibility_score(
            item,
            item.get("provider") or "unknown",
            item.get("source") or "未知来源",
        )
        if index < max_items and item.get("url") and not item.get("article_text"):
            article_text = fetch_article_text(item["url"])
            if article_text:
                item["article_text"] = article_text
                if len(item.get("content") or "") < 80:
                    item["content"] = article_text[:500]
                item["article_fetch_status"] = "ok"
            else:
                item["article_fetch_status"] = "empty"
        enriched.append(item)
    return enriched


def fetch_news_from_tencent_pages():
    print("🌐 正在抓取腾讯新闻网页...")
    news_list = []
    limit = max(1, env_int("NEWS_TENCENT_MAX_ITEMS", 30))

    for page_url in env_list("NEWS_TENCENT_PAGE_URLS", DEFAULT_TENCENT_PAGE_URLS):
        if len(news_list) >= limit:
            break
        try:
            res = requests.get(page_url, timeout=12, headers=REQUEST_HEADERS)
            res.raise_for_status()
            items = parse_tencent_page_items(res.text, page_url)
            print(f"✅ 腾讯网页抓取成功: {len(items)}条 | {page_url}")
            news_list.extend(items)
            news_list = dedupe_new_items(news_list, limit=limit)
        except Exception as e:
            print(f"⚠️ 腾讯网页抓取失败: {page_url} | {e}")

    return dedupe_new_items(news_list, limit=limit)


def fetch_news_from_tencent():
    if not env_flag("NEWS_TENCENT_ENABLED", True):
        print("⏭️ 腾讯新闻源已关闭")
        return []

    api_news = fetch_news_from_hot_api()
    page_news = fetch_news_from_tencent_pages()
    news_list = dedupe_new_items(api_news + page_news, limit=max(1, env_int("NEWS_TENCENT_MAX_ITEMS", 30)))
    print(f"✅ 腾讯新闻源合计: {len(news_list)}条")
    return news_list


def fetch_news():
    print("🚀 正在抓取新闻 Agent 2.0 多源新闻...")
    try:
        news_list = fetch_news_from_tencent()
        if env_flag("NEWS_RSS_SUPPLEMENT_ENABLED", True):
            rss_news = fetch_news_from_rss()
            news_list = dedupe_new_items(news_list + rss_news)
        print(f"✅ 多源抓取完成: {len(news_list)}条")
        return news_list

    except Exception as e:
        print("❌ 抓取失败:", e)
        return []


def parse_rss_datetime(raw_value):
    if not raw_value:
        return ""
    try:
        return parsedate_to_datetime(raw_value).astimezone(timezone.utc).isoformat()
    except Exception:
        return str(raw_value).strip()


def rss_urls():
    return env_list("NEWS_RSS_URLS", DEFAULT_RSS_URLS)


def parse_rss_items(xml_text, source_url):
    root = ET.fromstring(xml_text)
    fetched_at = datetime.now(timezone.utc).isoformat()
    items = []

    for item in root.findall(".//item"):
        title = strip_html(item.findtext("title") or "")
        content = strip_html(
            item.findtext("description")
            or item.findtext("{http://search.yahoo.com/mrss/}description")
            or ""
        )
        link = (item.findtext("link") or "").strip()
        source = strip_html(item.findtext("source") or "")
        published_at = parse_rss_datetime(item.findtext("pubDate") or item.findtext("published"))
        if not title:
            continue
        items.append(
            {
                "title": title,
                "content": content,
                "source": source or "RSS",
                "url": link,
                "published_at": published_at,
                "fetched_at": fetched_at,
                "provider": "rss",
                "feed_url": source_url,
                "source_domain": source_domain(link or source_url),
                "source_credibility_score": calculate_source_credibility_score(
                    {"url": link or source_url, "source": source or "RSS", "published_at": published_at},
                    "rss",
                    source or "RSS",
                ),
            }
        )

    atom_ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.findall(f".//{atom_ns}entry"):
        title = strip_html(entry.findtext(f"{atom_ns}title") or "")
        content = strip_html(
            entry.findtext(f"{atom_ns}summary")
            or entry.findtext(f"{atom_ns}content")
            or ""
        )
        link = ""
        link_node = entry.find(f"{atom_ns}link")
        if link_node is not None:
            link = (link_node.attrib.get("href") or "").strip()
        source = strip_html(entry.findtext(f"{atom_ns}source/{atom_ns}title") or "")
        published_at = entry.findtext(f"{atom_ns}published") or entry.findtext(f"{atom_ns}updated") or ""
        if not title:
            continue
        items.append(
            {
                "title": title,
                "content": content,
                "source": source or "RSS",
                "url": link,
                "published_at": str(published_at).strip(),
                "fetched_at": fetched_at,
                "provider": "rss",
                "feed_url": source_url,
                "source_domain": source_domain(link or source_url),
                "source_credibility_score": calculate_source_credibility_score(
                    {"url": link or source_url, "source": source or "RSS", "published_at": published_at},
                    "rss",
                    source or "RSS",
                ),
            }
        )

    return items


def fetch_news_from_rss():
    if not env_flag("NEWS_RSS_ENABLED", True):
        print("⏭️ RSS 新闻源已关闭")
        return []

    news_list = []
    for url in rss_urls():
        try:
            res = requests.get(
                url,
                timeout=12,
                headers=REQUEST_HEADERS,
            )
            res.raise_for_status()
            items = parse_rss_items(res.text, url)
            print(f"✅ RSS 抓取成功: {len(items)}条 | {url}")
            news_list.extend(items)
        except Exception as e:
            print(f"⚠️ RSS 抓取失败: {url} | {e}")

    return merge_news([], news_list)


def normalize_news_search_query(query):
    cleaned = strip_html(query or "")
    cleaned = re.sub(
        r"(最新|最近|今日|今天|现在|目前|这几天|近几天|过去\d+天|近\d+天|新闻|热点|消息|动态|资讯|有哪些|有什么|帮我|查询|搜索)",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "热点 新闻"


def fetch_news_from_search_rss(query):
    if not env_flag("NEWS_SEARCH_RSS_ENABLED", True):
        print("⏭️ 新闻搜索 RSS 已关闭")
        return []

    search_query = normalize_news_search_query(query)
    max_items = max(1, env_int("NEWS_SEARCH_RSS_MAX_ITEMS", 30))
    url = os.getenv("NEWS_SEARCH_RSS_URL", GOOGLE_NEWS_SEARCH_RSS).format(
        query=quote_plus(search_query)
    )

    try:
        res = requests.get(url, timeout=12, headers=REQUEST_HEADERS)
        res.raise_for_status()
        items = parse_rss_items(res.text, url)
        for item in items:
            item["provider"] = "search_rss"
            item["search_query"] = search_query
            item["source_credibility_score"] = calculate_source_credibility_score(
                item,
                "rss",
                item.get("source") or "Google News",
            )
        items = dedupe_new_items(items, limit=max_items)
        print(f"✅ 新闻搜索 RSS 抓取成功: {len(items)}条 | {search_query}")
        return items
    except Exception as e:
        print(f"⚠️ 新闻搜索 RSS 抓取失败: {search_query} | {e}")
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
            "source_domain": "",
            "source_credibility_score": calculate_source_credibility_score(
                item,
                "deepseek_fallback",
                "DeepSeek fallback",
            ),
        })
        seen_titles.add(title)

    return normalized


def fetch_news_with_llm():
    if not env_flag("NEWS_LLM_FALLBACK_ENABLED", False):
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

    merged = new_news + old_news

    seen = set()
    unique_news = []

    for news in merged:
        keys = news_identity_keys(news)
        if not keys or seen.intersection(keys):
            continue
        seen.update(keys)
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

    texts = [weighted_news_text(news) for news in news_list]

    if not texts:
        print("⚠️ 没有新闻文本，跳过embedding生成")
        return

    embeddings = embedding_model.encode(texts)
    embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    np.save(EMBED_PATH, embeddings)
    save_vector_store(embeddings, news_list, "sentence_transformer", SENTENCE_MODEL_NAME)

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
        os.path.join("scripts", "train_transformer_news.py"),
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


def persist_news_update(new_news, train_transformer=False):
    old_news = load_old_news()
    if not new_news:
        print("⚠️ 没有数据，不更新")
        return False

    enriched_news = enrich_news_items(new_news)
    merged_news = merge_news(old_news, enriched_news)
    if merged_news == old_news:
        print("ℹ️ 新闻列表无新增内容，跳过重建与训练")
        return False

    save_news(merged_news)
    build_embeddings(merged_news)

    generated_rows = build_generated_training_rows(merged_news)
    save_generated_training_rows(generated_rows)
    train_transformer_if_needed(train_transformer, generated_rows)
    return True


def run_news_pipeline(train_transformer=False):
    new_news = fetch_news()
    if not new_news:
        print("🔁 多源新闻不可用，尝试 RSS 新闻源...")
        new_news = fetch_news_from_rss()
    if not new_news:
        print("🔁 主新闻源不可用，尝试 LLM 新闻兜底...")
        new_news = fetch_news_with_llm()
    return persist_news_update(new_news, train_transformer=train_transformer)


def refresh_news_for_query(query, train_transformer=False):
    search_news = fetch_news_from_search_rss(query)
    general_news = []
    if env_flag("NEWS_QUERY_REFRESH_INCLUDE_GENERAL", True):
        general_news = fetch_news()
    new_news = dedupe_new_items(search_news + general_news)
    if not new_news:
        print("🔁 查询新闻源不可用，尝试 RSS 新闻源...")
        new_news = fetch_news_from_rss()
    if not new_news:
        print("🔁 查询新闻源不可用，尝试 LLM 新闻兜底...")
        new_news = fetch_news_with_llm()
    return persist_news_update(new_news, train_transformer=train_transformer)

# ========= 主流程 =========
if __name__ == "__main__":
    train_enabled = os.getenv("NEWS_AUTO_TRAIN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    run_news_pipeline(train_transformer=train_enabled)
