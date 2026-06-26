from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fairy_core.deepseek_client import deepseek_client
from fairy_core.paths import PROJECT_ROOT
from fairy_core.travel_qa.qa_service import TravelQAService


TOURISM_PROMPT = """你是中国旅游问答资料整理助手。
用户会提供一个由本地行政区划数据库解析出的地点 JSON。

你必须遵守：
- 行政区划必须以输入 JSON 为准，不要改省、市、区县归属。
- 只输出可靠、常见、适合旅游问答系统展示的信息。
- 如果是区县，优先给该区县内或所属城市范围内有代表性的内容，并说明范围。
- 不要输出具体票价、开放时间、电话、公交线路编号、实时活动等容易过期的信息。
- travel_tips 只写通用出行建议，不写实时交通和具体班次。
- 输出必须是严格 JSON，不要 Markdown，不要代码块。

JSON 格式：
{
  "intro": "80到150字地区简介",
  "attractions": ["景点1", "景点2", "景点3", "景点4", "景点5"],
  "foods": ["美食1", "美食2", "美食3", "美食4", "美食5"],
  "travel_tips": ["提示1", "提示2"]
}
"""

PLACE_INTENT_WORDS = (
    "地名",
    "行政区划",
    "属于",
    "哪里",
    "在哪",
    "旅游",
    "旅行",
    "景点",
    "美食",
    "简介",
    "介绍",
    "攻略",
    "路线",
    "怎么玩",
    "好吃",
    "好玩",
)


class FairyTravelAgent:
    """Fairy adapter around the original local place-name QA pipeline."""

    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root)
        self.data_root = root / "data" / "travel"
        self.qa_service = TravelQAService(self.data_root)
        self.knowledge_path = self.data_root / "data" / "tourism_knowledge.json"
        self._knowledge_lock = threading.Lock()

    def matches_query(self, query: str) -> bool:
        local_result = self.qa_service.answer(query)
        if local_result["candidate_count"] == 0:
            return False
        compact_len = len("".join(query.split()))
        return compact_len <= 8 or any(word in query for word in PLACE_INTENT_WORDS)

    def answer(self, query: str, model: str) -> str:
        local_result = self.qa_service.answer(query)
        if local_result["candidate_count"] != 1:
            return self.format_local_answer(local_result)

        candidate = local_result["candidates"][0]
        tourism = candidate.get("tourism", {})
        if self._has_complete_local_tourism(tourism):
            return self.format_enriched_answer(candidate, tourism)

        cached = self._find_cached(candidate)
        if cached is not None:
            return self.format_enriched_answer(candidate, cached)

        if not deepseek_client.is_enabled():
            return self.format_local_answer(local_result)

        try:
            generated = self._generate(candidate, model)
            return self.format_enriched_answer(candidate, generated)
        except Exception as exc:
            return f"{self.format_local_answer(local_result)}\n\n旅游资料补全失败：{exc}"

    def _generate(self, candidate: dict[str, Any], model: str) -> dict[str, Any]:
        payload = {
            "code": candidate["code"],
            "name": candidate["name"],
            "level": candidate["level"],
            "province": candidate["province"],
            "city": candidate["city"],
            "area": candidate["area"],
            "path": candidate["path"],
            "local_tourism": candidate.get("tourism", {}),
        }
        content = deepseek_client.chat(
            messages=[
                {"role": "system", "content": TOURISM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.3,
            max_tokens=900,
            model=model,
        )
        normalized = self._normalize(self._parse_json(content))
        normalized.update(
            {
                "code": str(candidate["code"]),
                "name": candidate["name"],
                "path": candidate["path"],
                "source": "deepseek_generated",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save(candidate["code"], normalized)
        return normalized

    def _find_cached(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        knowledge = self._load_knowledge()
        item = knowledge.get(str(candidate["code"]))
        if not isinstance(item, dict):
            return None
        result = dict(item)
        result["source"] = "local_knowledge"
        return result

    def _load_knowledge(self) -> dict[str, dict[str, Any]]:
        if not self.knowledge_path.exists():
            return {}
        try:
            with self.knowledge_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                normalized[key.rsplit(":", 1)[-1]] = dict(value)
        return normalized

    def _save(self, code: str, item: dict[str, Any]) -> None:
        with self._knowledge_lock:
            knowledge = self._load_knowledge()
            knowledge[str(code)] = item
            self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.knowledge_path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(knowledge, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.knowledge_path)

    @staticmethod
    def _has_complete_local_tourism(tourism: dict[str, Any]) -> bool:
        return (
            tourism.get("source") == "local_seed"
            and bool(tourism.get("intro"))
            and bool(tourism.get("attractions"))
            and bool(tourism.get("foods"))
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= start:
            content = content[start : end + 1]
        return json.loads(content)

    @staticmethod
    def _normalize(data: dict[str, Any]) -> dict[str, Any]:
        def as_list(value: Any, limit: int) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()][:limit]

        return {
            "intro": str(data.get("intro", "")).strip(),
            "attractions": as_list(data.get("attractions"), 8),
            "foods": as_list(data.get("foods"), 8),
            "travel_tips": as_list(data.get("travel_tips"), 4),
        }

    @staticmethod
    def format_local_answer(result: dict[str, Any]) -> str:
        if result["candidate_count"] == 0:
            return result["message"]

        lines = [result["message"]]
        for index, candidate in enumerate(result["candidates"], start=1):
            tourism = candidate["tourism"]
            lines.extend(
                [
                    "",
                    f"候选 {index}: {' -> '.join(candidate['path'])}",
                    f"行政代码: {candidate['code']}",
                    f"景点: {', '.join(tourism['attractions']) or '暂无'}",
                    f"美食: {', '.join(tourism['foods']) or '暂无'}",
                    f"简介: {tourism['intro']}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def format_enriched_answer(candidate: dict[str, Any], tourism: dict[str, Any]) -> str:
        lines = [
            f"行政区划: {' -> '.join(candidate['path'])}",
            f"行政代码: {candidate['code']}",
            f"简介: {tourism.get('intro') or '暂无'}",
            f"景点: {', '.join(tourism.get('attractions', [])) or '暂无'}",
            f"美食: {', '.join(tourism.get('foods', [])) or '暂无'}",
        ]
        tips = tourism.get("travel_tips", [])
        if tips:
            lines.append(f"出行建议: {'；'.join(tips)}")
        return "\n".join(lines)


travel_agent = FairyTravelAgent(PROJECT_ROOT)
