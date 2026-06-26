from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .models import PlaceRecord, TourismInfo


class TourismRepository:
    """Load seeded tourism knowledge and provide lightweight retrieval."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.items = self._load_items()
        self.vectors = {
            code: self._vectorize(self._doc_text(item))
            for code, item in self.items.items()
        }

    def get_info(self, place: PlaceRecord, query: str = "") -> TourismInfo:
        matched_code, item = self._find_best_item(place, query)
        if item is None:
            return self._fallback(place)
        exact_codes = {place.area_code, place.city_code, place.province_code}
        source = "local_seed" if matched_code in exact_codes and matched_code == place.code else "parent_seed"
        return TourismInfo(
            code=matched_code,
            attractions=item.get("attractions", []),
            foods=item.get("foods", []),
            intro=item.get("intro", ""),
            source=source,
        )

    def _load_items(self) -> dict[str, dict[str, object]]:
        if not self.data_path.exists():
            return {}
        with self.data_path.open("r", encoding="utf-8") as file:
            raw_items = json.load(file)
        return {str(item["code"]): item for item in raw_items}

    def _find_best_item(self, place: PlaceRecord, query: str) -> tuple[str | None, dict[str, object] | None]:
        for code in (place.area_code, place.city_code, place.province_code):
            if code and code in self.items:
                return code, self.items[code]

        if not query or not self.items:
            return None, None

        query_vector = self._vectorize(query)
        ranked = sorted(
            self.items.items(),
            key=lambda pair: self._cosine(query_vector, self.vectors[pair[0]]),
            reverse=True,
        )
        best_code, best_item = ranked[0]
        if self._cosine(query_vector, self.vectors[best_code]) > 0.15:
            return best_code, best_item
        return None, None

    @staticmethod
    def _fallback(place: PlaceRecord) -> TourismInfo:
        target = " / ".join(place.path) or place.name
        return TourismInfo(
            code=None,
            attractions=[],
            foods=[],
            intro=f"{target} 暂无本地旅游知识库记录，可通过 DeepSeek 补全景点、美食和简介数据。",
            source="fallback",
        )

    @staticmethod
    def _doc_text(item: dict[str, object]) -> str:
        parts: list[str] = []
        for key in ("name", "intro"):
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
        for key in ("attractions", "foods"):
            value = item.get(key)
            if isinstance(value, list):
                parts.extend(str(part) for part in value)
        return "".join(parts)

    @staticmethod
    def _vectorize(text: str, n: int = 2) -> Counter[str]:
        chars = [char for char in text if not char.isspace()]
        grams = [text for text in chars]
        grams.extend("".join(chars[index : index + n]) for index in range(max(0, len(chars) - n + 1)))
        return Counter(grams)

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        dot = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm)
