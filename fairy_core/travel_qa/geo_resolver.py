from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .models import PlaceRecord


ADMIN_SUFFIXES = (
    "特别行政区",
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "自治区",
    "自治州",
    "自治县",
    "地区",
    "林区",
    "新区",
    "省",
    "市",
    "县",
    "区",
    "盟",
    "州",
)

NOISE_WORDS = (
    "我想去",
    "想去",
    "我要去",
    "去",
    "旅游",
    "旅行",
    "玩",
    "攻略",
    "介绍",
    "一下",
)


class GeoResolver:
    """Resolve Chinese place names into province/city/area hierarchy."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.provinces = self._load_json("provinces.json")
        self.cities = self._load_json("cities.json")
        self.areas = self._load_json("areas.json")

        self.provinces_by_code = {item["code"]: item for item in self.provinces}
        self.cities_by_code = {item["code"]: item for item in self.cities}
        self.alias_index: dict[str, list[PlaceRecord]] = defaultdict(list)
        self._build_index()

    def resolve(self, query: str, limit: int = 20) -> list[PlaceRecord]:
        query = self._normalize_query(query)
        if not query:
            return []

        scored: dict[tuple[str, str], PlaceRecord] = {}
        for alias, records in self.alias_index.items():
            if not alias or alias not in query:
                continue

            for record in records:
                exact_bonus = 2.0 if query == alias or query == record.name else 0.0
                level_bonus = {"area": 0.3, "city": 0.2, "province": 0.1}[record.level]
                score = len(alias) + exact_bonus + level_bonus
                key = (record.level, record.code)
                current = scored.get(key)
                if current is None or score > current.score:
                    scored[key] = record.with_match(alias, alias, score)

        results = [self._add_context_score(item, query) for item in scored.values()]
        if not results:
            return []

        max_score = max(item.score for item in results)
        narrowed = [item for item in results if item.score == max_score]
        narrowed.sort(key=lambda item: (-item.score, item.province_name or "", item.city_name or "", item.name))
        return narrowed[:limit]

    def _load_json(self, filename: str) -> list[dict[str, str]]:
        with (self.data_dir / filename).open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_index(self) -> None:
        for province in self.provinces:
            record = PlaceRecord(
                code=province["code"],
                name=province["name"],
                level="province",
                province_code=province["code"],
                province_name=province["name"],
            )
            self._add_aliases(record)

        for city in self.cities:
            province = self.provinces_by_code[city["provinceCode"]]
            record = PlaceRecord(
                code=city["code"],
                name=city["name"],
                level="city",
                province_code=province["code"],
                province_name=province["name"],
                city_code=city["code"],
                city_name=city["name"],
            )
            self._add_aliases(record)

        for area in self.areas:
            province = self.provinces_by_code[area["provinceCode"]]
            city = self.cities_by_code[area["cityCode"]]
            record = PlaceRecord(
                code=area["code"],
                name=area["name"],
                level="area",
                province_code=province["code"],
                province_name=province["name"],
                city_code=city["code"],
                city_name=city["name"],
                area_code=area["code"],
                area_name=area["name"],
            )
            self._add_aliases(record)

    def _add_aliases(self, record: PlaceRecord) -> None:
        aliases = {record.name, self._strip_admin_suffix(record.name)}
        aliases = {alias for alias in aliases if len(alias) >= 2 and alias != "市辖"}
        for alias in aliases:
            self.alias_index[alias].append(record)

    def _add_context_score(self, record: PlaceRecord, query: str) -> PlaceRecord:
        score = record.score
        for name, boost in (
            (record.province_name, 4.0),
            (record.city_name, 6.0),
            (record.area_name, 2.0),
        ):
            if name and self._name_in_query(name, query):
                score += boost
        return record.with_match(record.matched_text or "", record.matched_alias or "", score)

    def _name_in_query(self, name: str, query: str) -> bool:
        return name in query or self._strip_admin_suffix(name) in query

    @staticmethod
    def _strip_admin_suffix(name: str) -> str:
        for suffix in ADMIN_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _normalize_query(query: str) -> str:
        query = re.sub(r"\s+", "", query)
        query = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]{}<>《》]", "", query)
        for word in NOISE_WORDS:
            query = query.replace(word, "")
        return query
