from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlaceRecord:
    code: str
    name: str
    level: str
    province_code: str | None = None
    province_name: str | None = None
    city_code: str | None = None
    city_name: str | None = None
    area_code: str | None = None
    area_name: str | None = None
    matched_text: str | None = None
    matched_alias: str | None = None
    score: float = 0.0

    @property
    def path(self) -> list[str]:
        names = [self.province_name, self.city_name, self.area_name]
        return [name for name in names if name]

    def with_match(self, matched_text: str, matched_alias: str, score: float) -> "PlaceRecord":
        return PlaceRecord(
            code=self.code,
            name=self.name,
            level=self.level,
            province_code=self.province_code,
            province_name=self.province_name,
            city_code=self.city_code,
            city_name=self.city_name,
            area_code=self.area_code,
            area_name=self.area_name,
            matched_text=matched_text,
            matched_alias=matched_alias,
            score=score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "level": self.level,
            "province": self.province_name,
            "city": self.city_name,
            "area": self.area_name,
            "path": self.path,
            "matched_text": self.matched_text,
            "matched_alias": self.matched_alias,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class TourismInfo:
    attractions: list[str]
    foods: list[str]
    intro: str
    source: str
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "attractions": self.attractions,
            "foods": self.foods,
            "intro": self.intro,
            "source": self.source,
        }
