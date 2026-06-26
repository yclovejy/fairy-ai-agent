from __future__ import annotations

from pathlib import Path
from typing import Any

from .geo_resolver import GeoResolver
from .tourism import TourismRepository


class TravelQAService:
    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root)
        self.resolver = GeoResolver(root)
        self.tourism = TourismRepository(root / "data" / "tourism_seed.json")

    def answer(self, query: str) -> dict[str, Any]:
        candidates = self.resolver.resolve(query)
        if not candidates:
            return {
                "query": query,
                "recognized_place": None,
                "candidate_count": 0,
                "candidates": [],
                "message": "没有识别到中国行政区划地名，请换一个更明确的地名试试。",
            }

        return {
            "query": query,
            "recognized_place": candidates[0].matched_text,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    **candidate.to_dict(),
                    "tourism": self.tourism.get_info(candidate, query).to_dict(),
                }
                for candidate in candidates
            ],
            "message": "存在重名地名，请从候选结果中选择目标地区。"
            if len(candidates) > 1
            else "识别成功。",
        }
