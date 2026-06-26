from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from fairy_core.deepseek_client import deepseek_client
from fairy_core.paths import DATA_DIR


OBJECT_GUIDES = {
    "水杯": "这是一个用于盛放饮品的水杯。使用后及时清洁，并注意热饮温度，避免烫伤。",
    "cup": "这是一个用于盛放饮品的水杯。使用后及时清洁，并注意热饮温度，避免烫伤。",
    "手机": "这是智能手机，可用于通信、学习、导航和信息查询。公共场合请注意保管个人设备与隐私。",
    "cell phone": "这是智能手机，可用于通信、学习、导航和信息查询。公共场合请注意保管个人设备与隐私。",
    "phone": "这是智能手机，可用于通信、学习、导航和信息查询。公共场合请注意保管个人设备与隐私。",
    "键盘": "这是计算机键盘，是文字输入和快捷操作的主要设备。保持键面清洁有助于改善使用体验。",
    "keyboard": "这是计算机键盘，是文字输入和快捷操作的主要设备。保持键面清洁有助于改善使用体验。",
    "鼠标": "这是计算机鼠标，用于控制指针和执行交互操作。调整合适的握姿可以减轻长时间使用带来的疲劳。",
    "mouse": "这是计算机鼠标，用于控制指针和执行交互操作。调整合适的握姿可以减轻长时间使用带来的疲劳。",
    "学生证": "这是学生身份凭证，通常用于身份核验、校园门禁和校内服务。请妥善保管，避免泄露个人信息。",
    "student id": "这是学生身份凭证，通常用于身份核验、校园门禁和校内服务。请妥善保管，避免泄露个人信息。",
    "student card": "这是学生身份凭证，通常用于身份核验、校园门禁和校内服务。请妥善保管，避免泄露个人信息。",
}


class VisionService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("VISION_DB_PATH")
        self.db_path = Path(configured_path) if configured_path else DATA_DIR / "vision_history.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pending_labels: set[str] = set()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    description_status TEXT NOT NULL DEFAULT 'ready',
                    source TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    frame_id TEXT,
                    bbox TEXT,
                    captured_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_descriptions (
                    label_key TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                connection,
                "vision_detections",
                "description_status",
                "TEXT NOT NULL DEFAULT 'ready'",
            )
            self._ensure_column(
                connection,
                "vision_detections",
                "updated_at",
                "TEXT",
            )
            connection.execute(
                """
                UPDATE vision_detections
                SET updated_at = created_at
                WHERE updated_at IS NULL OR updated_at = ''
                """
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    @staticmethod
    def _timestamp(value: str | None = None) -> str:
        if value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc).isoformat()
            except ValueError:
                pass
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _label_key(label: str) -> str:
        return " ".join(label.strip().lower().split())

    def _cached_description(self, label_key: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT description FROM vision_descriptions WHERE label_key = ?",
                (label_key,),
            ).fetchone()
        return str(row["description"]) if row else None

    def _store_description(self, label_key: str, description: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vision_descriptions (label_key, description, created_at)
                VALUES (?, ?, ?)
                """,
                (label_key, description, self._timestamp()),
            )

    def _known_or_cached_description(self, label: str) -> str | None:
        label_key = self._label_key(label)
        if label_key in OBJECT_GUIDES:
            return OBJECT_GUIDES[label_key]

        return self._cached_description(label_key)

    def _fallback_description(self, label: str) -> str:
        return f"Fairy 识别到“{label}”。这是视觉模型当前给出的类别，请结合现场环境确认识别结果。"

    def _generate_description(self, label: str, model: str | None = None) -> str:
        fallback = f"Fairy 识别到“{label}”。这是视觉模型当前给出的类别，请结合现场环境确认识别结果。"
        if not deepseek_client.is_enabled():
            return fallback

        try:
            description = deepseek_client.chat(
                model=model,
                temperature=0.35,
                max_tokens=160,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是AI视觉导览助手。请为识别到的物品生成一段中文讲解，"
                            "包含它是什么、常见用途和一条安全或使用提示。"
                            "控制在80字以内，不透露底层模型，不使用Markdown。"
                        ),
                    },
                    {"role": "user", "content": label},
                ],
            ).strip()
        except Exception:
            return fallback

        if not description:
            return fallback
        self._store_description(label_key, description)
        return description

    def describe(self, label: str, model: str | None = None) -> str:
        known = self._known_or_cached_description(label)
        if known:
            return known
        return self._generate_description(label, model=model)

    def _initial_description_state(self, label: str) -> tuple[str, str, bool]:
        known = self._known_or_cached_description(label)
        if known:
            return known, "ready", False

        if deepseek_client.is_enabled():
            return (
                f"正在检索“{label}”的物品信息，Fairy 会在完成后自动补全讲解。",
                "pending",
                True,
            )

        return self._fallback_description(label), "fallback", False

    def _update_generated_description(
        self,
        *,
        detection_id: int,
        label: str,
        label_key: str,
        description: str,
    ) -> None:
        now = self._timestamp()
        self._store_description(label_key, description)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE vision_detections
                SET description = ?,
                    description_status = 'ready',
                    updated_at = ?
                WHERE id = ? OR lower(label) = ?
                """,
                (description, now, detection_id, label_key),
            )

    def _generate_description_background(
        self,
        *,
        detection_id: int,
        label: str,
        label_key: str,
        model: str | None,
    ) -> None:
        try:
            description = self._generate_description(label, model=model)
            self._update_generated_description(
                detection_id=detection_id,
                label=label,
                label_key=label_key,
                description=description,
            )
        finally:
            with self._lock:
                self._pending_labels.discard(label_key)

    def _schedule_description_generation(
        self,
        *,
        detection_id: int,
        label: str,
        model: str | None,
    ) -> None:
        label_key = self._label_key(label)
        with self._lock:
            if label_key in self._pending_labels:
                return
            self._pending_labels.add(label_key)

        thread = threading.Thread(
            target=self._generate_description_background,
            kwargs={
                "detection_id": detection_id,
                "label": label,
                "label_key": label_key,
                "model": model,
            },
            daemon=True,
        )
        thread.start()

    def record(
        self,
        *,
        label: str,
        confidence: float,
        source: str = "k230",
        device_id: str = "lushan-pi-k230",
        frame_id: str | None = None,
        bbox: list[float] | None = None,
        captured_at: str | None = None,
        model: str | None = None,
    ) -> dict:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("label 不能为空")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence 必须在 0 到 1 之间")

        import json

        captured = self._timestamp(captured_at)
        created = self._timestamp()
        description, description_status, should_generate = self._initial_description_state(clean_label)
        bbox_json = json.dumps(bbox, ensure_ascii=False) if bbox else None

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO vision_detections (
                    label, confidence, description, description_status, source, device_id,
                    frame_id, bbox, captured_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_label,
                    confidence,
                    description,
                    description_status,
                    source.strip() or "k230",
                    device_id.strip() or "lushan-pi-k230",
                    frame_id,
                    bbox_json,
                    captured,
                    created,
                    created,
                ),
            )
            detection_id = cursor.lastrowid

        if should_generate:
            self._schedule_description_generation(
                detection_id=detection_id,
                label=clean_label,
                model=model,
            )

        return self.get(detection_id)

    @staticmethod
    def _serialize(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        import json

        item = dict(row)
        item["bbox"] = json.loads(item["bbox"]) if item.get("bbox") else None
        item["description_status"] = item.get("description_status") or "ready"
        item["updated_at"] = item.get("updated_at") or item.get("created_at")
        return item

    def get(self, detection_id: int) -> dict:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vision_detections WHERE id = ?",
                (detection_id,),
            ).fetchone()
        item = self._serialize(row)
        if item is None:
            raise LookupError(f"未找到视觉记录 {detection_id}")
        return item

    def latest(self) -> dict | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vision_detections ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return self._serialize(row)

    def history(self, limit: int = 50) -> list[dict]:
        safe_limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vision_detections ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._serialize(row) for row in rows]

    def clear_history(self) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM vision_detections")
        return max(cursor.rowcount, 0)

    def answer(self, query: str, model: str | None = None) -> str:
        latest = self.latest()
        if latest is None:
            return "我还没有收到庐山派的视觉识别结果。摄像头开始推送后，我就能告诉你现场看到了什么。"

        context = (
            f"最近识别物品：{latest['label']}；"
            f"置信度：{latest['confidence']:.1%}；"
            f"讲解：{latest['description']}；"
            f"讲解状态：{latest.get('description_status', 'ready')}；"
            f"识别时间：{latest['captured_at']}。"
        )
        if latest.get("description_status") == "pending":
            return context
        if not deepseek_client.is_enabled():
            return context

        try:
            return deepseek_client.chat(
                model=model,
                temperature=0.45,
                max_tokens=400,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是Fairy的视觉导览智能体。只能根据给出的视觉上下文回答，"
                            "不确定时明确说明，不虚构摄像头没有识别到的内容。"
                        ),
                    },
                    {"role": "system", "content": context},
                    {"role": "user", "content": query},
                ],
            )
        except Exception:
            return context


vision_service = VisionService()
