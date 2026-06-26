from __future__ import annotations

import os
import random
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from fairy_core.deepseek_client import deepseek_client
from fairy_core.paths import DATA_DIR


class EnvironmentService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("ENVIRONMENT_DB_PATH")
        self.db_path = (
            Path(configured_path)
            if configured_path
            else DATA_DIR / "environment_history.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS environment_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL,
                    humidity_simulated INTEGER NOT NULL DEFAULT 0,
                    light REAL NOT NULL,
                    light_raw REAL,
                    motion INTEGER NOT NULL,
                    firmware_version TEXT,
                    captured_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            existing_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(environment_readings)"
                ).fetchall()
            }
            migrations = {
                "humidity_simulated": (
                    "ALTER TABLE environment_readings "
                    "ADD COLUMN humidity_simulated INTEGER NOT NULL DEFAULT 0"
                ),
                "light_raw": (
                    "ALTER TABLE environment_readings ADD COLUMN light_raw REAL"
                ),
                "firmware_version": (
                    "ALTER TABLE environment_readings ADD COLUMN firmware_version TEXT"
                ),
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)

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
    def evaluate(reading: dict | None) -> dict:
        if reading is None:
            return {
                "level": "offline",
                "label": "等待设备",
                "summary": "尚未收到 ESP32 环境数据。",
                "alerts": [],
                "suggestions": ["连接 ESP32 或生成演示数据后即可开始监测。"],
            }

        temperature = float(reading["temperature"])
        humidity = float(reading["humidity"])
        humidity_simulated = bool(reading.get("humidity_simulated"))
        light = float(reading["light"])
        motion = bool(reading["motion"])
        alerts: list[str] = []
        suggestions: list[str] = []
        critical = False

        if temperature >= 35:
            alerts.append("温度过高")
            suggestions.append("请尽快通风降温，并检查热源。")
            critical = True
        elif temperature > 30:
            alerts.append("温度偏高")
            suggestions.append("建议开启通风或降温设备。")
        elif temperature < 10:
            alerts.append("温度过低")
            suggestions.append("建议检查保温或供暖状态。")
        elif temperature < 18:
            alerts.append("温度偏低")
            suggestions.append("可适当提高室内温度。")

        if not humidity_simulated:
            if humidity > 80:
                alerts.append("湿度过高")
                suggestions.append("建议除湿并检查是否存在积水。")
                critical = True
            elif humidity > 70:
                alerts.append("湿度偏高")
                suggestions.append("建议加强通风。")
            elif humidity < 25:
                alerts.append("湿度过低")
                suggestions.append("建议适当加湿。")
            elif humidity < 35:
                alerts.append("空气偏干")
                suggestions.append("可适当补充室内湿度。")

        if motion and light < 120:
            alerts.append("有人活动但光线不足")
            suggestions.append("建议开启照明，改善学习与操作环境。")
        elif not motion and light > 700:
            alerts.append("无人但照明较强")
            suggestions.append("可关闭不必要的照明以节约能源。")

        if not alerts:
            summary = "当前温度与相对光照处于适宜范围。"
            if humidity_simulated:
                summary += "湿度为模拟展示数据，不参与真实告警。"
            return {
                "level": "normal",
                "label": "环境舒适",
                "summary": summary,
                "alerts": [],
                "suggestions": ["继续保持当前环境状态。"],
            }

        level = "critical" if critical else "warning"
        return {
            "level": level,
            "label": "需要处理" if critical else "请留意",
            "summary": "；".join(alerts) + "。",
            "alerts": alerts,
            "suggestions": suggestions,
        }

    def record(
        self,
        *,
        device_id: str,
        temperature: float,
        humidity: float,
        light: float,
        motion: bool,
        humidity_simulated: bool = False,
        light_raw: float | None = None,
        firmware_version: str | None = None,
        captured_at: str | None = None,
    ) -> dict:
        clean_device_id = device_id.strip() or "esp32-classroom-01"
        if not -40 <= temperature <= 85:
            raise ValueError("temperature 必须在 -40 到 85 摄氏度之间")
        if not 0 <= humidity <= 100:
            raise ValueError("humidity 必须在 0 到 100 之间")
        if light < 0:
            raise ValueError("light 不能小于 0")
        if light_raw is not None and light_raw < 0:
            raise ValueError("light_raw 不能小于 0")

        captured = self._timestamp(captured_at)
        created = self._timestamp()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO environment_readings (
                    device_id, temperature, humidity, humidity_simulated,
                    light, light_raw, motion, firmware_version,
                    captured_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_device_id,
                    round(float(temperature), 2),
                    round(float(humidity), 2),
                    int(bool(humidity_simulated)),
                    round(float(light), 2),
                    round(float(light_raw), 2) if light_raw is not None else None,
                    int(bool(motion)),
                    firmware_version.strip() if firmware_version else None,
                    captured,
                    created,
                ),
            )
            reading_id = cursor.lastrowid
        return self.get(reading_id)

    @staticmethod
    def _serialize(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        item["motion"] = bool(item["motion"])
        item["humidity_simulated"] = bool(item.get("humidity_simulated"))
        return item

    def get(self, reading_id: int) -> dict:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM environment_readings WHERE id = ?",
                (reading_id,),
            ).fetchone()
        item = self._serialize(row)
        if item is None:
            raise LookupError(f"未找到环境记录 {reading_id}")
        return item

    def latest(self) -> dict | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM environment_readings ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return self._serialize(row)

    def history(self, limit: int = 60) -> list[dict]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM environment_readings ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._serialize(row) for row in rows]

    def snapshot(self, history_limit: int = 60) -> dict:
        reading = self.latest()
        return {
            "reading": reading,
            "status": self.evaluate(reading),
            "history": self.history(history_limit) if reading else [],
        }

    def simulate(self, device_id: str = "esp32-demo") -> dict:
        previous = self.latest()
        base_temperature = float(previous["temperature"]) if previous else 24.5
        base_humidity = float(previous["humidity"]) if previous else 52.0
        base_light = float(previous["light"]) if previous else 420.0
        return self.record(
            device_id=device_id,
            temperature=max(16, min(36, base_temperature + random.uniform(-1.2, 1.2))),
            humidity=max(25, min(85, base_humidity + random.uniform(-3.5, 3.5))),
            humidity_simulated=True,
            light=max(20, min(950, base_light + random.uniform(-120, 120))),
            light_raw=random.uniform(0, 4095),
            motion=random.random() > 0.38,
            firmware_version="fairy-demo",
        )

    def clear_history(self) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM environment_readings")
        return max(cursor.rowcount, 0)

    def answer(self, query: str, model: str | None = None) -> str:
        latest = self.latest()
        if latest is None:
            return (
                "我还没有收到 ESP32 的环境数据。连接设备后，我可以分析温度、湿度、"
                "光照和人体活动状态；目前也可以先在环境面板生成演示数据。"
            )

        status = self.evaluate(latest)
        humidity_note = "（模拟数据）" if latest.get("humidity_simulated") else ""
        context = (
            f"设备：{latest['device_id']}；温度：{latest['temperature']:.1f}℃；"
            f"湿度：{latest['humidity']:.1f}%{humidity_note}；"
            f"相对光照值：{latest['light']:.0f}/1000；"
            f"人体活动：{'检测到' if latest['motion'] else '未检测到'}；"
            f"综合状态：{status['label']}；判断：{status['summary']}；"
            f"建议：{'；'.join(status['suggestions'])}；"
            f"采集时间：{latest['captured_at']}。"
        )
        if not deepseek_client.is_enabled():
            return context

        try:
            return deepseek_client.chat(
                model=model,
                temperature=0.3,
                max_tokens=420,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 Fairy 的 Environment Guardian 环境智能体。"
                            "只能依据给出的 ESP32 传感器上下文回答。"
                            "固定阈值负责告警，你负责解释、总结和给出非危险性的建议。"
                            "不要虚构未提供的传感器、设备动作或历史数据。"
                        ),
                    },
                    {"role": "system", "content": context},
                    {"role": "user", "content": query},
                ],
            )
        except Exception:
            return context


environment_service = EnvironmentService()
