import tempfile
import unittest
import sqlite3
from pathlib import Path

from fairy_core.environment_service import EnvironmentService


class EnvironmentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = EnvironmentService(
            Path(self.temp_dir.name) / "environment_test.db"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_and_latest_preserve_sensor_values(self) -> None:
        created = self.service.record(
            device_id="esp32-test",
            temperature=24.6,
            humidity=51.2,
            humidity_simulated=True,
            light=380,
            light_raw=1556,
            motion=True,
            firmware_version="fairy-sense-1.0",
        )

        latest = self.service.latest()

        self.assertEqual(created["id"], latest["id"])
        self.assertEqual(latest["device_id"], "esp32-test")
        self.assertTrue(latest["motion"])
        self.assertTrue(latest["humidity_simulated"])
        self.assertEqual(latest["firmware_version"], "fairy-sense-1.0")
        self.assertAlmostEqual(latest["temperature"], 24.6)

    def test_normal_environment_has_no_alerts(self) -> None:
        status = self.service.evaluate(
            {
                "temperature": 24,
                "humidity": 52,
                "light": 420,
                "motion": True,
            }
        )

        self.assertEqual(status["level"], "normal")
        self.assertEqual(status["alerts"], [])

    def test_dark_room_with_motion_creates_lighting_alert(self) -> None:
        status = self.service.evaluate(
            {
                "temperature": 25,
                "humidity": 55,
                "light": 45,
                "motion": True,
            }
        )

        self.assertEqual(status["level"], "warning")
        self.assertIn("有人活动但光线不足", status["alerts"])

    def test_extreme_heat_is_critical(self) -> None:
        status = self.service.evaluate(
            {
                "temperature": 36,
                "humidity": 50,
                "light": 300,
                "motion": False,
            }
        )

        self.assertEqual(status["level"], "critical")
        self.assertIn("温度过高", status["alerts"])

    def test_simulated_humidity_does_not_create_real_alert(self) -> None:
        status = self.service.evaluate(
            {
                "temperature": 24,
                "humidity": 95,
                "humidity_simulated": True,
                "light": 400,
                "motion": True,
            }
        )

        self.assertEqual(status["level"], "normal")
        self.assertNotIn("湿度过高", status["alerts"])
        self.assertIn("模拟展示数据", status["summary"])

    def test_existing_database_is_migrated_for_sensor_metadata(self) -> None:
        database_path = Path(self.temp_dir.name) / "legacy.db"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                CREATE TABLE environment_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL,
                    light REAL NOT NULL,
                    motion INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

        migrated = EnvironmentService(database_path)
        item = migrated.record(
            device_id="esp32-migrated",
            temperature=25,
            humidity=50,
            humidity_simulated=True,
            light=400,
            light_raw=1600,
            motion=False,
            firmware_version="fairy-sense-1.0",
        )

        self.assertTrue(item["humidity_simulated"])
        self.assertEqual(item["light_raw"], 1600)
        self.assertEqual(item["firmware_version"], "fairy-sense-1.0")


if __name__ == "__main__":
    unittest.main()
