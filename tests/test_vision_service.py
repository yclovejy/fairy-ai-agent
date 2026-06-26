import tempfile
import unittest
from pathlib import Path

from fairy_core.vision_service import VisionService


class VisionServiceTest(unittest.TestCase):
    def test_records_known_object_without_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = VisionService(Path(temp_dir) / "vision.db")
            item = service.record(
                label="手机",
                confidence=0.95,
                source="k230-test",
                device_id="K230-LC-3",
            )

            self.assertEqual(item["label"], "手机")
            self.assertEqual(item["confidence"], 0.95)
            self.assertIn("智能手机", item["description"])
            self.assertEqual(service.latest()["id"], item["id"])
            self.assertEqual(len(service.history()), 1)

    def test_rejects_invalid_confidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = VisionService(Path(temp_dir) / "vision.db")
            with self.assertRaises(ValueError):
                service.record(label="水杯", confidence=1.5)


if __name__ == "__main__":
    unittest.main()
