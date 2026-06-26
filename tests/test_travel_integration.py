import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fairy_core.deepseek_client import deepseek_client
from fairy_core.travel_agent import FairyTravelAgent


ROOT = Path(__file__).resolve().parents[1]


class FairyTravelAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        shutil.copytree(ROOT / "data" / "travel", self.root / "data" / "travel")
        knowledge_path = self.root / "data" / "travel" / "data" / "tourism_knowledge.json"
        knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
        knowledge.pop("4403", None)
        knowledge_path.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
        self.agent = FairyTravelAgent(self.root)
        self.original_chat = deepseek_client.chat
        self.original_enabled = deepseek_client.is_enabled

    def tearDown(self) -> None:
        deepseek_client.chat = self.original_chat
        deepseek_client.is_enabled = self.original_enabled
        self.temp_dir.cleanup()

    def test_seeded_place_returns_without_model_call(self) -> None:
        deepseek_client.is_enabled = lambda: True
        deepseek_client.chat = lambda **_: self.fail("seeded place should not call DeepSeek")

        answer = self.agent.answer("杭州旅游", "deepseek-v4-flash")

        self.assertIn("西湖", answer)
        self.assertIn("浙江省 -> 杭州市", answer)

    def test_place_question_is_detected_for_main_fairy(self) -> None:
        self.assertTrue(self.agent.matches_query("杭州属于哪里"))
        self.assertTrue(self.agent.matches_query("成都"))
        self.assertFalse(self.agent.matches_query("你是谁"))

    def test_missing_tourism_uses_selected_model_and_is_cached(self) -> None:
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return json.dumps(
                {
                    "intro": "测试简介",
                    "attractions": ["测试景点"],
                    "foods": ["测试美食"],
                    "travel_tips": ["测试建议"],
                },
                ensure_ascii=False,
            )

        deepseek_client.is_enabled = lambda: True
        deepseek_client.chat = fake_chat

        first = self.agent.answer("深圳旅游", "deepseek-v4-pro")
        second = self.agent.answer("深圳旅游", "deepseek-v4-flash")

        self.assertIn("测试景点", first)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
