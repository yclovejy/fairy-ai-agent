import unittest
import ast
from pathlib import Path

import fairy_core.agent_v5 as agent_v5
from fairy_core.deepseek_client import deepseek_client


ROOT = Path(__file__).resolve().parents[1]


class FairyRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_chat = deepseek_client.chat
        self.original_enabled = deepseek_client.is_enabled

    def tearDown(self) -> None:
        deepseek_client.chat = self.original_chat
        deepseek_client.is_enabled = self.original_enabled

    def test_main_fairy_routes_place_question_to_local_nlp(self) -> None:
        deepseek_client.is_enabled = lambda: True
        deepseek_client.chat = lambda **_: self.fail("local place answer should not call DeepSeek")

        answer = agent_v5.agent_answer(
            "杭州属于哪里",
            [],
            agent_id="auto",
            model="deepseek-v4-pro",
        )

        self.assertIn("浙江省 -> 杭州市", answer)
        self.assertIn("西湖", answer)

    def test_identity_answer_never_calls_model_or_reveals_it(self) -> None:
        deepseek_client.is_enabled = lambda: True
        deepseek_client.chat = lambda **_: self.fail("identity answer should not call DeepSeek")

        for query in ("你是谁", "你是谁创造的", "Fairy 的主人是谁", "它是谁"):
            answer = agent_v5.agent_answer(
                query,
                [],
                agent_id="auto",
                model="deepseek-v4-pro",
            )
            self.assertIn("yongcheng", answer)
            self.assertIn("绝区零", answer)
            self.assertNotIn("DeepSeek", answer)
            self.assertNotIn("模型", answer)

    def test_auto_router_uses_same_selected_model_for_every_llm_call(self) -> None:
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][-1]["content"]
            return "chat" if "判断用户意图" in prompt else "OK"

        deepseek_client.is_enabled = lambda: True
        deepseek_client.chat = fake_chat

        answer = agent_v5.agent_answer(
            "陪我随便聊聊",
            [],
            agent_id="auto",
            model="deepseek-v4-pro",
        )

        self.assertEqual(answer, "OK")
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(call["model"] == "deepseek-v4-pro" for call in calls))

    def test_environment_agent_is_available_and_routes_explicit_queries(self) -> None:
        profile_ids = {profile["id"] for profile in agent_v5.get_agent_profiles()}
        self.assertIn("environment", profile_ids)
        self.assertEqual(
            agent_v5.resolve_agent_intent(
                "environment",
                "现在教室环境怎么样",
                "deepseek-v4-flash",
            ),
            "environment",
        )

    def test_every_deepseek_call_site_passes_selected_model(self) -> None:
        for relative_path in ("fairy_core/agent_v5.py", "fairy_core/travel_agent.py"):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            call_sites = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "chat" or not isinstance(node.func.value, ast.Name):
                    continue
                if node.func.value.id != "deepseek_client":
                    continue
                call_sites.append(node)
                self.assertIn(
                    "model",
                    {keyword.arg for keyword in node.keywords},
                    f"{relative_path}:{node.lineno} 未传递全局模型",
                )
            self.assertTrue(call_sites, f"{relative_path} 没有找到 DeepSeek 调用点")


if __name__ == "__main__":
    unittest.main()
