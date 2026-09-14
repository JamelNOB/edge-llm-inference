"""
API Integration Tests (FastAPI TestClient & SSE Protocol)
==========================================================
"""
import unittest
import sys
import json
from pathlib import Path

# 将项目根目录添加进 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.server.app import app

class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 使用 TestClient
        cls.client = TestClient(app)

    def test_index_page(self):
        """测试 Web 控制台首页返回"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Edge LLM", response.text)

    def test_health_endpoint(self):
        """测试健康检查探针数据结构与内存采样"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("memory", data)
        self.assertIn("process_ram_rss_mb", data["memory"])
        self.assertIn("system", data)
        self.assertIn("cpu_cores_logical", data["system"])

    def test_chat_stream_sse_protocol(self):
        """测试 SSE 流式推流协议规范 (text/event-stream 及 [DONE] 结束标记)"""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "测试一下"}
            ],
            "temperature": 0.7,
            "max_tokens": 64,
            "stream": True
        }

        response = self.client.post("/v1/chat/stream", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        lines = response.text.strip().split("\r\n\r\n")
        if len(lines) <= 1:
            lines = response.text.strip().split("\n\n")

        has_data_chunk = False
        has_done_signal = False

        for block in lines:
            for line in block.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    content = line[6:].strip()
                    if content == "[DONE]":
                        has_done_signal = True
                    else:
                        try:
                            parsed = json.loads(content)
                            if "choices" in parsed:
                                has_data_chunk = True
                        except Exception:
                            pass

        self.assertTrue(has_data_chunk, "必须接收到合规的 JSON 数据块")
        self.assertTrue(has_done_signal, "流式响应必须以 [DONE] 结束")

if __name__ == "__main__":
    unittest.main()
