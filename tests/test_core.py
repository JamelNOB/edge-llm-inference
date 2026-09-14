"""
Unit Tests for Core Architecture & KV Cache Warmup
===================================================
Verify:
1. Configuration loading and hardware thread policies
2. Pydantic request/response schema validation
3. warmup_kv_cache explicit execution & latency profiling
4. EdgeLLMEngine stream generator functionality
"""
import unittest
import sys
from pathlib import Path

# 将项目根目录添加进 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import (
    DEFAULT_MODEL_FILENAME,
    DEFAULT_CPU_THREADS,
    DEFAULT_SYSTEM_PROMPT
)
from src.core.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    HealthResponse,
    MemoryMetrics,
    SystemMetrics
)
from src.engine.kv_cache import warmup_kv_cache
from src.engine.llm_engine import MockLlamaEngine, EdgeLLMEngine

class TestCoreModules(unittest.TestCase):

    def test_config_sanity(self):
        """测试核心配置与线程分配策略"""
        self.assertTrue(DEFAULT_CPU_THREADS >= 1)
        self.assertIn("qwen2.5-1.5b", DEFAULT_MODEL_FILENAME.lower())
        self.assertTrue(len(DEFAULT_SYSTEM_PROMPT) > 10)

    def test_schemas_validation(self):
        """测试请求与指标契约 Schema 校验"""
        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

        req = ChatCompletionRequest(
            messages=[msg],
            temperature=0.8,
            max_tokens=256,
            stream=True
        )
        self.assertEqual(req.max_tokens, 256)
        self.assertTrue(req.stream)

        health = HealthResponse(
            status="healthy",
            model_name="test-model.gguf",
            model_loaded=True,
            is_mock=False,
            uptime_seconds=12.5,
            server_pid=1234,
            memory=MemoryMetrics(process_ram_rss_mb=1720.0, system_ram_percent=45.0),
            system=SystemMetrics(cpu_cores_logical=8, cpu_percent=12.0)
        )
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.memory.process_ram_rss_mb, 1720.0)

    def test_warmup_kv_cache_mechanism(self):
        """
        [简历技术点核心测试] 验证 KV Cache 预热机制执行
        确保 warmup_kv_cache 能被显式调用，完成分词并前向评估 System Prompt
        """
        mock_engine = MockLlamaEngine()
        cost_ms, tokens_count = warmup_kv_cache(
            mock_engine,
            DEFAULT_SYSTEM_PROMPT,
            verbose=False
        )

        self.assertTrue(cost_ms >= 0.0, "预热耗时必须有效计算")
        self.assertTrue(tokens_count > 0, "预热 Token 数量必须大于 0")

    def test_engine_streaming_generation(self):
        """验证推理引擎流式生成器规范"""
        engine = EdgeLLMEngine(auto_warmup=True)
        # 强制在 Mock 模式下进行单元测试验证
        engine.instance = MockLlamaEngine()
        engine.is_mock = True

        chunks = list(engine.stream_inference("你好", max_tokens=32))
        self.assertTrue(len(chunks) > 0, "生成器必须吐出 Token 流")
        
        full_text = "".join([c["token"] for c in chunks])
        self.assertTrue(len(full_text) > 0)
        self.assertEqual(chunks[-1]["finish_reason"], "stop")

if __name__ == "__main__":
    unittest.main()
