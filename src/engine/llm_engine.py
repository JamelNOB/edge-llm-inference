"""
Inference Engine Abstraction Layer (llama.cpp C++ Runtime & Adaptive Mock)
==========================================================================
[工程边界定位]
- 第三方底座：llama.cpp (提供高效 C++ GGUF 模型反量化与 SIMD CPU 矩阵乘法加速)
- 自研主理人职责：
  1. 硬件亲和性线程拓扑分配 (Thread Affinity & Auto Thread Allocation)；
  2. 初始化阶段的 System Prompt KV Cache 预热编排；
  3. 自回归流式 Generator 封装与 SSE 协议适配；
  4. 无权重/干净环境自适应 Mock 模式，确保离线回归评测与 CI/CD 100% 可跑。
"""
import os
import sys
import time
from pathlib import Path
from typing import Generator, Dict, Any, Optional

from src.core.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_CPU_THREADS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_GPU_LAYERS,
    DEFAULT_SYSTEM_PROMPT,
    FORCE_MOCK,
    ENABLE_MOCK_FALLBACK
)
from src.engine.kv_cache import warmup_kv_cache

class MockLlamaEngine:
    """
    自适应 Mock 推理模拟器。
    用于在 CI 环境、面试 30 秒快速验证或未下载 1GB 权重的纯净环境下保证调用链路完整可用。
    """
    def __init__(self, model_name: str = "qwen2.5-1.5b-instruct-q4_k_m-mock"):
        self.model_name = model_name
        self.is_mock = True

    def tokenize(self, text_bytes: bytes):
        return [i for i in range(len(text_bytes) // 4 + 1)]

    def eval(self, tokens):
        # 模拟前向计算
        time.sleep(0.01)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
        stop: Optional[list] = None
    ):
        mock_response_words = [
            "【Mock模式】", "端侧", "大模型", "已通过", "Q4_K_M", "量化", "加载，",
            "在普通", "CPU", "上有效", "克服了", "内存带宽", "瓶颈，",
            "实测", "首字延迟", "(TTFT)", "降至", "100ms", "以内，",
            "流式", "吞吐", "稳定在", "19~23", "tokens/s。",
            "服务", "运行", "正常！"
        ]

        if not stream:
            return {
                "choices": [{
                    "text": "".join(mock_response_words),
                    "finish_reason": "stop"
                }]
            }

        def mock_generator():
            for i, word in enumerate(mock_response_words):
                # 模拟 CPU 推理流式耗时 (~45ms/token -> ~22 tokens/s)
                time.sleep(0.045)
                is_last = (i == len(mock_response_words) - 1)
                yield {
                    "choices": [{
                        "text": word,
                        "finish_reason": "stop" if is_last else None
                    }]
                }

        return mock_generator()


class EdgeLLMEngine:
    """
    生产级端侧大模型推理引擎控制器
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        cpu_threads: Optional[int] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        auto_warmup: bool = True
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.cpu_threads = cpu_threads or DEFAULT_CPU_THREADS
        self.context_window = context_window
        self.gpu_layers = gpu_layers
        self.auto_warmup = auto_warmup
        
        self.instance: Any = None
        self.is_mock: bool = False
        self.warmup_cost_ms: float = 0.0
        self.warmup_tokens: int = 0

    def initialize(self) -> bool:
        """加载底层模型并执行 KV Cache 预热"""
        if FORCE_MOCK:
            print("[*] FORCE_MOCK is set. Starting in Simulation Mode...")
            self.instance = MockLlamaEngine(model_name=DEFAULT_MODEL_FILENAME)
            self.is_mock = True
            if self.auto_warmup:
                self.warmup_cost_ms, self.warmup_tokens = warmup_kv_cache(
                    self.instance, DEFAULT_SYSTEM_PROMPT, verbose=True
                )
            return True

        if not self.model_path.exists():
            print(f"[!] Warning: GGUF Model weights not found at {self.model_path}")
            if ENABLE_MOCK_FALLBACK:
                print("[+] Auto-fallback: Initializing Mock Inference Engine for verification...")
                self.instance = MockLlamaEngine(model_name=DEFAULT_MODEL_FILENAME)
                self.is_mock = True
                if self.auto_warmup:
                    self.warmup_cost_ms, self.warmup_tokens = warmup_kv_cache(
                        self.instance, DEFAULT_SYSTEM_PROMPT, verbose=True
                    )
                return True
            return False

        try:
            from llama_cpp import Llama
            print(f"[+] Loading GGUF Model from {self.model_path}...")
            print(f"[+] Hardware Policy: CPU Threads = {self.cpu_threads}, GPU Layers = {self.gpu_layers}, Context = {self.context_window}")

            self.instance = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_window,
                n_threads=self.cpu_threads,
                n_gpu_layers=self.gpu_layers,
                verbose=False
            )
            self.is_mock = False
            print("[+] Model loaded successfully into memory.")

            # 执行简历核心技术点：KV Cache 预热
            if self.auto_warmup:
                self.warmup_cost_ms, self.warmup_tokens = warmup_kv_cache(
                    self.instance, DEFAULT_SYSTEM_PROMPT, verbose=True
                )

            return True
        except Exception as e:
            print(f"[x] Failed to load llama-cpp-python runtime: {e}")
            if ENABLE_MOCK_FALLBACK:
                print("[+] Fallback: Switching to Mock Engine for non-blocking execution...")
                self.instance = MockLlamaEngine(model_name=DEFAULT_MODEL_FILENAME)
                self.is_mock = True
                return True
            return False

    def stream_inference(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Generator[Dict[str, Any], None, None]:
        """流式生成器接口"""
        if self.instance is None:
            raise RuntimeError("LLM Engine is not initialized.")

        stream_iter = self.instance(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        for chunk in stream_iter:
            token_text = chunk["choices"][0]["text"]
            finish_reason = chunk["choices"][0].get("finish_reason")
            yield {
                "token": token_text,
                "finish_reason": finish_reason
            }

    def close(self):
        """释放模型显存/内存资源"""
        if self.instance and not self.is_mock:
            del self.instance
            self.instance = None
            print("[+] LLM Engine unloaded.")


# 单例实例管理
_global_engine: Optional[EdgeLLMEngine] = None

def get_engine() -> EdgeLLMEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = EdgeLLMEngine()
        _global_engine.initialize()
    return _global_engine
