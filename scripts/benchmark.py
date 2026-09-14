"""
Automated Performance & Resource Benchmark Suite
=================================================
Measures:
- TTFT (Time To First Token in ms)
- Decoding Throughput (tokens/second)
- Peak RSS RAM Utilization (MB)
- Peak CPU Core Utilization (%)

Supports:
1. Live HTTP API Mode: benchmarks a running FastAPI server (http://127.0.0.1:8000)
2. In-Process Standalone Mode: benchmarks the engine directly (ideal for CI/CD and offline tests)
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 将项目根目录添加进 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.monitor import ServerResourceMonitor
from src.core.config import SERVER_PORT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BENCHMARK_PROMPTS = [
    {
        "category": "Short QA",
        "prompt": "用一句话解释什么是边缘计算？"
    },
    {
        "category": "Tech Concept",
        "prompt": "请详细解释 Transformer 架构中 Self-Attention 机制的核心思想与时间复杂度。"
    },
    {
        "category": "Code Gen",
        "prompt": "用 Python 实现一个快速排序算法函数并给出简要测试用例。"
    },
    {
        "category": "Analysis",
        "prompt": "对比端侧大模型部署 (Edge LLM) 与云端 API 调用在成本、隐私、延迟与吞吐方面的优劣势。"
    }
]

def run_in_process_benchmark(prompts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """单机直接内存推理性能基准测试 (无需预先启动 HTTP 网关)"""
    from src.engine.llm_engine import get_engine

    print("\n[*] Initializing In-Process LLM Engine...")
    engine = get_engine()
    print(f"[*] Engine Ready. Mock Mode = {engine.is_mock}")
    print(f"[*] KV Cache Warmup Cost = {engine.warmup_cost_ms:.2f}ms")

    results = []

    for item in prompts:
        category = item["category"]
        prompt = item["prompt"]
        print(f"\n--- [Testing: {category}] ---")
        print(f"Prompt: {prompt[:40]}...")

        monitor = ServerResourceMonitor(target_pid=os.getpid(), interval=0.03)
        monitor.start()

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_text = ""

        generator = engine.stream_inference(
            prompt=prompt,
            max_tokens=128,
            temperature=0.7,
            top_p=0.9
        )

        for chunk in generator:
            token = chunk["token"]
            if token:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                full_text += token

        end_time = time.perf_counter()
        monitor.stop()
        monitor.join(timeout=1.0)

        total_time = end_time - start_time
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
        gen_time = end_time - (first_token_time or start_time)
        speed = (token_count / gen_time) if gen_time > 0 else 0.0

        print(f"Result: TTFT={ttft_ms:.2f}ms | Tokens={token_count} | Speed={speed:.2f} T/s | Peak RAM={monitor.peak_ram_mb:.1f}MB")

        results.append({
            "category": category,
            "prompt_summary": prompt[:16],
            "ttft_ms": round(ttft_ms, 2),
            "throughput_tokens_per_sec": round(speed, 2),
            "total_tokens": token_count,
            "peak_ram_mb": round(monitor.peak_ram_mb, 1),
            "peak_cpu_percent": round(monitor.peak_cpu_percent, 1),
            "output_preview": full_text[:60]
        })

    return results

def run_http_benchmark(server_url: str, prompts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """对正在运行的 HTTP 服务进行流式基准压测"""
    import requests
    health_url = f"{server_url}/health"
    stream_url = f"{server_url}/v1/chat/stream"

    print(f"[*] Probing server health at {health_url}...")
    try:
        resp = requests.get(health_url, timeout=3)
        if resp.status_code != 200:
            print(f"[!] Server returned HTTP {resp.status_code}. Fallback to in-process benchmark.")
            return run_in_process_benchmark(prompts)
        server_info = resp.json()
        server_pid = server_info.get("server_pid")
        print(f"[+] Connected to live server (PID: {server_pid}, Mock: {server_info.get('is_mock')})")
    except Exception as e:
        print(f"[!] Cannot connect to {server_url} ({e}). Switching to in-process benchmark...")
        return run_in_process_benchmark(prompts)

    results = []
    for item in prompts:
        category = item["category"]
        prompt = item["prompt"]
        print(f"\n--- [Testing: {category}] ---")

        payload = {
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 128,
            "stream": True
        }

        monitor = ServerResourceMonitor(target_pid=server_pid, interval=0.03)
        monitor.start()

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_text = ""

        with requests.post(stream_url, json=payload, stream=True, timeout=60) as response:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            full_text += delta
                    except Exception:
                        pass

        end_time = time.perf_counter()
        monitor.stop()
        monitor.join(timeout=1.0)

        total_time = end_time - start_time
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
        gen_time = end_time - (first_token_time or start_time)
        speed = (token_count / gen_time) if gen_time > 0 else 0.0

        print(f"Result: TTFT={ttft_ms:.2f}ms | Tokens={token_count} | Speed={speed:.2f} T/s | Peak RAM={monitor.peak_ram_mb:.1f}MB")

        results.append({
            "category": category,
            "prompt_summary": prompt[:16],
            "ttft_ms": round(ttft_ms, 2),
            "throughput_tokens_per_sec": round(speed, 2),
            "total_tokens": token_count,
            "peak_ram_mb": round(monitor.peak_ram_mb, 1),
            "peak_cpu_percent": round(monitor.peak_cpu_percent, 1),
            "output_preview": full_text[:60]
        })

    return results

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 76)
    print("           🚀 Edge LLM 推理性能与资源基准评测报告 (Benchmark Report)           ")
    print("=" * 76)
    header = f"{'测试类别':<12} | {'Prompt 摘要':<14} | {'首字延迟TTFT':<12} | {'吞吐速度':<12} | {'生成Tokens':<10} | {'峰值 RAM':<10}"
    print(header)
    print("-" * 76)
    for r in results:
        line = (
            f"{r['category']:<12} | "
            f"{r['prompt_summary']:<14} | "
            f"{r['ttft_ms']:>8.2f} ms | "
            f"{r['throughput_tokens_per_sec']:>7.2f} T/s | "
            f"{r['total_tokens']:>8} tk | "
            f"{r['peak_ram_mb']:>7.1f} MB"
        )
        print(line)
    print("=" * 76)

def main():
    parser = argparse.ArgumentParser(description="Edge LLM Benchmark Suite")
    parser.add_argument("--url", type=str, default=f"http://127.0.0.1:{SERVER_PORT}", help="Server URL")
    parser.add_argument("--standalone", action="store_true", help="Run benchmark in-process without network")
    args = parser.parse_args()

    if args.standalone:
        results = run_in_process_benchmark(BENCHMARK_PROMPTS)
    else:
        results = run_http_benchmark(args.url, BENCHMARK_PROMPTS)

    print_summary_table(results)

    output_file = project_root / "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Benchmark results exported to {output_file.resolve()}")

if __name__ == "__main__":
    main()
