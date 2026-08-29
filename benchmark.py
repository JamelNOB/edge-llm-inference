"""
Edge LLM Automated Performance & Benchmark Suite
Measures: TTFT (ms), Decoding Throughput (tokens/s), Peak RAM (MB), Peak CPU (%)
"""
import os
import sys
import time
import json
import psutil
import requests
import threading
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SERVER_URL = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
STREAM_ENDPOINT = f"{SERVER_URL}/v1/chat/stream"

# 测试基准用例集（覆盖短问答、技术概念、代码生成与长文本总结）
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
        "category": "Code Generation",
        "prompt": "用 Python 实现一个快速排序算法函数并给出简要测试用例。"
    },
    {
        "category": "Analytical Summary",
        "prompt": "对比端侧大模型部署 (Edge LLM) 与云端 API 调用在成本、隐私、延迟与吞吐方面的优劣势。"
    }
]

class ServerResourceMonitor(threading.Thread):
    """后台服务端进程资源监控采样线程 (RAM Peak, CPU Peak)"""
    def __init__(self, target_pid: int = None, interval: float = 0.05):
        super().__init__()
        self.target_pid = target_pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_ram_mb = 0.0
        self.peak_cpu_percent = 0.0
        try:
            self.process = psutil.Process(target_pid) if target_pid else psutil.Process()
        except Exception:
            self.process = psutil.Process()

    def run(self):
        while not self.stop_event.is_set():
            try:
                ram_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent(interval=None)
                if ram_mb > self.peak_ram_mb:
                    self.peak_ram_mb = ram_mb
                if cpu_percent > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu_percent
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def run_single_benchmark(category: str, prompt: str, server_pid: int) -> Dict[str, Any]:
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True
    }

    monitor = ServerResourceMonitor(target_pid=server_pid, interval=0.05)
    monitor.start()

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = ""

    try:
        with requests.post(STREAM_ENDPOINT, json=payload, stream=True, timeout=120) as response:
            if response.status_code != 200:
                monitor.stop()
                monitor.join()
                return {"error": f"HTTP {response.status_code}"}

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                token_count += 1
                                full_text += content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        monitor.stop()
        monitor.join()
        return {"error": str(e)}

    end_time = time.perf_counter()
    monitor.stop()
    monitor.join()

    total_latency_s = end_time - start_time
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
    decode_time_s = end_time - (first_token_time or start_time)
    throughput = (token_count / decode_time_s) if decode_time_s > 0.001 else 0.0

    return {
        "category": category,
        "prompt_snippet": prompt[:20] + "...",
        "ttft_ms": round(ttft_ms, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "total_tokens": token_count,
        "total_duration_sec": round(total_latency_s, 2),
        "peak_ram_mb": round(monitor.peak_ram_mb, 2),
        "peak_cpu_percent": round(monitor.peak_cpu_percent, 1)
    }

def print_pretty_table(results: List[Dict[str, Any]]):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title="🚀 Edge LLM 推理性能与资源基准评测报告 (Benchmark Report)", header_style="bold magenta")
        table.add_column("测试类别", style="cyan", justify="left")
        table.add_column("Prompt 摘要", style="white", justify="left")
        table.add_column("首字延迟 (TTFT)", style="green", justify="right")
        table.add_column("吞吐速度", style="bold yellow", justify="right")
        table.add_column("生成 Tokens", style="blue", justify="right")
        table.add_column("总耗时", style="white", justify="right")
        table.add_column("后端峰值 RAM", style="red", justify="right")

        for r in results:
            if "error" in r:
                table.add_row(r.get("category", "N/A"), "Error", "N/A", "N/A", "N/A", "N/A", "N/A")
            else:
                table.add_row(
                    r["category"],
                    r["prompt_snippet"],
                    f"{r['ttft_ms']} ms",
                    f"{r['throughput_tokens_per_sec']} T/s",
                    str(r["total_tokens"]),
                    f"{r['total_duration_sec']} s",
                    f"{r['peak_ram_mb']} MB"
                )

        console.print(table)
    except Exception:
        print("\n" + "=" * 90)
        print(f"{'类别':<18} | {'TTFT(ms)':<10} | {'吞吐(T/s)':<10} | {'Tokens':<8} | {'总耗时(s)':<10} | {'峰值内存(MB)'}")
        print("-" * 90)
        for r in results:
            if "error" not in r:
                print(f"{r['category']:<18} | {r['ttft_ms']:<10} | {r['throughput_tokens_per_sec']:<10} | {r['total_tokens']:<8} | {r['total_duration_sec']:<10} | {r['peak_ram_mb']}")
        print("=" * 90 + "\n")

def main():
    print("=" * 70)
    print("🎯 Edge-LLM 自动化性能评测套件 (Automated Benchmark Suite)")
    print(f"🔗 测试服务地址: {SERVER_URL}")
    print("=" * 70)

    server_pid = None
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            server_pid = data.get("server_pid", os.getpid())
            print(f"[+] 成功连接推理服务 (PID: {server_pid}, 模型: {data.get('model_name')})")
        else:
            print("[x] 后端服务未就绪，请先运行: python app.py")
            return
    except Exception:
        print("[x] 无法连接到服务，请确认后端是否已启动: python app.py")
        return

    print("[*] 开始执行基准评测，共 4 组场景...\n")
    results = []
    
    for i, item in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"[{i}/{len(BENCHMARK_PROMPTS)}] 正在评测: {item['category']} ... ", end="", flush=True)
        res = run_single_benchmark(item["category"], item["prompt"], server_pid)
        results.append(res)
        if "error" in res:
            print(f"❌ 失败 ({res['error']})")
        else:
            print(f"✅ 完成 (TTFT: {res['ttft_ms']}ms | Speed: {res['throughput_tokens_per_sec']} tokens/s | RAM: {res['peak_ram_mb']}MB)")

    print("\n" + "=" * 70)
    print_pretty_table(results)

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_ttft = sum(r["ttft_ms"] for r in valid_results) / len(valid_results)
        avg_throughput = sum(r["throughput_tokens_per_sec"] for r in valid_results) / len(valid_results)
        max_ram = max(r["peak_ram_mb"] for r in valid_results)

        print("📈 【评测综合汇总 (Executive Summary)】:")
        print(f"   • 平均首字响应延迟 (Avg TTFT):       {avg_ttft:.2f} ms")
        print(f"   • 平均生成吞吐速率 (Avg Throughput): {avg_throughput:.2f} tokens/sec")
        print(f"   • 运行时内存峰值 (Max RAM RSS):     {max_ram:.2f} MB (~{max_ram/1024:.2f} GB)")
        print("=" * 70)

        with open("benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_ttft_ms": round(avg_ttft, 2),
                    "avg_throughput_tokens_per_sec": round(avg_throughput, 2),
                    "max_ram_mb": round(max_ram, 2)
                },
                "details": valid_results
            }, f, indent=2, ensure_ascii=False)
        print("💾 评测数据已保存至: benchmark_results.json\n")

if __name__ == "__main__":
    main()
