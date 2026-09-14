"""
Edge LLM Streaming Inference Client
====================================
Terminal Typewriter Test with TTFT and Throughput Metrics
"""
import sys
import time
import json
import requests
from pathlib import Path

# 将项目根目录添加进 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import SERVER_PORT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
STREAM_ENDPOINT = f"{SERVER_URL}/v1/chat/stream"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def check_server_health() -> bool:
    """Verify backend server health and model status"""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            print(f"{GREEN}[+] 服务状态正常 (Status: {data.get('status')})!{RESET}")
            print(f"    - 加载模型: {CYAN}{data.get('model_name')}{RESET} (Mock: {data.get('is_mock')})")
            print(f"    - 进程内存占用 (RAM RSS): {CYAN}{data.get('memory', {}).get('process_ram_rss_mb')} MB{RESET}")
            print(f"    - CPU 逻辑核心数: {CYAN}{data.get('system', {}).get('cpu_cores_logical')}{RESET}")
            return True
        else:
            print(f"{YELLOW}[!] 服务返回状态码: {resp.status_code}{RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{YELLOW}[x] 无法连接到推理服务: {SERVER_URL}{RESET}")
        return False

def stream_chat(user_prompt: str):
    """发送请求并以打字机流式输出展示"""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful and concise AI assistant."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True
    }

    print(f"\n{BOLD}{CYAN}=== Edge LLM 流式推理打字机效果测试 ==={RESET}")
    print(f"{BOLD}用户 Prompt:{RESET} {user_prompt}\n")
    print(f"{BOLD}{GREEN}模型回复:{RESET} ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""

    try:
        with requests.post(STREAM_ENDPOINT, json=payload, stream=True, timeout=60) as response:
            if response.status_code != 200:
                print(f"\n[x] 请求失败，HTTP 状态码 {response.status_code}: {response.text}")
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # 解析 SSE 协议格式 (data: {...})
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
                                full_response += content
                                sys.stdout.write(content)
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        continue

        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 计算核心性能指标
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        gen_time = end_time - (first_token_time or start_time)
        tokens_per_sec = (token_count / gen_time) if gen_time > 0 else 0

        print(f"\n\n{MAGENTA}" + "-" * 50 + f"{RESET}")
        print(f"{BOLD}📊 性能统计 (Performance Metrics):{RESET}")
        print(f"  • 首字时延 (TTFT):    {CYAN}{ttft_ms:.2f} ms{RESET} (KV Cache 预热保障)")
        print(f"  • 生成 Token 数量:   {CYAN}{token_count}{RESET} tokens")
        print(f"  • 解码吞吐速度:       {GREEN}{tokens_per_sec:.2f} tokens/s{RESET}")
        print(f"  • 总响应耗时:         {CYAN}{total_time:.2f} s{RESET}")
        print(f"{MAGENTA}" + "-" * 50 + f"{RESET}\n")

    except Exception as e:
        print(f"\n[x] 请求异常: {e}")

if __name__ == "__main__":
    if check_server_health():
        prompt = "请简述端侧大模型为什么在量化压缩后在普通 CPU 上的生成吞吐速度反而更快？"
        stream_chat(prompt)
    else:
        print(f"\n{YELLOW}[!] 提示: 请先运行 'python main.py' 启动服务端。{RESET}")
