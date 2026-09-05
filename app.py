"""
Edge LLM Inference Service
FastAPI + llama.cpp (GGUF Q4_K_M) + Server-Sent Events (SSE)
"""
import os
import sys
import time
import json
import psutil
import asyncio
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Model configuration
MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH = Path(__file__).parent / "models" / MODEL_FILENAME

# Global state
llm_instance = None
start_time = time.time()

def load_llama_model():
    global llm_instance
    if not MODEL_PATH.exists():
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")
        return None
    
    try:
        from llama_cpp import Llama
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        print(f"[+] Loading model from {MODEL_PATH}...")
        print(f"[+] Hardware Allocation: CPU Threads = {cpu_threads}, GPU Layers = auto")
        
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=cpu_threads,
            n_gpu_layers=-1,
            verbose=False
        )
        print("[+] Model loaded successfully into memory.")
        return llm
    except Exception as e:
        print(f"[x] Error initializing llama.cpp model: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_instance
    llm_instance = load_llama_model()
    yield
    if llm_instance:
        del llm_instance
        print("[+] Model unloaded from memory.")

app = FastAPI(
    title="Edge LLM High-Performance Streaming Inference Service",
    description="Lightweight GGUF-quantized LLM Inference Service powered by llama.cpp & FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message text content")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history list")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    stream: bool = Field(default=True, description="Enable SSE streaming output")

HTML_UI = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Edge LLM High-Performance Streaming Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; }
    header { background: #1e293b; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
    .title-area h1 { font-size: 16px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .badge { background: #0369a1; color: #e0f2fe; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    #chat-container { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 10px; font-size: 14px; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: #2563eb; color: #fff; }
    .msg.bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
    .metrics { font-size: 11px; color: #94a3b8; margin-top: 6px; font-family: monospace; border-top: 1px dashed #334155; padding-top: 4px; }
    footer { background: #1e293b; padding: 16px 24px; border-top: 1px solid #334155; }
    .input-box { display: flex; gap: 12px; }
    input[type="text"] { flex: 1; background: #0f172a; border: 1px solid #475569; border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 14px; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }
    button { background: #0284c7; color: #fff; border: 0; border-radius: 8px; padding: 0 20px; font-weight: 600; cursor: pointer; transition: background .15s; }
    button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <header>
    <div class="title-area">
      <h1>Edge LLM Streaming Console <span class="badge">Qwen2.5-1.5B Q4_K_M</span></h1>
    </div>
    <div style="font-size:12px; color:#94a3b8;">FastAPI + llama.cpp C++ Engine (SSE)</div>
  </header>
  <div id="chat-container">
    <div class="msg bot">你好！我是运行在你本地普通 CPU 上的轻量化大语言模型 (Qwen2.5-1.5B INT4)。支持毫秒级流式打字机响应，请输入你的问题体验端侧推理！</div>
  </div>
  <footer>
    <div class="input-box">
      <input type="text" id="userInput" placeholder="输入问题测试流式响应与 TTFT / 吞吐性能..." onkeydown="if(event.key==='Enter') sendMsg()" />
      <button onclick="sendMsg()">发送 (Send)</button>
    </div>
  </footer>
  <script>
    async function sendMsg() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      const container = document.getElementById('chat-container');
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = text;
      container.appendChild(userDiv);

      const botDiv = document.createElement('div');
      botDiv.className = 'msg bot';
      const textSpan = document.createElement('span');
      botDiv.appendChild(textSpan);
      container.appendChild(botDiv);
      container.scrollTop = container.scrollHeight;

      const t0 = performance.now();
      let ttft = null;
      let tokenCount = 0;

      const resp = await fetch('/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: text }],
          max_tokens: 512,
          stream: true
        })
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const raw = line.substring(6).trim();
            if (raw === '[DONE]') break;
            try {
              const data = JSON.parse(raw);
              const delta = data.choices[0]?.delta?.content || '';
              if (delta) {
                if (ttft === null) ttft = performance.now() - t0;
                tokenCount++;
                textSpan.textContent += delta;
                container.scrollTop = container.scrollHeight;
              }
            } catch(e) {}
          }
        }
      }

      const totalTimeSec = (performance.now() - t0) / 1000.0;
      const speed = tokenCount > 0 ? (tokenCount / totalTimeSec).toFixed(1) : 0;
      const metricsDiv = document.createElement('div');
      metricsDiv.className = 'metrics';
      metricsDiv.textContent = `⚡ TTFT: ${ttft ? ttft.toFixed(0) : 0}ms | Tokens: ${tokenCount} | Speed: ${speed} tokens/s`;
      botDiv.appendChild(metricsDiv);
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=HTML_UI)

@app.get("/health")
def health_check():
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    system_ram_percent = psutil.virtual_memory().percent
    uptime_sec = round(time.time() - start_time, 2)
    
    return {
        "status": "healthy" if llm_instance is not None else "model_not_loaded",
        "model_name": MODEL_FILENAME,
        "model_loaded": llm_instance is not None,
        "uptime_seconds": uptime_sec,
        "server_pid": os.getpid(),
        "memory": {
            "process_ram_rss_mb": round(ram_mb, 2),
            "system_ram_percent": system_ram_percent
        },
        "system": {
            "cpu_cores_logical": os.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=None)
        }
    }

def format_prompt_from_messages(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatCompletionRequest):
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llama_model()
        if llm_instance is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

    prompt = format_prompt_from_messages(request.messages)
    req_id = f"chatcmpl-{int(time.time()*1000)}"

    async def event_generator():
        try:
            stream_iter = llm_instance(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            for chunk in stream_iter:
                token_text = chunk["choices"][0]["text"]
                data_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "qwen2.5-1.5b-instruct-q4_k_m",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": chunk["choices"][0].get("finish_reason")
                    }]
                }
                yield {"data": json.dumps(data_payload, ensure_ascii=False)}
                await asyncio.sleep(0.001)

            yield {"data": "[DONE]"}
        except Exception as e:
            error_payload = {"error": str(e)}
            yield {"data": json.dumps(error_payload)}

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
