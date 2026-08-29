"""
Edge LLM Inference Service
FastAPI + llama.cpp (GGUF Q4_K_M) + Server-Sent Events (SSE)
"""
import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Model configuration
MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH = Path(__file__).parent / "models" / MODEL_FILENAME

# Global state
llm_instance = None
start_time = time.time()

def load_llama_model():
    """Initialize the llama.cpp model engine with optimal hardware config"""
    global llm_instance
    if not MODEL_PATH.exists():
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")
        print("[!] Please run `python download_model.py` first to pull the model.")
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

# CORS middleware for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender: system, user, assistant")
    content: str = Field(..., description="Message text content")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history list")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    stream: bool = Field(default=True, description="Enable SSE streaming output")

@app.get("/health")
def health_check():
    """Health check and real-time resource utilization monitor"""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    system_ram_percent = psutil.virtual_memory().percent
    uptime_sec = round(time.time() - start_time, 2)
    
    return {
        "status": "healthy" if llm_instance is not None else "model_not_loaded",
        "model_name": MODEL_FILENAME,
        "model_loaded": llm_instance is not None,
        "uptime_seconds": uptime_sec,
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
    """Format chat messages according to Qwen2.5 ChatML format"""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatCompletionRequest):
    """
    Standard Server-Sent Events (SSE) Streaming Inference Endpoint
    """
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llama_model()
        if llm_instance is None:
            raise HTTPException(status_code=503, detail="Model is not loaded. Please run `python download_model.py`.")

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

            # Send standard SSE termination signal
            yield {"data": "[DONE]"}
        except Exception as e:
            error_payload = {"error": str(e)}
            yield {"data": json.dumps(error_payload)}

    return EventSourceResponse(event_generator())

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint supporting both stream and non-stream"""
    if request.stream:
        return await chat_stream(request)
    
    global llm_instance
    if llm_instance is None:
        llm_instance = load_llama_model()
        if llm_instance is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

    prompt = format_prompt_from_messages(request.messages)
    output = llm_instance(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen2.5-1.5b-instruct-q4_k_m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output["choices"][0]["text"]
            },
            "finish_reason": output["choices"][0].get("finish_reason", "stop")
        }],
        "usage": output.get("usage", {})
    }

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting Edge LLM Inference Server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
