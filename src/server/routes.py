"""
API Route Handlers (SSE Streaming & Health Probes)
===================================================
"""
import os
import time
import json
import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from src.core.schemas import ChatCompletionRequest, ChatMessage, HealthResponse
from src.core.config import DEFAULT_MODEL_FILENAME
from src.engine.llm_engine import get_engine
from src.utils.monitor import get_system_metrics
from src.server.templates import HTML_CONSOLE_PAGE

router = APIRouter()
_server_start_time = time.time()

def format_prompt_from_messages(messages: List[ChatMessage]) -> str:
    """ChatML 格式化函数"""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

@router.get("/", response_class=HTMLResponse, summary="Web 交互控制台")
def index_page():
    return HTMLResponse(content=HTML_CONSOLE_PAGE)

@router.get("/health", response_model=HealthResponse, summary="服务健康与硬件资源监控探针")
def health_check():
    engine = get_engine()
    sys_metrics = get_system_metrics()
    uptime_sec = round(time.time() - _server_start_time, 2)

    status_str = "healthy"
    if engine.is_mock:
        status_str = "mock_mode"
    elif engine.instance is None:
        status_str = "model_not_loaded"

    return {
        "status": status_str,
        "model_name": engine.model_path.name if engine.model_path else DEFAULT_MODEL_FILENAME,
        "model_loaded": engine.instance is not None,
        "is_mock": engine.is_mock,
        "uptime_seconds": uptime_sec,
        "server_pid": os.getpid(),
        "memory": sys_metrics["memory"],
        "system": sys_metrics["system"]
    }

@router.post("/v1/chat/stream", summary="OpenAI 兼容 SSE 流式推理接口")
async def chat_stream(request: ChatCompletionRequest):
    engine = get_engine()
    if engine.instance is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    prompt = format_prompt_from_messages(request.messages)
    req_id = f"chatcmpl-{int(time.time()*1000)}"

    async def event_generator():
        try:
            # 迭代引擎的生成器
            generator = engine.stream_inference(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

            for item in generator:
                token_text = item["token"]
                finish_reason = item["finish_reason"]

                data_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": engine.model_path.name if engine.model_path else DEFAULT_MODEL_FILENAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": finish_reason
                    }]
                }
                yield {"data": json.dumps(data_payload, ensure_ascii=False)}
                await asyncio.sleep(0.001)

            yield {"data": "[DONE]"}
        except Exception as e:
            error_payload = {"error": str(e)}
            yield {"data": json.dumps(error_payload, ensure_ascii=False)}

    return EventSourceResponse(event_generator())
