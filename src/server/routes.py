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

from src.core.schemas import (
    ChatCompletionRequest, ChatMessage, HealthResponse,
    MotionCoachRequest, MotionCoachResponse
)
from src.core.config import DEFAULT_MODEL_FILENAME
from src.engine.llm_engine import get_engine
from src.utils.monitor import get_system_metrics
from src.server.templates import HTML_CONSOLE_PAGE
from src.pipelines.lat_pulldown_pipeline import LatPulldownEdgePipeline
from src.pipelines.posture_coach import EdgePostureCoach

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

@router.post("/v1/motion/coach", response_model=MotionCoachResponse, summary="端侧运动力学质检与即时纠错接口")
def motion_coach(request: MotionCoachRequest):
    """
    结合参数化 RAG 规则引擎与端侧大模型底座，执行亚秒级姿态质检与微口令提炼
    """
    engine = get_engine()
    if engine.instance is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    t0 = time.perf_counter()
    action = request.action.lower()

    if "lat" in action or "pull" in action:
        pipeline = LatPulldownEdgePipeline(engine=engine)
        if request.user_injury:
            pipeline.setup_user(user_id=request.user_id, injury=request.user_injury)

        torso_angle = float(request.metrics.get("torso_angle", 14.0))
        scapula_ratio = float(request.metrics.get("scapula_ratio", 0.30))
        pull_pos_ratio = float(request.metrics.get("pull_pos_ratio", 0.12))
        grip_ratio = float(request.metrics.get("grip_ratio", 1.3))

        res = pipeline.process_motion_event(
            user_id=request.user_id,
            torso_angle=torso_angle,
            scapula_ratio=scapula_ratio,
            pull_pos_ratio=pull_pos_ratio,
            grip_ratio=grip_ratio
        )
        total_cost = (time.perf_counter() - t0) * 1000
        return {
            "action": "高位下拉",
            "variant": res.get("grip_variant"),
            "is_compliant": res.get("is_compliant", True),
            "faults": res.get("rule_faults", []),
            "coach_cue": res.get("coach_cue", "保持核心收紧！"),
            "raw_output": res.get("raw_output"),
            "llm_latency_ms": res.get("llm_latency_ms", 0.0),
            "total_latency_ms": round(total_cost, 2)
        }
    else:
        coach = EdgePostureCoach(engine=engine)
        if request.user_injury:
            coach.setup_user_profile(user_id=request.user_id, injury=request.user_injury)

        res = coach.analyze_and_coach(
            user_id=request.user_id,
            metrics=request.metrics,
            action="squat"
        )
        total_cost = (time.perf_counter() - t0) * 1000
        return {
            "action": "深蹲",
            "variant": "标准深蹲",
            "is_compliant": res.get("verdict") == "达标",
            "faults": [res["core_defect"]] if res.get("core_defect") and res["core_defect"] != "无明显缺陷" else [],
            "coach_cue": res.get("coach_cue", "保持平稳！"),
            "raw_output": res.get("raw_output"),
            "llm_latency_ms": res.get("llm_cost_ms", 0.0),
            "total_latency_ms": round(total_cost, 2)
        }
