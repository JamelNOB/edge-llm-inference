"""
Pydantic Data Schemas & API Contracts
======================================
Defines strict request/response data models for validation and OpenAPI generation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str = Field(..., description="Content text of the message")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    stream: bool = Field(default=True, description="Whether to stream response tokens via SSE")

class MemoryMetrics(BaseModel):
    process_ram_rss_mb: float = Field(..., description="Process RSS Memory in MB")
    system_ram_percent: float = Field(..., description="Overall system RAM usage percentage")

class SystemMetrics(BaseModel):
    cpu_cores_logical: int = Field(..., description="Total logical CPU cores")
    cpu_percent: float = Field(..., description="Instantaneous CPU usage percentage")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('healthy', 'mock_mode', 'model_not_loaded')")
    model_name: str = Field(..., description="Loaded GGUF model filename or identifier")
    model_loaded: bool = Field(..., description="Whether inference engine is ready")
    is_mock: bool = Field(..., description="Whether running in mock simulation mode")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    server_pid: int = Field(..., description="Process ID of server")
    memory: MemoryMetrics
    system: SystemMetrics

class MotionCoachRequest(BaseModel):
    action: str = Field(default="lat_pulldown", description="运动动作名称 (如 'lat_pulldown', 'squat')")
    user_id: str = Field(default="demo_user", description="学员唯一标识")
    metrics: Dict[str, Any] = Field(..., description="前端解算出的多维连续运动力学特征字典")
    user_injury: Optional[str] = Field(default=None, description="学员突发生理状态或伤病史 (支持 TTL 时效)")

class MotionCoachResponse(BaseModel):
    action: str = Field(..., description="动作名称")
    variant: Optional[str] = Field(default=None, description="动作变体 (如宽握/窄握)")
    is_compliant: bool = Field(..., description="力学规则判定是否合规")
    faults: List[str] = Field(default_factory=list, description="力学违规项明细清单")
    coach_cue: str = Field(..., description="端侧大模型提炼的 8~12 字高穿透力即时纠错短口令")
    raw_output: Optional[str] = Field(default=None, description="端侧大模型原始输出")
    llm_latency_ms: float = Field(..., description="大模型单次前向推理时延 (ms)")
    total_latency_ms: float = Field(..., description="全链路端到端总时延 (ms)")
