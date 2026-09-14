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
