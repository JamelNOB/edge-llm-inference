"""
FastAPI Application Entry & Middleware Assembly
================================================
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.server.routes import router
from src.engine.llm_engine import get_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动初始化：加载权重并预热 KV Cache
    engine = get_engine()
    yield
    # 服务优雅关机：释放内存/显存资源
    engine.close()

def create_app() -> FastAPI:
    app = FastAPI(
        title="Edge LLM High-Performance Streaming Inference Service",
        description="Lightweight GGUF-quantized LLM Inference Service powered by llama.cpp & FastAPI",
        version="1.1.0",
        lifespan=lifespan
    )

    # 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载路由
    app.include_router(router)
    return app

app = create_app()
