"""
Global Configuration & Hardware Allocation Policy
==================================================
[架构设计决策与量化标尺]
1. 硬件瓶颈定位：
   端侧 CPU 推理核心瓶颈在于【内存总线带宽 (Memory Bandwidth Bound)】而非算力饥饿。
   自回归大语言模型在生成阶段（Decode）每产出一个 Token，均需对全量模型参数矩阵从主板内存搬运至 CPU L3 缓存。
   以 Qwen2.5-1.5B-Instruct 为例：
   - 原生 FP16 体积为 3.09 GB，受限于双通道 DDR 内存带宽，解码吞吐仅为 ~6.2 tokens/s；
   - 经 GGUF Q4_K_M 混合精度量化后，模型权重剧降至 934.69 MiB（压缩率达 70.3%），动态 KV Cache 仅 7.00 MiB；
   - 总线访存压力削减逾 70%，在普通 8 线程 CPU 上流式 Decode 吞吐大幅跃升至 26.56 tokens/s（提升 4.2 倍），
     Prefill 阶段达到 186.69 tokens/s。

2. 混合精度选型 (Q4_K_M vs Q4_0):
   采用非均匀混合精度策略：
   - 注意力层 (Attention: Q, K, V): 关系长程语义关联与自注意力矩阵稳定，保留 6-bit 较高精度；
   - 前馈网络层 (FFN: Gate, Up, Down): 参数占比超 60% 但量化容错率高，深度压缩至 4-bit；
   在实测困惑度（PPL）恶化小于 0.045 的极微损失下，榨干端侧内存带宽极限。
"""
import os
from pathlib import Path
from typing import Optional, List

# 路径基准定义
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
KNOWLEDGE_DIR = PROJECT_ROOT / "src" / "knowledge"

DEFAULT_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

def resolve_model_path() -> Path:
    """动态解析模型权重绝对路径（支持环境变量与候选目录自动嗅探）"""
    env_path = os.getenv("EDGE_LLM_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).resolve()

    candidates: List[Path] = [
        MODELS_DIR / DEFAULT_MODEL_FILENAME,
        MODELS_DIR / "qwen_lat_q4_k_m.gguf",
        MODELS_DIR / "qwen_lora_merged.Q4_K_M.gguf",
        PROJECT_ROOT / DEFAULT_MODEL_FILENAME,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (MODELS_DIR / DEFAULT_MODEL_FILENAME).resolve()

DEFAULT_MODEL_PATH = resolve_model_path()

# 推理硬件资源配置
# CPU 线程分配：默认绑定 4 线程（与主流移动端 4 大核对齐），规避 NUMA 跨节点自旋颠簸
DEFAULT_CPU_THREADS = int(os.getenv("EDGE_LLM_THREADS", "4"))
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(DEFAULT_CPU_THREADS)
DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_GPU_LAYERS = -1  # 0 为纯 CPU 模式，-1 为自动尝试 GPU/NPU 卸载（如 Metal/Vulkan/CUDA 支持）

# 默认预热系统提示词 (用于服务初始化阶段的 KV Cache 预热计算)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible while being safe."
)

# 业务规则与记忆存储默认路径
DEFAULT_RULES_PATH = KNOWLEDGE_DIR / "biomechanics_rules.json"
DEFAULT_MEMORY_DB = PROJECT_ROOT / "posture_user_memory.db"

# 网络服务与网关配置
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))

# 环境运行模式 (支持无模型文件时的自适应 Mock 验证，确保在测试/面试演示中 30 秒跑通)
FORCE_MOCK = os.getenv("FORCE_MOCK", "false").lower() in ("true", "1", "yes")
ENABLE_MOCK_FALLBACK = True
